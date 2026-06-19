import math
from typing import Optional

import torch
import torch.nn as nn

from model.rope import RotaryEmbedding


class LlamaAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.max_batch_size = 0
        self.max_seq_len = 0

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
        )

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    def setup_cache(
        self,
        max_batch_size: int,
        max_seq_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if max_seq_len > self.rope.max_position_embeddings:
            raise ValueError(
                f"max_seq_len={max_seq_len} exceeds RoPE capacity "
                f"{self.rope.max_position_embeddings}"
            )

        device = device if device is not None else self.wk.weight.device
        dtype = dtype if dtype is not None else self.wk.weight.dtype
        cache_shape = (max_batch_size, self.n_heads, max_seq_len, self.head_dim)

        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.cache_k = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.cache_v = torch.zeros(cache_shape, device=device, dtype=dtype)

    def clear_cache(self) -> None:
        if self.cache_k is not None:
            self.cache_k.zero_()
        if self.cache_v is not None:
            self.cache_v.zero_()

    def _ensure_cache_ready(
        self,
        batch_size: int,
        end_pos: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if self.cache_k is None or self.cache_v is None:
            raise RuntimeError(
                "KV cache is not initialized. Call setup_cache(max_batch_size, "
                "max_seq_len) before running with use_cache=True."
            )
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"batch_size={batch_size} exceeds cache max_batch_size="
                f"{self.max_batch_size}"
            )
        if end_pos > self.max_seq_len:
            raise ValueError(
                f"end_pos={end_pos} exceeds cache max_seq_len={self.max_seq_len}"
            )
        if self.cache_k.device != device or self.cache_k.dtype != dtype:
            raise RuntimeError(
                "KV cache device/dtype does not match current attention tensors. "
                "Call setup_cache again after moving or casting the model."
            )

    def _causal_mask(
        self,
        seq_len: int,
        total_len: int,
        start_pos: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if seq_len == 1:
            return None

        mask = torch.full((seq_len, total_len), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=start_pos + 1)
        return mask.view(1, 1, seq_len, total_len)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> torch.Tensor:
        if start_pos < 0:
            raise ValueError("start_pos must be non-negative")

        batch_size, seq_len, dim = x.shape
        end_pos = start_pos + seq_len

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k, seq_len=seq_len, start_pos=start_pos)

        if use_cache:
            if self.training and torch.is_grad_enabled():
                raise RuntimeError("KV cache is inference-only; call eval() before use_cache=True.")
            self._ensure_cache_ready(batch_size, end_pos, k.device, k.dtype)
            with torch.no_grad():
                self.cache_k[:batch_size, :, start_pos:end_pos, :].copy_(k)
                self.cache_v[:batch_size, :, start_pos:end_pos, :].copy_(v)
            keys = self.cache_k[:batch_size, :, :end_pos, :]
            values = self.cache_v[:batch_size, :, :end_pos, :]
            mask_start_pos = start_pos
        else:
            keys = k
            values = v
            mask_start_pos = 0

        scores = torch.matmul(q, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self._causal_mask(
            seq_len=seq_len,
            total_len=keys.size(2),
            start_pos=mask_start_pos,
            device=x.device,
            dtype=scores.dtype,
        )
        if mask is not None:
            scores = scores + mask

        scores = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)

        return self.wo(output)
