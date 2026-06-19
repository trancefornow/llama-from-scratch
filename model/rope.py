import torch
import torch.nn as nn
from typing import Tuple


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        positions = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        end_pos = start_pos + seq_len
        if end_pos > self.max_position_embeddings:
            raise ValueError(
                f"RoPE position {end_pos} exceeds max_position_embeddings="
                f"{self.max_position_embeddings}"
            )

        cos = self.cos_cached[start_pos:end_pos, :].to(device=q.device, dtype=q.dtype)
        sin = self.sin_cached[start_pos:end_pos, :].to(device=q.device, dtype=q.dtype)

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed
