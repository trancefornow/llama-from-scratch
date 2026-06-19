import torch
import torch.nn as nn

from model.attention import LlamaAttention
from model.rmsnorm import RMSNorm
from model.swiglu import FeedForward


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        multiple_of: int = 256,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()

        self.attention_norm = RMSNorm(dim)
        self.attention = LlamaAttention(
            dim=dim,
            n_heads=n_heads,
            max_position_embeddings=max_position_embeddings,
        )

        self.ffn_norm = RMSNorm(dim)

        hidden_dim = int(2 * 4 * dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=hidden_dim)

    def setup_cache(self, max_batch_size: int, max_seq_len: int) -> None:
        self.attention.setup_cache(max_batch_size=max_batch_size, max_seq_len=max_seq_len)

    def clear_cache(self) -> None:
        self.attention.clear_cache()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> torch.Tensor:
        h = x + self.attention(
            self.attention_norm(x),
            start_pos=start_pos,
            use_cache=use_cache,
        )
        out = h + self.feed_forward(self.ffn_norm(h))

        return out
