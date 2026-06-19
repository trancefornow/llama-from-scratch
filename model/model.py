import torch
import torch.nn as nn

from model.rmsnorm import RMSNorm
from model.transformer import LlamaDecoderLayer


class Llama(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.max_position_embeddings = max_position_embeddings

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(
                    dim=dim,
                    n_heads=n_heads,
                    max_position_embeddings=max_position_embeddings,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def setup_cache(self, max_batch_size: int, max_seq_len: int) -> None:
        if max_seq_len > self.max_position_embeddings:
            raise ValueError(
                f"max_seq_len={max_seq_len} exceeds max_position_embeddings="
                f"{self.max_position_embeddings}"
            )

        for layer in self.layers:
            layer.setup_cache(max_batch_size=max_batch_size, max_seq_len=max_seq_len)

    def clear_cache(self) -> None:
        for layer in self.layers:
            layer.clear_cache()

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> torch.Tensor:
        h = self.tok_embeddings(tokens)

        for layer in self.layers:
            h = layer(h, start_pos=start_pos, use_cache=use_cache)

        h = self.norm(h)
        logits = self.output(h)

        return logits
