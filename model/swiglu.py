import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()

        #w1, w2, w3
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        self.silu = nn.SELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #( Swish(xW1) * xW3 ) W2
        gate = self.silu(self.w1(x))
        current_value = self.w3(x)
        return self.w2(gate * current_value)