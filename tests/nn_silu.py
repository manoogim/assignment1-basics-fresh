from einops import einsum
from jaxtyping import Float
from torch import Tensor, nn
import torch

class MySiluFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        w1 = torch.empty((d_ff, d_model), device=device, dtype=dtype)
        w2 = torch.empty((d_model, d_ff), device=device, dtype=dtype)

        std_dev = (2.0 / (d_model + d_ff)) ** 0.5
        nn.init.trunc_normal_(w1, 0, std_dev, -3*std_dev, 3*std_dev)
        nn.init.trunc_normal_(w2, 0, std_dev, -3*std_dev, 3*std_dev)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)

    def forward(self, x: Float[Tensor, "... d_model"]):
        w1x = einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu = w1x * torch.sigmoid(w1x)
        result = einsum(self.w2, silu, "d_model d_ff, ... d_ff -> ... d_model")
        return result
