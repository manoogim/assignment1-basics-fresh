from einops import einsum
from jaxtyping import Float
from torch import Tensor, nn
import torch


class MySwiglu(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device = None, dtype = None):
        super().__init__()
        w1 = torch.empty((d_ff, d_model),device=device, dtype=dtype)
        w2 = torch.empty((d_model, d_ff),device=device, dtype=dtype)
        w3 = torch.empty((d_ff, d_model),device=device, dtype=dtype)

        std_dev = (2.0 / (d_model + d_ff)) ** 0.5
        nn.init.trunc_normal_(w1, 0, std_dev, -3*std_dev, 3*std_dev)
        nn.init.trunc_normal_(w2, 0, std_dev, -3*std_dev, 3*std_dev)
        nn.init.trunc_normal_(w3, 0, std_dev, -3*std_dev, 3*std_dev)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.w3 = nn.Parameter(w3)

    def forward(self, x: Float[Tensor, '... d_model']):
        """
        Implement the SwiGLU feed-forward network, composed of a SiLU activation function and a GLU.
        FFN(x) = SwiGLU(x,W1,W2,W3) = W2(SiLU(W1x) ⊙ W3x)
        """
        w1x = einsum(self.w1, x, 'd_ff d_model, ... d_model -> ... d_ff')
        w3x = einsum(self.w3, x, 'd_ff d_model, ... d_model -> ... d_ff')
        silu = w1x * torch.sigmoid(w1x)
        element_wise = silu * w3x
        result = einsum(self.w2, element_wise, 'd_model d_ff, ... d_ff -> ... d_model')
        return result

