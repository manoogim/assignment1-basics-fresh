from einops import einsum
from torch import nn
import torch

"""
Construct the RMSNorm module. This function should accept the following parameters:
    d_model: int  Hidden dimension of the model
    eps: float = 1e-5  Epsilon value for numerical stability
    device: torch.device | None = None  Device to store the parameters on
    dtype: torch.dtype | None = None  Data type of the parameters
"""
class MyRmsNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        w = torch.ones(d_model, device=device, dtype=dtype)
        self.gamma = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        
        """
        assert x.shape[-1] == self.d_model, f'Input tensor last dim is wrong: {x.shape[-1]}, expected: {self.d_model}'
        in_type = x.dtype
        x = x.to(torch.float32)
        # Compute mean of squared values along the last dimension (d_model), keep dims for broadcasting
        variance = x.pow(2).mean(dim=-1, keepdim=True)

        normalizer = torch.sqrt(variance + self.eps) 
        result = (x / normalizer) * self.gamma
        return result.to(in_type)