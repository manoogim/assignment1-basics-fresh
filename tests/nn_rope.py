import math
from jaxtyping import Float
import torch
from torch import nn
"""
Deliverable: Implement a class RotaryPositionalEmbedding that applies RoPE to the input 
tensor.
The following interface is recommended:
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) 
Construct the RoPE module and create buffers if needed.
    theta: float  Θ value for the RoPE
    d_k: int  dimension of query and key vectors
    max_seq_len: int  Maximum sequence length that will be input
    device: torch.device | None = None  Device to store the buffer on

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor 
Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note 
that you should tolerate 𝑥 with an arbitrary number of batch dimensions. 
You should assume that the token positions are a tensor of shape (..., seq_len) specifying the token positions 
of  𝑥 along the sequence dimension.

"""
def my_rotation (alpha, device=None, dtype=None):
    c = math.cos(alpha)
    s = math.sin(alpha)
    rot = torch.tensor([[c, -s], [s, c]], device=device, dtype=dtype)
    return rot

def precompute_rope_blocks(theta_const: float, d_k: int, max_seq_len: int, device=None, dtype=None):
    half = d_k // 2
    frequencies = [theta_const ** (-2 * k / d_k) for k in range(d_k //2)]
    ropes = torch.empty(max_seq_len, d_k, 2, 2, device = device, dtype = dtype)
    for pos in range (max_seq_len):
        alphas = [pos * freq for freq in frequencies]
        for k in range (half):
            alpha = alphas[k]
            ropes[pos, k] = my_rotation(alpha)
    return ropes

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta_const: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d = d_k
        self.theta = theta_const
        self.max_seq_len = max_seq_len
        rope_blocks = precompute_rope_blocks(theta_const,d_k, max_seq_len, device)
        self.register_buffer('rope_cache', rope_blocks, persistent = False)
        pass

    def forward(self, x: Float[torch.Tensor, '... seq_len d_k'], token_positions: torch.Tensor) -> torch.Tensor:
        """
        returns Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input
        """
        seq_len = x.shape[-2]
        half = x.shape[-1] // 2
        for pos in range(0, seq_len):
            for k in range(0, half):
                pair = x[..., pos, 2*k : 2*k+2]
                rope_block = self.rope_cache[pos,k] # type: ignore
                # need to transpose b/c we implemented rope matrix as defined in the papers (so it expected column-vectors not row-vectors)
                x[..., pos, 2*k : 2*k + 2] = pair @ rope_block.T
        return x

if __name__ == "__main__":
    z = precompute_rope_blocks(10_000, 64, 8)
    print(z.shape)