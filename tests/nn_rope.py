import math
from einops import einsum
from jaxtyping import Float, Int
import torch
from torch import Tensor, nn
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
    ropes = torch.empty(max_seq_len, d_k//2, 2, 2, device = device, dtype = dtype)
    for pos in range (max_seq_len):
        alphas = [pos * freq for freq in frequencies]
        for k in range (half):
            alpha = alphas[k]
            ropes[pos, k] = my_rotation(alpha)
    return ropes

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta_const: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        rope_blocks = precompute_rope_blocks(theta_const,d_k, max_seq_len, device, dtype)
        self.register_buffer('rope_cache', rope_blocks, persistent = False)
        pass

    def forward(self,  x: Float[Tensor, " ... sequence_length d_k"], token_positions_in: Int[Tensor, " ... sequence_length"]) -> torch.Tensor: 
        *batch, seq_len, d_k = x.shape
        token_positions = token_positions_in if token_positions_in is not None else  torch.arange(seq_len, device = x.device)
        half = d_k // 2

        x_pairs = x.reshape(*batch, seq_len, half, 2)          # (..., seq_len, half, 2)
        # batchness of rope_blocks comes from the token_positions
        # print(f'rope cache: {self.rope_cache.shape}, positions: {token_positions.shape}')
        rope_blocks = self.rope_cache[token_positions]          # type: ignore # (..., seq_len, half, 2, 2)
        # print(f'rope_blocks: {rope_blocks.shape}, x_pairs: {x_pairs.shape}')
        
        rotated = einsum (rope_blocks, x_pairs,'... heads pairs i j, ... heads pairs j -> ... heads pairs i')
        reshaped = rotated.reshape(*batch, seq_len, d_k)
        return reshaped

    def _token_positions(self, x):
        seq_len = x.shape[-2]
        positions = torch.arange(seq_len, device = x.device)
        return positions

    # not used - but could be used to implement a serial version of RoPE, in which case do not overwrite the input tensor X
    def forward_serial(self, x: Float[torch.Tensor, '... seq_len d_k'], token_positions: torch.Tensor) -> torch.Tensor:
        """
        returns Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input
        """

        half = x.shape[-1] // 2
        for pos in token_positions:
            for k in range(0, half):
                # slicing to get the last two
                pair = x[..., pos, 2*k : 2*k+2]
                rope_block = self.rope_cache[pos,k] # type: ignore
                # need to transpose b/c we implemented rope matrix as defined in the papers (so it expected column-vectors not row-vectors)
                x[..., pos, 2*k : 2*k + 2] = pair @ rope_block.T
        # code overwrites the input tensor above which can messup autograd if put in use in a nn.Module
        return x

if __name__ == "__main__":
    z = precompute_rope_blocks(10_000, 64, 8)
    print(z.shape)