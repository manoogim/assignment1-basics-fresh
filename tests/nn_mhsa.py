from torch import Tensor, nn
from jaxtyping import Float
from einops import rearrange, einsum
import torch

from tests.nn_rope import RotaryPositionalEmbedding
from tests.nn_utils import scaled_dot_product_attention


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, device = None, dtype = None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.dk = self.dv = d_model // num_heads
        self.q = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.k = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.v = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)

    def _apply_pos_encoding(self, qx, kx, token_positions=None):
        # base class: identity. RoPE subclass overrides this.
        return qx, kx

    
    def forward(self, x: Float[Tensor, '... seq dm'], token_positions=None):          
        qx, kx, vx = self.q(x), self.k(x), self.v(x)
        
        qx = rearrange(qx, '... seq (h dk) -> ... h seq dk', h=self.num_heads)
        kx = rearrange(kx, '... seq (h dk) -> ... h seq dk', h=self.num_heads)
        vx = rearrange(vx, '... seq (h dv) -> ... h seq dv', h=self.num_heads)

        qx, kx = self._apply_pos_encoding(qx, kx, token_positions)

        result = scaled_dot_product_attention(qx, kx, vx)
        result = rearrange(result, '... h seq dv -> ... seq ( h dv)')
        result = self.o_proj(result)
         
        return result


class MultiheadSelfattentionRoped(MultiHeadSelfAttention):
    def __init__(self, d_model: int, num_heads: int, theta: float = 10000.0, max_seq_len: int = 128, device=None, dtype=None):
        super().__init__(d_model, num_heads, device, dtype)
        self.rope = RotaryPositionalEmbedding( theta, self.dk, max_seq_len, device, dtype)  # precomputes/caches cos & sin


    def _apply_pos_encoding(self, qx, kx, token_positions=None):
        qx = self.rope(qx, token_positions)
        kx = self.rope(kx, token_positions)
        return qx, kx