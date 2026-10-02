from torch import nn

from tests.nn_mhsa import MultiHeadSelfAttention, MultiheadSelfattentionRoped
from tests.nn_norm import MyRmsNorm
from tests.nn_swiglu import MySwiglu
from tests.nn_yaml import make_norm

"""
    Inputs:
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        d_ff: int Dimensionality of the position-wise feed-forward inner layer.
"""
class MyTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 max_seq_len: int,
                 norm: dict,
                 eps: float = 0.00001, 
                 theta: float = 10_000,
                 
                 device = None, dtype = None):
        super().__init__()

        self.rms_norm1 = make_norm(d_model, norm, "attn", device=device, dtype=dtype)
        # self.mha = MultiHeadSelfAttention(d_model, num_heads, device, dtype)
        self.mha = MultiheadSelfattentionRoped(d_model, num_heads, theta, max_seq_len, device, dtype)

        self.rms_norm2 = make_norm(d_model, norm, "ffn", device=device, dtype=dtype)
        self.ff_block = MySwiglu(d_model, d_ff, device, dtype)
        
    def forward(self, x, token_positions = None):
        y = self.rms_norm1(x + self.mha(x, token_positions=token_positions))
        y = self.rms_norm2(y + self.ff_block(y))
        return y
    
    def forward2(self, x, token_positions = None):
        y = x + self.mha(self.rms_norm1(x), token_positions=token_positions)
        y = y + self.ff_block(self.rms_norm2(y))
        return y