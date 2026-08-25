from torch import nn

from tests.nn_mhsa import MultiHeadSelfAttention, MultiheadSelfattentionRoped
from tests.nn_norm import MyRmsNorm
from tests.nn_swiglu import MySwiglu

"""
    Inputs:
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        d_ff: int Dimensionality of the position-wise feed-forward inner layer.
"""
class MyTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 max_seq_len: int,
                 eps: float = 0.00001, 
                 theta: float = 10_000,
                 
                 device = None, dtype = None):
        super().__init__()
        self.rms_norm1 = MyRmsNorm(d_model, eps, device, dtype)
        # self.mha = MultiHeadSelfAttention(d_model, num_heads, device, dtype)
        self.mha = MultiheadSelfattentionRoped(d_model, num_heads, theta, max_seq_len, device, dtype)

        self.rms_norm2 = MyRmsNorm(d_model, eps, device, dtype)
        self.ff_block = MySwiglu(d_model, d_ff, device, dtype)
        

    def forward(self, x):
        y = x + self.mha(self.rms_norm1(x))
        y = y + self.ff_block(self.rms_norm2(y))
        return y