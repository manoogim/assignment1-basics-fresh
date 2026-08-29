from torch import nn

from tests.nn_block import MyTransformerBlock
from tests.nn_embedding import MyEmbedding
from tests.nn_linear import MyLinear
from tests.nn_norm import MyRmsNorm


class MyTransformer(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 num_layers: int,
                 max_context: int,
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int,
                 eps: float = 0.00001, 
                 theta: float = 10_000,                 
                 device = None, dtype = None):
        super().__init__()
        
        blocks = [MyTransformerBlock(d_model, num_heads, d_ff, max_context, eps, theta, device, dtype) for _ in range(num_layers)]
        self.blocks = nn.Sequential(*blocks)

        self.input_embedding = MyEmbedding(vocab_size, d_model, device, dtype)

        self.norm = MyRmsNorm(d_model, eps, device, dtype)

        self.lm_head = MyLinear(d_model, vocab_size, device, dtype)

    def forward(self, in_tokens):
        x = self.input_embedding(in_tokens)
        x = self.norm(self.blocks(x))

        logits = self.lm_head(x)

        return logits
