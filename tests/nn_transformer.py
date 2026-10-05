from torch import nn
import torch
from jaxtyping import Int, Float

from tests.nn_block import MyTransformerBlock
from tests.nn_embedding import MyEmbedding
from tests.nn_linear import MyLinear
from tests.nn_norm import MyRmsNorm
from tests.nn_yaml import  ModelConfig

import torch.utils.checkpoint as checkpoint
from enum import Enum

class ForwardMode(Enum):
    PLAIN = lambda block, x: block(x)
    CHECKPOINT = lambda block, x: checkpoint.checkpoint(block, x, use_reentrant=False)

    @classmethod
    def from_arg(cls, arg: str ) :
        a = arg.upper().strip()
        if a == 'PLAIN':
            return cls.PLAIN
        elif a == 'CHECKPOINT':
            return cls.CHECKPOINT
        else:
            raise ValueError(f"Invalid forward mode: {arg}")

class MyTransformer(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 num_layers: int,
                 max_context: int,
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int,
                 forward_mode: str,
                 eps: float = 0.00001, 
                 theta: float = 10_000,                 
                 device = None, dtype = None):
        super().__init__()
        
        blocks = [MyTransformerBlock(d_model, num_heads, d_ff, max_context, eps, theta, device, dtype) for _ in range(num_layers)]
        self.blocks = nn.Sequential(*blocks)

        self.input_embedding = MyEmbedding(vocab_size, d_model, device, dtype)

        self.norm = MyRmsNorm(d_model, eps, device, dtype)

        self.lm_head = MyLinear(d_model, vocab_size, device, dtype)

        self.forward_mode = ForwardMode.from_arg(forward_mode)

        # will be used for reporting only
        self.num_params = sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, dd: ModelConfig, device):
        return cls(dd.vocab_size, dd.num_layers, dd.seq_len, dd.d_model, dd.num_heads, dd.d_ff, dd.forward_mode, device=device)
    
    def forward(self, in_tokens: Int[torch.Tensor, 'batch_size seq_len']) -> Float[torch.Tensor, 'batch_size seq_len vocab_size']:
        x = self.input_embedding(in_tokens)

        for block in self.blocks:
            x = self.forward_mode(block, x)
        x = self.norm(x)

        logits = self.lm_head(x)

        return logits
