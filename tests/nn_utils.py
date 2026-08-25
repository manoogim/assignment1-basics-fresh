import math

from einops import einsum
import torch
from torch import Tensor
from jaxtyping import Float, Int

def softmax(x: torch.Tensor, dim=-1):
    big = torch.max(x, dim=dim, keepdim=True)
    x = x - big.values
    x = torch.exp(x)
    x = x / torch.sum(x, dim=dim, keepdim=True)
    return x

def scaled_dot_product_attention(
        Q: Float[Tensor, '... Q dk'], 
        K: Float[Tensor, '... K dk'], 
        V: Float[Tensor, '... K dv'], 
        maskin = None) -> Float[Tensor, '... Q dv']:
    
    dk = Q.shape[-1]
    scores = einsum(Q, K, '... Q dk, ... K dk -> ... Q K') / math.sqrt(dk)
    mask = maskin if maskin is not None else build_mask(scores.shape)
    scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = softmax(scores, -1) 
    result = einsum(attention_weights, V, '... Q K, ... K dv -> ... Q dv')
    return result

def causal_mask(scores):
    """
    put -inf in upper triangle
    """
    ones = torch.ones(scores.shape)
    mask = ones.triu(diagonal=1).bool()
    result = scores.masked_fill(mask ==1, float('-inf'))
    return result


def build_mask(dims):
    ones =  torch.ones(dims)
    mask = ones.tril(diagonal=0)
    return mask

if __name__ == '__main__':
    x = [[1, 2, 3],[4,5,6], [7,8,9]]
    x = torch.tensor(x)
    # m = causal_mask(x)
    m = build_mask(x.shape)
    print(m)
    
