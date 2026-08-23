import torch


def softmax(x: torch.Tensor, dim=-1):
    big = torch.max(x, dim=dim, keepdim=True)
    x = x - big.values
    x = torch.exp(x)
    x = x / torch.sum(x, dim=dim, keepdim=True)
    return x