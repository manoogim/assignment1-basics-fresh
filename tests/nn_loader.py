import os
import random
from typing import Tuple
import typing

from einops import rearrange
import torch
from jaxtyping import Float, Int
from torch import nn

from tests.bpe_tokenizer import read_tokens_binary



def get_batch(x, batch_size, ctx_len, device=None) -> Tuple[Int[torch.Tensor, 'batch_size ctx_len'], Int[torch.Tensor, 'batch_size ctx_len']]:
    max_start = len(x) - ctx_len - 1
    if max_start < 0:
        raise ValueError(f'Not enough elements: {len(x)} cannot support matrix {batch_size} x {ctx_len}')

    inputs = []
    outputs = []
    for _ in range(batch_size):
        start = random.randint(0, max_start)
        # total segment to consume - note adding 1 to accommodate shift by one place for outputs
        ids = x[start : start + ctx_len + 1]
        inputs.append(ids[:-1])
        outputs.append(ids[1:])
    result = torch.tensor(inputs, device=device), torch.tensor(outputs, device=device)
    return result

def sample_batch():
    x = [12, 7, 7, 99, 42, 7, 13, 88, 5, 5, 5, 90]
    b = 2
    ctx_len = 4
    result = get_batch (x, b, ctx_len)
    print (result)

def vocab_batch():
    vocab_folder = r"C:\Users\Melissa\stanford\cs336\assignment1-basics-fresh\out\tinystories_GPT4"
    tokens_file = os.path.join(vocab_folder,'tokens_valid.bin')
    x = read_tokens_binary(tokens_file)
    result = get_batch(x, 32, 7)
    print(result)

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration:int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    obj = {}
    obj['iteration'] = iteration
    obj['model_state'] = model.state_dict()
    obj['adamw_state'] = optimizer.state_dict()
    torch.save(obj, out)
    print(f'Saved state at iteration: {iteration} to: {out}')


def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    obj = torch.load(src)
    model.load_state_dict(obj['model_state'])
    optimizer.load_state_dict(obj['adamw_state'])
    iteration = obj['iteration']
    return iteration
