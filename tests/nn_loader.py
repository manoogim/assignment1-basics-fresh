import os
import random
from typing import Tuple
import typing

import numpy
import torch
from jaxtyping import Int
from torch import nn


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
    # converting type from uint16 to int32 for lookups is mandatory, otherwise torch will throw an error when trying to index with uint16
    #  wraping [] with numpy.array is recommended to avoid torch warning about creating tensor from list of numpy arrays
    result = torch.tensor(numpy.array(inputs), device=device, dtype=torch.int32), torch.tensor(numpy.array(outputs), device=device, dtype=torch.int32)
    return result

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration:int, out_path: str ):
    obj = {}
    obj['iteration'] = iteration
    obj['model_state'] = model.state_dict()
    obj['adamw_state'] = optimizer.state_dict()
    # save to tmp and atomically rename 
    tmp_path = out_path + '.tmp'
    torch.save(obj, tmp_path)
    os.replace(tmp_path, out_path)



def load_checkpoint( model: nn.Module, optimizer: torch.optim.Optimizer | None, src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], device) -> int:
    obj = torch.load(src, map_location=device)
    model.load_state_dict(obj['model_state'])
    if optimizer is not None:
        optimizer.load_state_dict(obj['adamw_state'])
    iteration = obj['iteration']
    return iteration
