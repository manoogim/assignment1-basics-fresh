import os
import random
import time
from typing import Tuple
import typing

import numpy
import torch
from jaxtyping import Int
from torch import nn


def get_batch(x, batch_size, ctx_len, g: torch.Generator = None, device=None) -> Tuple[Int[torch.Tensor, 'batch_size ctx_len'], Int[torch.Tensor, 'batch_size ctx_len'], float]:
    start_load = time.perf_counter()
    max_start = len(x) - ctx_len - 1
    if max_start < 0:
        raise ValueError(f'Not enough elements: {len(x)} cannot support matrix {batch_size} x {ctx_len}')

    inputs = []
    outputs = []
    for _ in range(batch_size):
        # use torch.randint instead of random.randint
        start = torch.randint(
            low=0,
            high=max_start,
            size=(1,),
            generator=g
        ).item()
        
        # total segment to consume - note adding 1 to accommodate shift by one place for outputs
        ids = x[start : start + ctx_len + 1]
        inputs.append(ids[:-1])
        outputs.append(ids[1:])
    # converting type from uint16 to int32 for lookups is mandatory, otherwise torch will throw an error when trying to index with uint16
    #  wraping [] with numpy.array is recommended to avoid torch warning about creating tensor from list of numpy arrays
    load_time = time.perf_counter() - start_load
    result = torch.tensor(numpy.array(inputs), device=device, dtype=torch.int32), torch.tensor(numpy.array(outputs), device=device, dtype=torch.int32), load_time   
    return result

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration:int, out_path: str, sched_info: dict = {} ):
    obj = {}
    obj['iteration'] = iteration
    obj['sched_info'] = sched_info
    obj['model_state'] = model.state_dict()
    obj['adamw_state'] = optimizer.state_dict()

    # save to tmp and atomically rename 
    tmp_path = out_path + '.tmp'
    torch.save(obj, tmp_path)
    os.replace(tmp_path, out_path)
    # Save step to a text file
    txt_path = out_path + ".step.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(str(iteration))

def load_checkpoint( src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: nn.Module, optimizer: torch.optim.Optimizer | None,  device) -> tuple[int, dict]:
    obj = torch.load(src, map_location=device)
    model.load_state_dict(obj['model_state'])
    if optimizer is not None:
        optimizer.load_state_dict(obj['adamw_state'])
    
    sched_info = obj['sched_info']
    iteration = obj['iteration']
    return iteration, sched_info
