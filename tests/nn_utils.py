from collections.abc import Iterable
import math
import random

from einops import einsum, rearrange
import numpy as np
import torch
from torch import Tensor
from jaxtyping import Float, Int

from tests.nn_loader import get_batch

def softmax(x: torch.Tensor, dim=-1):
     # using property of exp(a) / exp(b), that we can subtract same value from a and b 
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
    mask = maskin if maskin is not None else build_mask(scores.shape, device=scores.device)
    scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = softmax(scores, -1) 
    result = einsum(attention_weights, V, '... Q K, ... K dv -> ... Q dv')
    return result

# alternative impl - not used
def causal_mask(scores):
    """
    put -inf in upper triangle
    """
    ones = torch.ones(scores.shape)
    mask = ones.triu(diagonal=1).bool()
    result = scores.masked_fill(mask ==1, float('-inf'))
    return result


def build_mask(dims, device=None):
    ones =  torch.ones(dims, device=device)
    mask = ones.tril(diagonal=0)
    return mask

def stable_log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # sx = softmax(x, dim)
    # result = torch.log(sx)
    max_val = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - max_val   
    sumexp = torch.sum(torch.exp(x_shifted), dim=dim, keepdim=True)
    log_denom = torch.log(sumexp)
    log_num = x_shifted
    result = log_num - log_denom
    return result

def cross_entropy_loss_slow(
        inputs: Float[Tensor, " batch_size vocab_size"], 
        targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy loss across examples.

        Args:
            inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
                unnormalized logit of jth class for the ith example.
            targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
                Each value must be between 0 and `num_classes - 1`.

        Returns:
            Float[Tensor, ""]: The average cross-entropy loss across examples.
        """

    nn = len(targets)
    total_loss = 0.0
    for ii in range(nn):
        x = inputs[ii]
        predicted = stable_log_softmax(x)
        
        idx = int(targets[ii])
        loss = - predicted[idx]
        total_loss += loss
    # result = -torch.Tensor([sum / nn])
    result = total_loss / nn  # scalar tensor with grad_fn
    return result # type: ignore

def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"], reduction: str = 'mean'):
    """Given a tensor of inputs and targets, compute the average cross-entropy loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    log_probs = torch.log_softmax(inputs, dim=-1)
    rows = torch.arange(len(targets)) # this is 0,1,2, ... NN-1
    loss = -log_probs[rows, targets]  # shape: [batch_size]
    if reduction == 'mean':
        return loss.mean()  # scalar with NllLoss-like backward
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def get_lr_cosine_sched(t, alphamax, alphamin, tw, tc):
    """
    Computes learning rate for step t. 
        LR is guaranteed to be within range [alphamin, alphamx] when in cosine annealing stage, but not while in warmup stage.
    Inputs:
    t: step, starting from 1 whatever
    alphamax: max learning rate
    alphamin: min learning rate
    tw: step when warmup ends, after this anealing starts (during tw period lr is a flat line, upward trending towards alphamax)
    tc: step when anealing ends, after this it's flat rate alphamin (during tc period lr is smooth curve tending downward)

        (Warm-up) If 𝑡 < 𝑇𝑤, then lr = t * alphamax / tw
        (Cosine annealing) If 𝑇𝑤 ≤ 𝑡 ≤ 𝑇𝑐, then lr = alphamin + 0.5 * cos( 1 + pi * (t - tw)/tc - tw)) * (alphamax - alphamin)
        (Post-annealing) If 𝑡 > 𝑇𝑐, then lr = alphamin
    """
    if t < tw:                      #number of warmup steps
        result = t * alphamax / tw
    elif tw == tc:                  # drop from alphamax to alphamin without any cosine phase in between
        result = alphamin           
    elif t < tc:                   # performe gradual cosine annealing
        result = alphamin + 0.5 * (1 + math.cos( math.pi * ( t - tw) / (tc - tw))) * (alphamax - alphamin)
    else:
        result = alphamin
    return result

def clip_gradient(params: Iterable[torch.nn.Parameter], maxgrad, eps = 1e-6):
    params2 = [p for p in params if p.grad is not None]
    norms = [p.grad.norm(2) for p in params2] # type: ignore
    norms_tensor = torch.stack(norms)
    l2 = torch.norm(norms_tensor, 2)
    clip_factor = maxgrad / (l2 + eps)
    if clip_factor < 1:
        for p in params2:
            p.grad.mul_(clip_factor) # type: ignore
    return l2.item()


def compute_safe_ppl(loss):
    try:
        return math.exp(loss)
    except OverflowError:
        return float('inf')

def compute_loss(model, input_tokens, output_tokens):
    logits = model(input_tokens)
    logits = rearrange(logits, 'b c d -> (b c) d')
    output_tokens = rearrange(output_tokens, 'b c -> (b c)')
    result = cross_entropy(logits, output_tokens)
    return result

def compute_loss_sum(model, input_tokens, output_tokens):
    logits = model(input_tokens)
    logits = rearrange(logits, "b c d -> (b c) d")
    targets = rearrange(output_tokens, "b c -> (b c)")

    loss_sum = cross_entropy(
        logits,
        targets,
        reduction="sum",
    )

    return loss_sum, targets.numel()

def add_stats(batch_losses, crit_value = 1.984):
    mean = np.mean(batch_losses)
    std = np.std(batch_losses, ddof=1)
    se = std / np.sqrt(len(batch_losses) )
    margin = crit_value * se
    result = {
        'mean': float(mean),
        'std': float(std),
        'se': float(se),
        'ci984': (float(mean-margin), float(mean+margin))
    }
    return result

def calc_validation_loss(model, validation_tokens, eval_batch_size, seq_size, num_eval_batches, eval_seed, device=None):
    if num_eval_batches <= 0:
        raise ValueError("num_eval_batches must be positive")

    g = torch.Generator()
    g.manual_seed(eval_seed)

    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    eval_batches = 0
    eval_sequences = 0
    batch_losses = []
    try:
        with torch.inference_mode():
            for _ in range(num_eval_batches):
                input_tokens, output_tokens = get_batch(validation_tokens, eval_batch_size, seq_size, g, device)
                loss_sum, valid_tokens = compute_loss_sum(model, input_tokens, output_tokens)
                batch_losses.append(loss_sum.item() / valid_tokens)

                # accounting
                total_loss += loss_sum.item()
                eval_batches += 1
                total_tokens += valid_tokens
                eval_sequences += input_tokens.shape[0]
        val_loss = total_loss / total_tokens

        result = {}
        result['val_loss'] = val_loss
        result['val_ppl'] = compute_safe_ppl(val_loss)
        result['eval_batches'] = eval_batches # this is known from input
        result['eval_tokens'] = total_tokens
        result['eval_sequences'] = eval_sequences
        return result, batch_losses
    finally:
        model.train(was_training)

def calc_stats(batch_losses, crit_value = 1.984):
    mean = np.mean(batch_losses)
    std = np.std(batch_losses, ddof=1)
    se = std / np.sqrt(len(batch_losses) )
    margin = crit_value * se
    result = {
        'mean': float(mean),
        'std': float(std),
        'se': float(se),
        'ci984': (float(mean-margin), float(mean+margin))
    }
    return result

def silu(x: Float[Tensor, "d_model d_ff"]) -> Float[Tensor, "d_model d_ff"]:
    result = x * torch.sigmoid(x)
    return result

def derive_ckpt_name(step, save_every_steps, keep_last):
    if step % save_every_steps != 0:
        print(f'Saving off-schedule at step: {step}')
        off = '_off'
    else:
        off = ''

    save_event_idx = step // save_every_steps -1 
    slot = save_event_idx % keep_last
    suffix = chr(ord('a') + slot)
    return f'ckpt_{suffix}{off}.pt'

def plant_seed (seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g
