from collections.abc import Iterable
import math

from einops import einsum, rearrange
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

def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]):
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
    return loss.mean()  # scalar with NllLoss-like backward

def get_lr_cosine_sched(t, alphamax, alphamin, tw, tc):
    """
        (Warm-up) If 𝑡 < 𝑇𝑤, then lr = t * alphamax / tw
        (Cosine annealing) If 𝑇𝑤 ≤ 𝑡 ≤ 𝑇𝑐, then lr = alphamin + 0.5 * cos( 1 + pi * (t - tw)/tc - tw)) * (alphamax - alphamin)
        (Post-annealing) If 𝑡 > 𝑇𝑐, then lr = alphamin
    """
    if t < tw:                      #number of warmup steps
        result = t * alphamax / tw
    elif t <= tc:                   # number of cosine annealing steps
        result = alphamin + 0.5 * (1 + math.cos( math.pi * ( t - tw) / (tc - tw))) * (alphamax - alphamin)
    else:
        result = alphamin
    return result

def clip_gradient(params: Iterable[torch.nn.Parameter], maxgrad, eps = 1e-6):
    params2 = [p for p in params if p.grad is not None]
    norms = [p.grad.norm(2) for p in params2]
    norms_tensor = torch.stack(norms)
    l2 = torch.norm(norms_tensor, 2)
    clip_factor = maxgrad / (l2 + eps)
    if clip_factor < 1:
        for p in params2:
            p.grad.mul_(clip_factor)
    return l2.item()


def compute_loss(model, input_tokens, output_tokens):
    logits = model(input_tokens)
    logits = rearrange(logits, 'b c d -> (b c) d')
    output_tokens = rearrange(output_tokens, 'b c -> (b c)')
    result = cross_entropy(logits, output_tokens)
    return result

def calc_validation_loss(model, validation_tokens, eval_batch_size, seq_size, num_eval_batches=20, device=None):
    model.eval()
    try:
        losses = []
        with torch.no_grad():
            for _ in range(num_eval_batches):
                input_tokens, output_tokens = get_batch(validation_tokens, eval_batch_size, seq_size, device)
                loss = compute_loss(model, input_tokens, output_tokens)
                losses.append(loss.item())
        return sum(losses) / len(losses)
    finally:
        model.train()

def silu(x: Float[Tensor, "d_model d_ff"]) -> Float[Tensor, "d_model d_ff"]:
    result = x * torch.sigmoid(x)
    return result
