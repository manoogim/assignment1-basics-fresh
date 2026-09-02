import os

import torch

from tests.bpe_tokenizer import read_tokens_binary
from tests.nn_adamw import MyAdamW
from tests.nn_loader import get_batch


def adamw_sample(lr, iters=100):
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = MyAdamW([weights], lr, 0.1, (0.9, 0.999))

    losses = []

    for t in range(iters):
        opt.zero_grad()
        loss = (weights**2).mean()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        print(f'LR:{lr}, t:{t}, loss:{loss.item()}')

    # ---- Summary ----
    first = losses[0]
    last = losses[-1]
    min_loss = min(losses)
    max_loss = max(losses)
    pct_decrease = 100 * (first - last) / first

    print("\n=== Loss Summary ===")
    print(f"First loss: {first:.6f}")
    print(f"Last loss:  {last:.6f}")
    print(f"Min loss:   {min_loss:.6f}")
    print(f"Max loss:   {max_loss:.6f}")
    print(f"Percent decrease: {pct_decrease:.2f}%")


def sample_batch():
    x = [12, 7, 7, 99, 42, 7, 13, 88, 5, 5, 5, 90]
    b = 2
    ctx_len = 4
    result = get_batch (x, b, ctx_len)
    print (result)
    
def vocab_sample_batch():
    vocab_folder = r"C:\Users\Melissa\stanford\cs336\assignment1-basics-fresh\out\tinystories_GPT4"
    tokens_file = os.path.join(vocab_folder,'tokens_valid.bin')
    x = read_tokens_binary(tokens_file)
    result = get_batch(x, 32, 7)
    print(result)
    
if __name__ == '__main__':
    iters = 10_000
    for lr in [ 1e-3]:
        adamw_sample(lr, iters)