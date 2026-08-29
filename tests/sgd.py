from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None): # type: ignore
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss

def main(lr=1, iters=100):
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)

    losses = []

    for t in range(iters):
        opt.zero_grad()
        loss = (weights**2).mean()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        # print(f'LR:{lr}, t:{t}, loss:{loss.item()}')

    # ---- Summary ----
    first = losses[0]
    last = losses[-1]
    min_loss = min(losses)
    max_loss = max(losses)
    pct_decrease = 100 * (first - last) / first

    print("\n=== Loss Summary ===")
    print(f'LR: {lr}')
    print(f"First loss: {first:.6f}")
    print(f"Last loss:  {last:.6f}")
    print(f"Min loss:   {min_loss:.6f}")
    print(f"Max loss:   {max_loss:.6f}")
    print(f"Percent decrease: {pct_decrease:.2f}%")


if __name__ == '__main__':
    iters = 10
    for lr in [ 10]:
        main(lr, iters)