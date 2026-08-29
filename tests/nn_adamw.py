from typing import Callable, Optional

import torch


class MyAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=1e-2, betas = (0.9, 0.999), eps= 1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0:
            raise ValueError('Epsilon is negative: {eps}')

        defaults = {
            'alpha': lr,
            'lambda_decay': weight_decay,
            'beta1': betas[0],
            'beta2': betas[1],
            'epsilon': eps 
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None): # type: ignore
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha, lambda_decay, beta1, beta2, epsilon = group['alpha'], group['lambda_decay'], group['beta1'], group['beta2'], group['epsilon']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get('t', 1) # Get iteration number from the state, or 1.
                    
                m = state.get('m', torch.zeros_like(p)) # get 1st moment from state or initialize to zeros
                v = state.get('v', torch.zeros_like(p)) # get 2nd moment from state or initialize to zeros
                grad = p.grad # Get the gradient of loss with respect to p.
                # alpha_factor = (1 - beta2 ** t) ** (.5) / (1 - beta1 ** t) will be accounted for with m_hat and v_hat
                # TODO we can remove no_grad block if we annotate step signature with @no_grad 
                with torch.no_grad(): # disabling grad tracking b/c we are updating state in place
                    p.mul_(1 - alpha * lambda_decay) # apply weight decay


                    m.mul_(beta1).add_(grad, alpha = 1 - beta1) # update m in place:  m = beta1 * m + (1 - beta1) * grad
                    v.mul_(beta2).addcmul_(grad, grad, value = 1-beta2) # update v in place: v = beta2 * v + (1-beta2) * grad * grad

                    m_hat = m / (1 - beta1 ** t)
                    v_hat = v / (1 - beta2 ** t)

                    p.addcdiv_(m_hat, torch.sqrt(v_hat) + epsilon, value = -alpha) # p = p + m_hat /( (v_hat ** 0.5) + eps) * (-lr)
                    state['t'] = t + 1
                    state['m'] = m
                    state['v'] = v

def main(lr, iters=100):
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


if __name__ == '__main__':
    iters = 10_000
    for lr in [ 1e-3]:
        main(lr, iters)