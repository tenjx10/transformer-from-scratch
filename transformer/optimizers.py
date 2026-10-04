import torch
import math
from collections.abc import Callable, Iterable
from typing import Optional

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None: 
                    continue

                state = self.state[p]

                # initial iteration
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["t"] = 1

                beta1, beta2 = betas
                g = p.grad.data

                t = state["t"]
                m = state["m"]
                v = state["v"]

                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= lr * weight_decay * p.data
                m = beta1 * m + (1-beta1) * g
                v = beta2 * v + (1-beta2) * g**2
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)

                # update/save m,v,t
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
        
        return loss

def lr_cosine_schedule(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int):
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    elif warmup_iters <= it and it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos(((it - warmup_iters) / (cosine_cycle_iters - warmup_iters)) * math.pi)) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    # list of the gradients
    param = [p for p in parameters if p.grad is not None]
    l2 = torch.sqrt(sum((p.grad**2).sum() for p in param))
    
    if l2 > max_l2_norm:
        scale_val = max_l2_norm / (l2 + 1e-6)
        for p in param:
            p.grad *= scale_val


