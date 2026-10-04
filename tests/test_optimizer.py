import math

import numpy as np
import torch

from transformer.data import get_batch
from transformer.optimizers import AdamW, gradient_clipping, lr_cosine_schedule


def test_adamw_matches_torch():
    torch.manual_seed(0)
    ours = torch.nn.Linear(8, 4)
    ref = torch.nn.Linear(8, 4)
    ref.load_state_dict(ours.state_dict())

    hp = dict(lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    opt_ours = AdamW(ours.parameters(), **hp)
    opt_ref = torch.optim.AdamW(ref.parameters(), **hp)

    for _ in range(20):
        x = torch.randn(16, 8)
        for model, opt in [(ours, opt_ours), (ref, opt_ref)]:
            opt.zero_grad()
            (model(x) ** 2).mean().backward()
            opt.step()

    for p, q in zip(ours.parameters(), ref.parameters()):
        torch.testing.assert_close(p, q, atol=1e-5, rtol=1e-4)


def test_gradient_clipping_matches_torch():
    torch.manual_seed(0)
    a = [torch.nn.Parameter(torch.randn(5, 5)) for _ in range(3)]
    b = [torch.nn.Parameter(p.detach().clone()) for p in a]
    for params in (a, b):
        sum((p * 10).sum() ** 2 for p in params).backward()

    gradient_clipping(a, max_l2_norm=1.0)
    torch.nn.utils.clip_grad_norm_(b, max_norm=1.0)
    for p, q in zip(a, b):
        torch.testing.assert_close(p.grad, q.grad)


def test_cosine_schedule_shape():
    lr = lambda it: lr_cosine_schedule(it, 1.0, 0.1, warmup_iters=10, cosine_cycle_iters=110)
    assert lr(0) == 0.0 and math.isclose(lr(5), 0.5)       # linear warmup
    assert math.isclose(lr(10), 1.0)                         # peak
    assert math.isclose(lr(60), 0.55)                        # halfway down the cosine
    assert math.isclose(lr(110), 0.1) and lr(500) == 0.1    # floor


def test_get_batch_targets_are_shifted_inputs():
    data = np.arange(1000)
    x, y = get_batch(data, batch_size=8, context_length=16, device="cpu")
    assert x.shape == y.shape == (8, 16)
    torch.testing.assert_close(y, x + 1)
