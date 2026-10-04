import torch
from einops import rearrange

def cross_entropy(logits, targets):
    logits = rearrange(logits, "... vocab -> (...) vocab")
    targets = rearrange(targets, "... -> (...)")

    shifted = logits - torch.max(logits, dim=-1, keepdim=True).values
    loss = torch.log(torch.exp(shifted).sum(dim=-1)) - shifted[torch.arange(shifted.shape[0]), targets]
    return loss.mean()


