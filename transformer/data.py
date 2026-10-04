import numpy as np
import numpy.typing as npt
import torch

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    pos = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    idxs = pos[:, None] + np.arange(context_length)[None, :]
    
    input_seq = torch.tensor(dataset[idxs], dtype=torch.long, device=device)
    targets = torch.tensor(dataset[idxs+1], dtype=torch.long, device=device)
    return input_seq, targets
