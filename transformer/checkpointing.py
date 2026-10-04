import json
import os
import typing

import torch


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    obj = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(obj, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


def load_pretrained(model_dir: str | os.PathLike, device: str = "cpu"):
    """
    load a model + tokenizer for inference from a directory holding config.json, vocab.pkl,
    merges.pkl and either model.pt (weights only) or checkpoint.pt (a full training checkpoint)
    """
    from transformer.model import TransformerLM
    from transformer.tokenizer import Tokenizer

    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    model = TransformerLM.from_config(config)

    weights_path = os.path.join(model_dir, "model.pt")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(model_dir, "checkpoint.pt")
    state = torch.load(weights_path, map_location="cpu")
    state = state.get("model", state)

    # released weights are stored in float16 to keep the file small; compute in float32
    model.load_state_dict({k: v.float() for k, v in state.items()})
    model.to(device).eval()

    tokenizer = Tokenizer.from_files(os.path.join(model_dir, "vocab.pkl"), os.path.join(model_dir, "merges.pkl"),
                                     special_tokens=["<|endoftext|>"])
    return model, tokenizer
