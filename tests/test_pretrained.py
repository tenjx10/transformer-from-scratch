"""smoke tests for the released TinyStories model"""
import math
import os

import pytest
import torch

from transformer import generate, load_pretrained
from transformer.loss import cross_entropy

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "tinystories")

STORY = ("Once upon a time, there was a little girl named Lily. She loved to play outside in the sun. "
         "One day, she saw a big red ball in the park. She ran to the ball and kicked it.")


@pytest.fixture(scope="module")
def pretrained():
    return load_pretrained(MODEL_DIR, device="cpu")


def test_loads_with_expected_size(pretrained):
    model, tokenizer = pretrained
    assert sum(p.numel() for p in model.parameters()) == 22_696_448
    assert len(tokenizer.vocab) == model.vocab_size == 10_000


def test_low_perplexity_on_a_story(pretrained):
    model, tokenizer = pretrained
    ids = torch.tensor([tokenizer.encode(STORY)])
    with torch.no_grad():
        loss = cross_entropy(model(ids[:, :-1]), ids[:, 1:]).item()
    # an untrained model sits at ln(10000) ~ 9.2
    assert math.exp(loss) < 15


def test_generates_text(pretrained):
    model, tokenizer = pretrained
    torch.manual_seed(0)
    text = generate(model, tokenizer, "Once upon a time,", max_new_tokens=40)
    assert text.startswith("Once upon a time,") and len(text.split()) > 10

    greedy = generate(model, tokenizer, "Once upon a time,", max_new_tokens=20, temperature=0)
    assert greedy == generate(model, tokenizer, "Once upon a time,", max_new_tokens=20, temperature=0)
