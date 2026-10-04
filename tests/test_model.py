"""check each from-scratch component against PyTorch's reference implementation"""
import math

import torch
import torch.nn.functional as F

from transformer.loss import cross_entropy
from transformer.model import (Embedding, Linear, MultiHeadSelfAttention, RMSNorm, RotaryPositionalEmbedding,
                               SwiGLU, TransformerLM, scaled_dot_product_attention, silu, softmax)

torch.manual_seed(0)


def small_model(**kwargs):
    return TransformerLM(vocab_size=100, context_length=32, d_model=64, num_layers=2, num_heads=4, d_ff=128,
                         rope_theta=10000.0, **kwargs)


def test_linear_matches_matmul():
    layer = Linear(16, 8)
    x = torch.randn(2, 5, 16)
    torch.testing.assert_close(layer(x), x @ layer.weight.T)


def test_embedding_lookup():
    emb = Embedding(10, 4)
    ids = torch.tensor([[1, 2], [9, 0]])
    torch.testing.assert_close(emb(ids), emb.weight[ids])


def test_softmax_matches_torch_and_is_stable():
    x = torch.randn(4, 10) * 50
    torch.testing.assert_close(softmax(x, dim=-1), torch.softmax(x, dim=-1))
    # huge logits must not overflow to nan
    assert torch.isfinite(softmax(torch.tensor([1000.0, 1001.0]), dim=0)).all()


def test_silu_matches_torch():
    x = torch.randn(100)
    torch.testing.assert_close(silu(x), F.silu(x))


def test_rmsnorm_matches_torch():
    x = torch.randn(3, 7, 32)
    ours = RMSNorm(32, eps=1e-5)
    ref = torch.nn.RMSNorm(32, eps=1e-5)
    torch.testing.assert_close(ours(x), ref(x))


def test_swiglu_formula():
    ffn = SwiGLU(16, 48)
    x = torch.randn(2, 3, 16)
    expected = (F.silu(x @ ffn.w1.weight.T) * (x @ ffn.w3.weight.T)) @ ffn.w2.weight.T
    torch.testing.assert_close(ffn(x), expected)


def test_attention_matches_torch_causal():
    q, k, v = (torch.randn(2, 4, 10, 16) for _ in range(3))
    mask = ~torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1)
    torch.testing.assert_close(scaled_dot_product_attention(q, k, v, mask),
                               F.scaled_dot_product_attention(q, k, v, is_causal=True))


def test_rope_preserves_norm_and_encodes_relative_position():
    rope = RotaryPositionalEmbedding(10000.0, d_k=16, max_seq_len=64)
    q, k = torch.randn(16), torch.randn(16)
    rotated = rope(q, torch.tensor(5))
    torch.testing.assert_close(rotated.norm(), q.norm())

    # <R_m q, R_n k> only depends on m - n
    def score(m, n):
        return rope(q, torch.tensor(m)) @ rope(k, torch.tensor(n))
    torch.testing.assert_close(score(10, 7), score(40, 37))


def test_attention_is_causal():
    attn = MultiHeadSelfAttention(32, 4, max_seq_len=16, theta=10000.0)
    x = torch.randn(1, 8, 32)
    changed = x.clone()
    changed[:, 5:] = torch.randn(1, 3, 32)
    # editing tokens 5..7 must not change the outputs for tokens 0..4
    torch.testing.assert_close(attn(x)[:, :5], attn(changed)[:, :5])


def test_cross_entropy_matches_torch():
    logits = torch.randn(4, 6, 50) * 10
    targets = torch.randint(0, 50, (4, 6))
    torch.testing.assert_close(cross_entropy(logits, targets), F.cross_entropy(logits.view(-1, 50), targets.view(-1)))


def test_lm_shapes_and_param_count():
    model = small_model()
    ids = torch.randint(0, 100, (3, 20))
    assert model(ids).shape == (3, 20, 100)

    # embedding + lm_head + per-layer (attention 4 d^2, SwiGLU 3 d d_ff, two norms) + final norm
    d, d_ff, layers, vocab = 64, 128, 2, 100
    expected = 2 * vocab * d + layers * (4 * d * d + 3 * d * d_ff + 2 * d) + d
    assert sum(p.numel() for p in model.parameters()) == expected


def test_ablation_variants_run():
    ids = torch.randint(0, 100, (2, 16))
    for kwargs in [{"no_rmsnorm": True}, {"post_norm": True}, {"no_rope": True}, {"silu_ffn": True}]:
        logits = small_model(**kwargs)(ids)
        assert logits.shape == (2, 16, 100) and torch.isfinite(logits).all(), kwargs


def test_model_can_overfit_one_batch():
    model = small_model()
    ids = torch.randint(0, 100, (4, 17))
    x, y = ids[:, :-1], ids[:, 1:]
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    start = cross_entropy(model(x), y).item()
    for _ in range(100):
        loss = cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert start > math.log(100) * 0.8 and loss.item() < 0.5
