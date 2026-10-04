import torch
from torch import nn
from einops import rearrange, einsum


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # creating the weight matrix
        w = torch.empty(self.out_features, self.in_features, device=device, dtype=dtype)
        std = (2 / (self.in_features + self.out_features)) ** 0.5
        torch.nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3.0*std, b=3.0*std)
        self.weight = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")
        return y


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        embedding_w = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(embedding_w, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.weight = nn.Parameter(embedding_w)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # for each token id index into the embedding matrix
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        # creating the gain vector; learnable
        gain = torch.ones(self.d_model, device=device, dtype=dtype)
        self.weight = nn.Parameter(gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.weight
        return result.to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ffn = self.w2(silu(self.w1(x)) * self.w3(x))
        return ffn


class SiLUFeedForward(nn.Module):
    # ungated baseline for the SwiGLU ablation: W2 SiLU(W1 x)
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        freq = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2, device=device).float() / self.d_k))
        positions = torch.arange(self.max_seq_len, device=device).float()

        # every position times every frequency
        angles = einsum(positions, freq, "i, k -> i k")

        self.register_buffer("sin", torch.sin(angles), persistent=False)
        self.register_buffer("cos", torch.cos(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # look up the angles at each token position
        sin = self.sin[token_positions]
        cos = self.cos[token_positions]

        a = x[..., 0::2]
        b = x[..., 1::2]

        # rotate
        new_a = a*cos - b*sin
        new_b = a*sin + b*cos

        result = rearrange(torch.stack([new_a, new_b], dim=-1), "... d_pair ab -> ... (d_pair ab)")
        return result


def softmax(x: torch.Tensor, dim: int):
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    result = exp_x / exp_x.sum(dim=dim, keepdim=True)
    return result


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    pre_softmax = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / d_k ** 0.5

    # perform the masking
    if mask is not None:
        pre_softmax = torch.where(mask, pre_softmax, float("-inf"))

    weights = softmax(pre_softmax, dim=-1)
    return einsum(weights, V, "... queries keys, ... keys d_v -> ... queries d_v")


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # split them into heads
        q = rearrange(q, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads)
        k = rearrange(k, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads)
        v = rearrange(v, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads)

        # check if we are doing rope
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(x.shape[-2], device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # create the causal mask
        mask = ~torch.triu(torch.ones(x.shape[-2], x.shape[-2], dtype=torch.bool, device=x.device), diagonal=1)

        # calculate the attention with mask passed
        out = scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(out, "... heads seq d_k -> ... seq (heads d_k)")

        return self.output_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float | None,
                 no_rmsnorm: bool = False, post_norm: bool = False, silu_ffn: bool = False):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.post_norm = post_norm

        # nn.Identity stands in for RMSNorm in the "no RMSNorm" ablation
        self.ln1 = nn.Identity() if no_rmsnorm else RMSNorm(d_model)
        self.ln2 = nn.Identity() if no_rmsnorm else RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(self.d_model, self.num_heads, max_seq_len=max_seq_len, theta=theta)
        self.ffn = SiLUFeedForward(d_model, d_ff) if silu_ffn else SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor):
        if self.post_norm:
            # original Transformer ordering: normalize after the residual add
            z = self.ln1(x + self.attn(x))
            return self.ln2(z + self.ffn(z))

        # pre-norm: normalize the input to each sub-layer, keep the residual stream clean
        first_sub_layer = x + self.attn(self.ln1(x))

        # the second part where the position-wise ffn is located
        return first_sub_layer + self.ffn(self.ln2(first_sub_layer))


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int,
                 rope_theta: float, no_rmsnorm: bool = False, post_norm: bool = False, no_rope: bool = False,
                 silu_ffn: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embeddings = Embedding(vocab_size, d_model)

        # NoPE ablation: no positional information at all, the causal mask is the only signal
        theta = None if no_rope else self.rope_theta

        # use a ModuleList (not a python list) so the parameters are registered
        self.layers = nn.ModuleList([
            TransformerBlock(self.d_model, self.num_heads, self.d_ff, self.context_length, theta,
                             no_rmsnorm=no_rmsnorm, post_norm=post_norm, silu_ffn=silu_ffn)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.Identity() if no_rmsnorm else RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    @classmethod
    def from_config(cls, config: dict):
        """build a model from a training config (the config.json written by scripts/train.py)"""
        keys = ["vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff", "rope_theta",
                "no_rmsnorm", "post_norm", "no_rope", "silu_ffn"]
        return cls(**{k: config[k] for k in keys if k in config})

    def forward(self, x: torch.Tensor):
        x = self.token_embeddings(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        return self.lm_head(x)
