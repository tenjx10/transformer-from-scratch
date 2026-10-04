# Transformer Language Model from Scratch

A GPT-style, decoder-only Transformer language model built from first principles in PyTorch. That covers the **byte-level BPE tokenizer, every neural network layer, the AdamW optimizer, the learning-rate schedule, the training loop and the sampling code**. It doesn't use `torch.nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `F.scaled_dot_product_attention`, `F.cross_entropy` or `torch.optim`. The only PyTorch pieces it relies on are tensors, autograd and `nn.Parameter`.

The architecture follows the modern LLaMA-style recipe: **pre-norm blocks, RMSNorm, SwiGLU feed-forward layers and rotary position embeddings (RoPE)**. I trained a 22.7M-parameter model on the [TinyStories](https://arxiv.org/abs/2305.07759) dataset on a laptop GPU (Apple M5, ~70 min). It reaches a **validation loss of 1.63 (perplexity 5.10)** and writes short, coherent stories:

> **Once upon a time,** there was a little dog named Max. Max liked to walk and play all day. He liked to run and jump. Max liked to have fun with his friends.
> One day, Max found a big stick in the yard. He wanted to play with it. He tried to push the stick, but it was too heavy. Max was sad. He wanted his stick back.
> Max's friend, a little bird, came to help him. The bird flew up high and picked up the stick. Max carried the stick to the bird. The bird was very happy. They played with the stick together all day.

*(unedited sample from the included checkpoint: `temperature=0.8, top_p=0.95, seed=0`)*

The trained weights are in the repo, so you can generate text right after cloning.

---

## Contents

- [Quickstart](#quickstart)
- [Architecture](#architecture)
- [How it was built](#how-it-was-built)
  - [1. Byte-level BPE tokenizer](#1-byte-level-bpe-tokenizer)
  - [2. Model components](#2-model-components)
  - [3. Training](#3-training)
  - [4. Decoding](#4-decoding)
- [Results and experiments](#results-and-experiments)
- [Training your own model](#training-your-own-model)
- [Tests](#tests)
- [Project structure](#project-structure)
- [References](#references)

---

## Quickstart

This project uses [uv](https://docs.astral.sh/uv/) for dependency management (`brew install uv` or `pip install uv`).

```sh
git clone https://github.com/<your-username>/transformer-from-scratch.git
cd transformer-from-scratch
uv sync

# generate a story with the pretrained model (picks CUDA, Apple MPS, or CPU automatically)
uv run python scripts/generate.py --prompt "Once upon a time,"
```

Useful flags:

| Flag | Default | Effect |
|---|---|---|
| `--prompt` | `"Once upon a time,"` | Text to continue |
| `--temperature` | `0.8` | Lower is more conservative; `0` is greedy decoding |
| `--top-p` | `0.95` | Nucleus sampling threshold |
| `--max-new-tokens` | `256` | Generation stops here or at `<\|endoftext\|>` |
| `--num-samples` | `1` | Number of independent completions |
| `--seed` | none | Makes sampling reproducible |
| `--device` | auto | `cuda`, `mps` or `cpu` |

```sh
uv run python scripts/generate.py --prompt "Tom and his dog went to the park." --num-samples 3 --temperature 0.7
```

### Using it from Python

```python
import torch
from transformer import load_pretrained, generate

model, tokenizer = load_pretrained("checkpoints/tinystories", device="cpu")
print(generate(model, tokenizer, "Lily found a shiny key in the garden.", temperature=0.8, top_p=0.95))

# lower-level access
ids = tokenizer.encode("Once upon a time")   # -> list[int]
logits = model(torch.tensor([ids]))          # -> (1, seq_len, 10000)
```

> **Note:** The model was trained only on TinyStories, which is simple children's stories written with a ~3-year-old's vocabulary. It writes those well. It isn't a general-purpose chatbot, and prompts far from that domain give odd results.

---

## Architecture

```
token ids ─► Embedding (10000 × 512)
               │
               ▼
     ┌───────────────────────────────────────┐
     │  Transformer block  × 4               │
     │                                       │
     │   x ─┬─► RMSNorm ─► Causal MHSA ──►(+) │   16 heads, d_k = 32, RoPE on Q and K
     │      └────────────────────────────►┘  │
     │   z ─┬─► RMSNorm ─► SwiGLU FFN  ──►(+) │   d_ff = 1344 ≈ 8/3 · d_model
     │      └────────────────────────────►┘  │
     └───────────────────────────────────────┘
               │
               ▼
           RMSNorm ─► Linear (512 × 10000) ─► next-token logits
```

| Hyperparameter | Value |
|---|---|
| Vocabulary size | 10,000 (byte-level BPE trained on TinyStories) |
| Context length | 256 tokens |
| `d_model` | 512 |
| Layers | 4 |
| Attention heads | 16 |
| `d_ff` (SwiGLU) | 1,344 |
| RoPE Θ | 10,000 |
| Parameters | 22,696,448 (17.6M excluding the input embedding matrix) |

---

## How it was built

Each piece below is implemented by hand. The table maps every component to its source file and the paper it comes from.

| Component | File | Paper |
|---|---|---|
| Byte-pair encoding | [`transformer/bpe.py`](transformer/bpe.py) | Gage, 1994; Sennrich et al., 2016 |
| Byte-level vocabulary | [`transformer/bpe.py`](transformer/bpe.py) | Wang et al., 2019; Radford et al., 2019 (GPT-2) |
| GPT-2 pre-tokenization regex | [`transformer/bpe.py`](transformer/bpe.py) | Radford et al., 2019 |
| Tokenizer encode/decode | [`transformer/tokenizer.py`](transformer/tokenizer.py) | Sennrich et al., 2016 |
| Decoder-only Transformer LM | [`transformer/model.py`](transformer/model.py) | Vaswani et al., 2017; Radford et al., 2018 |
| Scaled dot-product attention | [`transformer/model.py`](transformer/model.py) | Vaswani et al., 2017 (§3.2.1) |
| Multi-head self-attention | [`transformer/model.py`](transformer/model.py) | Vaswani et al., 2017 (§3.2.2) |
| Rotary position embeddings | [`transformer/model.py`](transformer/model.py) | Su et al., 2021 (RoFormer) |
| RMSNorm | [`transformer/model.py`](transformer/model.py) | Zhang & Sennrich, 2019 |
| Pre-norm residual blocks | [`transformer/model.py`](transformer/model.py) | Nguyen & Salazar, 2019; Xiong et al., 2020 |
| SiLU / Swish activation | [`transformer/model.py`](transformer/model.py) | Hendrycks & Gimpel, 2016; Elfwing et al., 2017 |
| Gated linear units | [`transformer/model.py`](transformer/model.py) | Dauphin et al., 2017 |
| SwiGLU feed-forward | [`transformer/model.py`](transformer/model.py) | Shazeer, 2020 |
| Bias-free linear layers, overall recipe | [`transformer/model.py`](transformer/model.py) | Touvron et al., 2023 (LLaMA); Chowdhery et al., 2022 (PaLM) |
| Cross-entropy / perplexity | [`transformer/loss.py`](transformer/loss.py) | — |
| Adam / AdamW | [`transformer/optimizers.py`](transformer/optimizers.py) | Kingma & Ba, 2015; Loshchilov & Hutter, 2019 |
| AdamW β = (0.9, 0.95) | [`scripts/train.py`](scripts/train.py) | Brown et al., 2020 (GPT-3); Touvron et al., 2023 |
| Nucleus (top-p) sampling | [`transformer/generation.py`](transformer/generation.py) | Holtzman et al., 2020 |
| Dataset | — | Eldan & Li, 2023 (TinyStories) |

### 1. Byte-level BPE tokenizer

`transformer/bpe.py`, `transformer/tokenizer.py`

Byte-pair encoding was first a compression algorithm ([Gage, 1994](#references)). [Sennrich et al. (2016)](https://arxiv.org/abs/1508.07909) adapted it to build subword vocabularies. Like GPT-2 ([Radford et al., 2019](#references); [Wang et al., 2019](https://arxiv.org/abs/1909.03341)), this tokenizer runs on **raw UTF-8 bytes**, so the 256 base tokens cover any input string and there are never unknown tokens.

**Training** (`train_bpe`):

1. **Pre-tokenization.** The corpus is split with the GPT-2 regex `'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`, so merges never cross word or punctuation boundaries. Special tokens like `<|endoftext|>` are split out first so they never get merged.
2. **Parallelism.** The 2.1 GB training file is cut into chunks at `<|endoftext|>` boundaries, and the chunks are counted in parallel with `multiprocessing.Pool`. Each worker streams matches with `re.finditer` and returns only a `Counter` of pre-token frequencies, so no worker holds the full token list in memory.
3. **Incremental merging.** Rescanning the corpus for every merge is far too slow. Instead the trainer keeps two indexes:
   - `pair_counts`: the total frequency of every adjacent byte pair, weighted by pre-token count.
   - `pair_to_words`: an inverted index from each pair to the unique pre-tokens that contain it.

   After picking the most frequent pair (ties go to the lexicographically greater pair), only the pre-tokens that contain it are rewritten. Their old pairs are subtracted from the counts and their new pairs are added. Each merge costs time proportional to the words it touches, not the size of the corpus.

**Encoding** (`Tokenizer.encode`) pre-tokenizes the text and then, within each pre-token, repeatedly applies the merge with the **lowest rank**, meaning the one learned earliest in training. This reproduces exactly the segmentation the trainer would have produced. `encode_iterable` streams arbitrarily large files by cutting the buffer only at whitespace, so a token is never split across reads. That is how the full 2.1 GB training set was tokenized into a `uint16` NumPy array (541M tokens) in bounded memory.

On TinyStories, the 10K vocabulary compresses text to about **4.0 bytes per token**.

### 2. Model components

`transformer/model.py`

**Linear and Embedding.** These are plain `nn.Parameter` matrices. Linear layers have **no bias**, as in PaLM and LLaMA, and are initialized from a truncated normal with σ² = 2 / (d_in + d_out), truncated at ±3σ. Embeddings use a truncated N(0, 1). The matrix multiplies use `einops.einsum` with named dimensions, which makes the shapes self-documenting.

**RMSNorm** ([Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)) replaces LayerNorm ([Ba et al., 2016](https://arxiv.org/abs/1607.06450)):

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\tfrac{1}{d}\sum_i x_i^2 + \varepsilon}} \odot g$$

It skips the mean subtraction and the bias, which makes it cheaper with no loss in quality. The input is upcast to float32 inside the norm so squaring can't overflow under mixed precision.

**Pre-norm residual blocks.** The original Transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) applied normalization *after* each residual add (post-norm). Following [Nguyen & Salazar (2019)](https://arxiv.org/abs/1910.05895) and [Xiong et al. (2020)](https://arxiv.org/abs/2002.04745), this model normalizes the *input* of each sub-layer instead:

$$z = x + \text{MHSA}(\text{RMSNorm}(x)), \qquad y = z + \text{FFN}(\text{RMSNorm}(z))$$

This leaves an unnormalized "residual stream" running from the embeddings to the output, which keeps gradients well-behaved. A final RMSNorm is applied before the LM head.

**SwiGLU feed-forward** ([Shazeer, 2020](https://arxiv.org/abs/2002.05202)) combines the SiLU/Swish activation ([Hendrycks & Gimpel, 2016](https://arxiv.org/abs/1606.08415); [Elfwing et al., 2017](https://arxiv.org/abs/1702.03118)) with a gated linear unit ([Dauphin et al., 2017](https://arxiv.org/abs/1612.08083)):

$$\text{FFN}(x) = W_2\big(\text{SiLU}(W_1 x) \odot W_3 x\big), \qquad \text{SiLU}(x) = x \cdot \sigma(x)$$

`d_ff = 1344 ≈ 8/3 · d_model` (rounded to a multiple of 64). This keeps the parameter count close to a standard 4× FFN despite the third matrix. Llama 3 ([Grattafiori et al., 2024](https://arxiv.org/abs/2407.21783)) and Qwen 2.5 ([Yang et al., 2024](https://arxiv.org/abs/2412.15115)) use the same design.

**Rotary position embeddings** ([Su et al., 2021](https://arxiv.org/abs/2104.09864)). Instead of adding a position vector to the input, RoPE *rotates* each consecutive pair of dimensions of the query and key vectors by an angle that depends on the token's position *i*:

$$\theta_{i,k} = \frac{i}{\Theta^{(2k-2)/d}}, \qquad k = 1, \dots, d/2$$

Because rotations preserve dot products up to the angle between them, ⟨R_m q, R_n k⟩ depends only on the **relative offset** m − n. The sin/cos tables are precomputed once and stored as non-persistent buffers. The rotation is computed on even and odd slices, not as a d×d matrix multiply. RoPE is applied after the heads are split, so it acts on each head's `d_k` independently.

**Scaled dot-product attention** (Vaswani et al., 2017, §3.2.1):

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

The causal mask *M* sets every future position to −∞ before the softmax. The softmax is written by hand and subtracts the row max first, so large logits can't overflow `exp`. **Multi-head attention** (§3.2.2) projects to Q, K and V with three `d_model × d_model` matrices, splits them into 16 heads of size 32 with `einops.rearrange`, attends within each head, then concatenates the heads and applies an output projection.

### 3. Training

`scripts/train.py`, `transformer/optimizers.py`, `transformer/loss.py`, `transformer/data.py`

- **Objective.** Next-token cross-entropy, implemented with the log-sum-exp trick (subtract the max before `exp`) for numerical stability. Validation is reported as loss and perplexity, `exp(loss)`.
- **AdamW** ([Kingma & Ba, 2015](https://arxiv.org/abs/1412.6980); [Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101)) is written as a `torch.optim.Optimizer` subclass. It keeps first and second moment estimates per parameter, applies bias correction through the step size α_t = α·√(1−β₂ᵗ)/(1−β₁ᵗ), and applies weight decay **decoupled** from the gradient update (θ ← θ − αλθ). I used β = (0.9, 0.95) as in GPT-3 and LLaMA, and λ = 0.1.
- **Learning-rate schedule.** Linear warmup for 200 steps, then cosine decay to 10% of the peak learning rate.
- **Gradient clipping.** Gradients are rescaled when their global L2 norm exceeds 1.0.
- **Data loading.** The tokenized training set (541M tokens, 1 GB) is opened with `np.load(mmap_mode="r")`, so batches of random 256-token windows are read straight from disk without loading the whole array into RAM.
- **Checkpointing and logging.** Model, optimizer state and step are saved periodically, and training can resume with `--resume`. Every run writes `config.json` plus a JSON-lines log with step, wall-clock time, train loss, validation loss and learning rate. Training stops automatically if the loss becomes non-finite.

### 4. Decoding

`transformer/generation.py`

Generation is autoregressive. Each step feeds the last ≤256 tokens through the model and samples from the final position's distribution:

- **Temperature scaling.** softmax(logits / τ). τ → 0 approaches greedy decoding (`--temperature 0` is exactly greedy).
- **Nucleus (top-p) sampling** ([Holtzman et al., 2020](https://arxiv.org/abs/1904.09751)). Keep the smallest set of most-likely tokens whose cumulative probability reaches *p*, renormalize, and sample. This removes the long tail of low-probability tokens that causes incoherent text.
- Generation stops at `<|endoftext|>` or after `max_new_tokens`.

---

## Results and experiments

The released model (`checkpoints/tinystories/`) was trained for 2,500 steps × 64 sequences × 256 tokens = **41M tokens** on a single Apple M5 laptop GPU (PyTorch MPS backend) in about 70 minutes.

| Metric | Value |
|---|---|
| Validation loss (2M held-out tokens) | **1.629** |
| Validation perplexity | **5.10** |
| Loss of a random-init model | 9.21 (= ln 10,000) |

The released weights are stored in float16 (45 MB instead of a 260 MB training checkpoint), with no measurable change in validation loss.

To understand *why* each design choice matters, I ran a set of controlled experiments. All runs use the architecture above. Logs for every run are in [`experiments/`](experiments/). To draw the learning curves, run `uv run python scripts/plot.py`, which saves them to `plots/`. The final losses in the tables below are averages over 20 random validation batches, logged during training.

### Learning-rate sweep

| Peak LR | Final val loss | Notes |
|---|---|---|
| 3e-4 | 1.870 | Stable, but too slow |
| 1e-3 | 1.665 | |
| **3e-3** | **1.643** | Best |
| 1e-2 | 2.39 @ step 2k | Converging noticeably slower; stopped early |
| 1e-1 | 4.71 @ step 2k | Plateaued high |
| 1 | diverged | Loss blew up to ~95 |

The best learning rate sits just below the point where training starts to become unstable. Going an order of magnitude past it, training doesn't diverge right away, but it settles at a much worse loss.

### Batch size

At a fixed token budget (~41M tokens), batch size 64 gave the best loss (1.637), against 1.643 for batch 32 and 1.810 for batch 8. Batch 8 had to take 8× more optimizer steps on noisier gradients. On the M5, batch 256 didn't run to completion and batch 128 was much slower per token, so 64 was the sweet spot for this hardware.

### Ablations

Each ablation can be reproduced with a flag on `scripts/train.py`.

| Variant | Flag | Final val loss | vs. baseline (1.643) |
|---|---|---|---|
| Baseline: pre-norm, RMSNorm, RoPE, SwiGLU | — | 1.643 | — |
| No RMSNorm, LR 3e-3 | `--no-rmsnorm` | diverged at step ~250 | — |
| No RMSNorm, LR 1e-3 | `--no-rmsnorm --lr 1e-3` | 1.690 | +0.047 |
| Post-norm (original Transformer) | `--post-norm` | 1.706 | +0.063 |
| No position embeddings (NoPE) | `--no-rope` | 1.813 | +0.170 |
| Ungated SiLU FFN (`d_ff = 2048`, parameter-matched) | `--silu-ffn --d-ff 2048` | 1.637 | −0.006 |

What I learned:

- **RMSNorm buys stability.** Without it, the model diverges at the best learning rate (the loss jumped to ~5×10⁹). It only trains at a lower learning rate, and it ends up worse.
- **Where the norm goes matters.** Post-norm trains, but it falls behind pre-norm throughout, consistent with [Xiong et al. (2020)](https://arxiv.org/abs/2002.04745).
- **Position information matters most.** A causal decoder *can* infer position without explicit embeddings ([Tsai et al., 2019](https://aclanthology.org/D19-1443/); [Kazemnejad et al., 2023](https://arxiv.org/abs/2305.19466)), and NoPE does learn. But it was the largest regression of any ablation.
- **Gating is close to a wash at this scale.** A parameter-matched SiLU FFN matched SwiGLU here. The gains [Shazeer (2020)](https://arxiv.org/abs/2002.05202) reports likely need larger models or longer training to appear.

---

## Training your own model

```sh
# 1. download TinyStories (~2.1 GB)
mkdir -p data && cd data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
cd ..

# 2. train a 10K byte-level BPE tokenizer -> data/tinystories_{vocab,merges}.pkl
#    (add --profile to see where time goes)
uv run python scripts/train_tokenizer.py --vocab-size 10000

# 3. tokenize train/valid into uint16 arrays -> data/tinystories_{train,valid}_tokens.npy
uv run python scripts/encode_dataset.py

# 4. train (the settings used for the released model)
uv run python scripts/train.py --batch-size 64 --max-steps 2500 --lr 3e-3 --out-dir runs/my_model

# 5. package it as a small inference-only model and sample from it
uv run python scripts/export_model.py --run-dir runs/my_model --out-dir checkpoints/my_model
uv run python scripts/generate.py --model-dir checkpoints/my_model
```

`scripts/train.py --help` lists every option: model size, optimizer hyperparameters, eval and checkpoint intervals, `--resume`, `--compile` (`torch.compile`) and the ablation flags.

---

## Tests

```sh
uv run pytest
```

The test suite checks the from-scratch components against PyTorch's own implementations and tests key properties of the model:

- `softmax`, `silu`, `RMSNorm`, attention and `cross_entropy` match their `torch` / `torch.nn.functional` counterparts.
- `AdamW` matches `torch.optim.AdamW` over 20 steps, and gradient clipping matches `clip_grad_norm_`.
- RoPE preserves vector norms, and attention scores depend only on relative position.
- Attention is causal: changing future tokens never changes past outputs.
- The parameter count matches the analytic formula, a small model can overfit a batch, and every ablation variant runs.
- The BPE trainer learns the expected merges, and the tokenizer round-trips arbitrary Unicode and handles special tokens.
- The released checkpoint loads, has low perplexity on a sample story and generates deterministic greedy output.

---

## Project structure

```
transformer-from-scratch/
├── transformer/                # the library
│   ├── model.py                # Linear, Embedding, RMSNorm, SwiGLU, RoPE, attention, TransformerLM
│   ├── bpe.py                  # parallel byte-level BPE training
│   ├── tokenizer.py            # encode / decode / streaming encode
│   ├── pretokenization.py      # split a large file into chunks at special-token boundaries
│   ├── optimizers.py           # AdamW, cosine LR schedule, gradient clipping
│   ├── loss.py                 # numerically stable cross-entropy
│   ├── data.py                 # random batch sampling from a memory-mapped token array
│   ├── generation.py           # temperature + nucleus sampling
│   ├── checkpointing.py        # save/resume training, load the pretrained model
│   └── device.py               # pick cuda / mps / cpu
├── scripts/
│   ├── train_tokenizer.py      # train BPE on TinyStories
│   ├── encode_dataset.py       # tokenize the corpus to .npy
│   ├── train.py                # training loop with logging, eval, checkpointing
│   ├── export_model.py         # training checkpoint -> small fp16 inference model
│   ├── generate.py             # CLI text generation
│   └── plot.py                 # draw learning curves from the experiment logs
├── checkpoints/tinystories/    # released model: config, fp16 weights, tokenizer
├── experiments/                # config + loss log for every experiment run
└── tests/                      # unit tests + pretrained-model smoke tests
```

---

## References

**Architecture**
- Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS 2017.
- Radford et al. [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf). 2018. (GPT)
- Radford et al. [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). 2019. (GPT-2)
- Ba, Kiros & Hinton. [Layer Normalization](https://arxiv.org/abs/1607.06450). 2016.
- Zhang & Sennrich. [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467). NeurIPS 2019.
- Nguyen & Salazar. [Transformers without Tears: Improving the Normalization of Self-Attention](https://arxiv.org/abs/1910.05895). IWSLT 2019.
- Xiong et al. [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745). ICML 2020.
- Su et al. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864). 2021.
- Dauphin et al. [Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083). 2017.
- Hendrycks & Gimpel. [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415). 2016.
- Elfwing, Uchibe & Doya. [Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning](https://arxiv.org/abs/1702.03118). 2017.
- Shazeer. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202). 2020.
- Tsai et al. [Transformer Dissection: A Unified Understanding of Transformer's Attention via the Lens of Kernel](https://aclanthology.org/D19-1443/). EMNLP 2019.
- Kazemnejad et al. [The Impact of Positional Encoding on Length Generalization in Transformers](https://arxiv.org/abs/2305.19466). NeurIPS 2023.

**Tokenization**
- Gage. A New Algorithm for Data Compression. *C Users Journal*, 1994.
- Sennrich, Haddow & Birch. [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909). ACL 2016.
- Wang, Cho & Gu. [Neural Machine Translation with Byte-Level Subwords](https://arxiv.org/abs/1909.03341). 2019.

**Optimization and training**
- Kingma & Ba. [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). ICLR 2015.
- Loshchilov & Hutter. [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101). ICLR 2019.
- Kaplan et al. [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361). 2020.
- Hoffmann et al. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556). 2022.

**Decoding**
- Holtzman et al. [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751). ICLR 2020.

**Modern LLMs whose design this follows**
- Brown et al. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165). NeurIPS 2020. (GPT-3)
- Chowdhery et al. [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311). 2022.
- Touvron et al. [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971). 2023.
- Grattafiori et al. [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783). 2024.
- Yang et al. [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115). 2024.

**Data**
- Eldan & Li. [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759). 2023.

## Acknowledgments

The project's scope and component breakdown follow the "Language Modeling from Scratch" curriculum from [Stanford CS336](https://stanford-cs336.github.io/). The file-chunking helper in `transformer/pretokenization.py` is adapted from its starter code (MIT License).

## License

[MIT](LICENSE)
