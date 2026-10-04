import argparse
import json
import math
import os
import time

import numpy as np
import torch

from transformer.checkpointing import load_checkpoint, save_checkpoint
from transformer.data import get_batch
from transformer.loss import cross_entropy
from transformer.device import default_device
from transformer.model import TransformerLM
from transformer.optimizers import AdamW, gradient_clipping, lr_cosine_schedule


def parse_args():
    parser = argparse.ArgumentParser(description="train a Transformer LM on pre-tokenized data")

    # data
    parser.add_argument("--train-data", default="data/tinystories_train_tokens.npy")
    parser.add_argument("--valid-data", default="data/tinystories_valid_tokens.npy")

    # model (defaults match the released TinyStories model)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    # architecture ablations
    parser.add_argument("--no-rmsnorm", action="store_true", help="remove every RMSNorm")
    parser.add_argument("--post-norm", action="store_true", help="post-norm blocks instead of pre-norm")
    parser.add_argument("--no-rope", action="store_true", help="no positional embeddings (NoPE)")
    parser.add_argument("--silu-ffn", action="store_true", help="ungated SiLU feed-forward instead of SwiGLU")

    # optimization
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3, help="max learning rate")
    parser.add_argument("--min-lr", type=float, default=None, help="final learning rate (default: lr / 10)")
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # logging, evaluation, checkpointing
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--out-dir", default="runs/default", help="where the log and checkpoints go")
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")

    # hardware
    parser.add_argument("--device", default=default_device(), help="cpu, mps, or cuda (default: best available)")
    parser.add_argument("--compile", action="store_true", help="torch.compile the model")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, dataset, args, device):
    """average loss over a few random validation batches"""
    model.eval()
    losses = []
    for _ in range(args.eval_batches):
        x, y = get_batch(dataset, args.batch_size, args.context_length, device)
        losses.append(cross_entropy(model(x), y).item())
    model.train()
    return sum(losses) / len(losses)


def main():
    args = parse_args()

    if args.min_lr is not None:
        min_lr = args.min_lr
    else:
        min_lr = args.lr / 10

    device = args.device

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    # memory-mapped
    train_data = np.load(args.train_data, mmap_mode="r")
    valid_data = np.load(args.valid_data, mmap_mode="r")

    model = TransformerLM.from_config(vars(args)).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay)

    start_step = 0

    # if checkpointed, get the step from that checkpoint
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer)
        print(f"resumed from {args.resume} at step {start_step}")

    # compile after loading so the checkpoint's parameter names still match
    train_model = model
    if args.compile:
        train_model = torch.compile(model, backend="aot_eager" if device == "mps" else "inductor")

    num_params = sum(p.numel() for p in model.parameters())

    # calculate the total number of tokens: batch size × total step count × context length
    tokens_total = args.batch_size * args.max_steps * args.context_length
    print(f"device: {device} | parameters: {num_params:,} | tokens to process: {tokens_total:,}")

    # one JSON object per line, to help in plotting later
    log_path = os.path.join(args.out_dir, "log.jsonl")
    log_file = open(log_path, "a")

    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    start_time = time.time()
    for step in range(start_step, args.max_steps):
        # using cosine schedule
        lr = lr_cosine_schedule(step, args.lr, min_lr, args.warmup_steps, args.max_steps)

        for group in optimizer.param_groups:
            group["lr"] = lr

        
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        loss = cross_entropy(train_model(x), y)

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        train_loss = loss.item()

        elapsed = time.time() - start_time

        # check if we reached last step
        is_last = step == args.max_steps - 1

        # stop early on divergence, safety check
        if not math.isfinite(train_loss):
            print(f"step {step}: loss is {train_loss}, stopping (diverged)")
            log_file.write(json.dumps({"step": step, "time": elapsed, "train_loss": train_loss, "lr": lr, "diverged": True}) + "\n")
            break

        if step % args.log_interval == 0 or is_last:
            print(f"step {step:6d} | train loss {train_loss:.4f} | lr {lr:.2e} | {elapsed:7.1f}s")
            log_file.write(json.dumps({"step": step, "time": elapsed, "train_loss": train_loss, "lr": lr}) + "\n")

        if (step > 0 and step % args.eval_interval == 0) or is_last:
            val_loss = evaluate(train_model, valid_data, args, device)
            print(f"step {step:6d} | val loss {val_loss:.4f} | perplexity {math.exp(val_loss):.2f}")
            log_file.write(json.dumps({"step": step, "time": time.time() - start_time, "val_loss": val_loss}) + "\n")

        log_file.flush()

        if (step > 0 and step % args.checkpoint_interval == 0) or is_last:
            save_checkpoint(model, optimizer, step + 1, os.path.join(args.out_dir, "checkpoint.pt"))

    log_file.close()
    print(f"done in {time.time() - start_time:.1f}s; log in {log_path}")


if __name__ == "__main__":
    main()
