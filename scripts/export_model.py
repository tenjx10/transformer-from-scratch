import argparse
import json
import os
import shutil

import torch

KEEP_CONFIG = ["vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff", "rope_theta",
               "no_rmsnorm", "post_norm", "no_rope", "silu_ffn"]


def main():
    parser = argparse.ArgumentParser(description="turn a training run into a small inference-only model directory")
    parser.add_argument("--run-dir", required=True, help="a scripts/train.py --out-dir (holds config.json and checkpoint.pt)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vocab", default="data/tinystories_vocab.pkl")
    parser.add_argument("--merges", default="data/tinystories_merges.pkl")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # drop the optimizer state (2/3 of the checkpoint) and store weights in float16: ~260 MB -> ~45 MB
    checkpoint = torch.load(os.path.join(args.run_dir, "checkpoint.pt"), map_location="cpu")
    weights = {k: v.half() for k, v in checkpoint["model"].items()}
    torch.save(weights, os.path.join(args.out_dir, "model.pt"))

    with open(os.path.join(args.run_dir, "config.json")) as f:
        config = json.load(f)
    config = {k: config[k] for k in KEEP_CONFIG if k in config}
    config["trained_steps"] = checkpoint["iteration"]
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    shutil.copy(args.vocab, os.path.join(args.out_dir, "vocab.pkl"))
    shutil.copy(args.merges, os.path.join(args.out_dir, "merges.pkl"))
    print(f"exported step {checkpoint['iteration']} of {args.run_dir} to {args.out_dir}")


if __name__ == "__main__":
    main()
