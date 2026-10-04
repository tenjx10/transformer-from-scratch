import argparse

import torch

from transformer.checkpointing import load_pretrained
from transformer.device import default_device
from transformer.generation import generate


def main():
    parser = argparse.ArgumentParser(description="generate text from a trained model")
    parser.add_argument("--model-dir", default="checkpoints/tinystories",
                        help="directory with config.json, vocab.pkl, merges.pkl and model.pt (or a training checkpoint.pt)")
    parser.add_argument("--prompt", default="Once upon a time,")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--device", default=default_device(), help="cpu, mps, or cuda (default: best available)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, tokenizer = load_pretrained(args.model_dir, args.device)

    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"--- sample {i + 1} ---")
        print(generate(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, args.top_p))
        print()


if __name__ == "__main__":
    main()
