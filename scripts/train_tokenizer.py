import argparse
import cProfile
import os
import pickle
import pstats
import resource
import sys
import time

from transformer.bpe import train_bpe


def peak_memory_gb(who):
    max_rss = resource.getrusage(who).ru_maxrss
    
    if sys.platform == "darwin":
        bytes_used = max_rss
    else:
        bytes_used = max_rss * 1024
    return bytes_used / 1024**3


def main():
    parser = argparse.ArgumentParser(description="train a byte-level BPE tokenizer on TinyStories")
    parser.add_argument("--input", default="data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--profile", action="store_true", help="run under cProfile and print the slowest functions")
    args = parser.parse_args()

    special_tokens = ["<|endoftext|>"]

    start = time.time()
    if args.profile:
        profiler = cProfile.Profile()
        vocab, merges = profiler.runcall(train_bpe, args.input, args.vocab_size, special_tokens)
    else:
        vocab, merges = train_bpe(args.input, args.vocab_size, special_tokens)
    elapsed = time.time() - start

    os.makedirs(args.out_dir, exist_ok=True)
    vocab_path = os.path.join(args.out_dir, "tinystories_vocab.pkl")
    merges_path = os.path.join(args.out_dir, "tinystories_merges.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    longest = max(vocab.values(), key=len)
    print(f"training time: {elapsed:.1f}s ({elapsed / 60:.2f} min)")
    print(f"peak memory, main process: {peak_memory_gb(resource.RUSAGE_SELF):.2f} GB")
    print(f"peak memory, largest worker: {peak_memory_gb(resource.RUSAGE_CHILDREN):.2f} GB")
    print(f"vocab size: {len(vocab)}, merges: {len(merges)}")
    print(f"longest token ({len(longest)} bytes): {longest!r}")
    print(f"saved {vocab_path} and {merges_path}")

    if args.profile:
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(15)


if __name__ == "__main__":
    main()
