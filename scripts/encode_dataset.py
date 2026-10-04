import argparse
import time

import numpy as np

from transformer.tokenizer import Tokenizer

SPECIAL_TOKEN = "<|endoftext|>"


def encode_dataset(tokenizer, input_path, output_path):
    """encode a whole text file to a uint16 numpy array, streaming it with encode_iterable"""
    start = time.time()
    with open(input_path, encoding="utf-8") as f:
        # fromiter grows a uint16 buffer, so we never hold a Python list of hundreds of millions of ints
        ids = np.fromiter(tokenizer.encode_iterable(f), dtype=np.uint16)
    np.save(output_path, ids)
    print(f"{input_path}: {len(ids):,} tokens in {time.time() - start:.1f}s -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="tokenize the TinyStories text files into .npy token arrays for training")
    parser.add_argument("--vocab", default="data/tinystories_vocab.pkl")
    parser.add_argument("--merges", default="data/tinystories_merges.pkl")
    parser.add_argument("--train", default="data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--valid", default="data/TinyStoriesV2-GPT4-valid.txt")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, [SPECIAL_TOKEN])
    # uint16 holds ids 0..65535
    assert len(tokenizer.vocab) <= 2**16

    encode_dataset(tokenizer, args.valid, "data/tinystories_valid_tokens.npy")
    encode_dataset(tokenizer, args.train, "data/tinystories_train_tokens.npy")


if __name__ == "__main__":
    main()
