import os
import regex as re
from collections import Counter, defaultdict
from multiprocessing import Pool
from transformer.pretokenization import find_chunk_boundaries

# GPT-2 style pre-tokenization pattern (Radford et al., 2019)
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def count_pretokens(text, special_tokens):
    if special_tokens:
        pattern = "|".join(re.escape(s_token) for s_token in special_tokens)
        chunks = re.split(pattern, text)
    else:
        chunks = [text]

    counts = Counter()
    for chunk in chunks:
        # finditer so we never hold the full list of pre-tokens in memory
        for match in re.finditer(PAT, chunk):
            counts[match.group().encode("utf-8")] += 1
    return counts


def count_file_range(args):
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")
    return count_pretokens(text, special_tokens)


def pretokenize(input_path, special_tokens, num_processes=None):
    num_processes = num_processes or os.cpu_count() or 1
    if special_tokens:
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))
    else:
        # no special token to split on, so the whole file is one chunk
        boundaries = [0, os.path.getsize(input_path)]

    jobs = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    counts = Counter()
    with Pool(num_processes) as pool:
        for partial in pool.imap_unordered(count_file_range, jobs):
            counts.update(partial)
    return counts


def merge_word(tokens, pair, new_token):
    merged = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            merged.append(new_token)
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged


def train_bpe(input_path, vocab_size, special_tokens) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # initialize the vocabulary
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # count the pre-tokens, then work on each unique pre-token once, weighted by its frequency
    word_counts = pretokenize(input_path, special_tokens)
    words = [[bytes([b]) for b in word] for word in word_counts]
    freqs = list(word_counts.values())

    # index every adjacent pair: its total count, and which words contain it
    pair_counts = defaultdict(int)
    pair_to_words = defaultdict(set)

    for idx, tokens in enumerate(words):
        for pair in zip(tokens, tokens[1:]):
            pair_counts[pair] += freqs[idx]
            pair_to_words[pair].add(idx)

    merges = []
    while len(vocab) < vocab_size and pair_counts:
        # most frequent pair, ties broken by the lexicographically greater pair
        pair = max(pair_counts, key=lambda p: (pair_counts[p], p))

        new_token = pair[0] + pair[1]
        merges.append(pair)
        vocab[len(vocab)] = new_token

        # only words containing the pair change; swap their old pairs out of the counts and their new pairs in
        for idx in list(pair_to_words[pair]):
            old_tokens = words[idx]
            new_tokens = merge_word(old_tokens, pair, new_token)
            freq = freqs[idx]

            for p in zip(old_tokens, old_tokens[1:]):
                pair_counts[p] -= freq
                pair_to_words[p].discard(idx)
                if pair_counts[p] == 0:
                    del pair_counts[p]

            for p in zip(new_tokens, new_tokens[1:]):
                pair_counts[p] += freq
                pair_to_words[p].add(idx)
            words[idx] = new_tokens

    return vocab, merges

