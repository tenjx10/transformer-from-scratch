import pickle
import regex as re
from transformer.bpe import PAT, merge_word
from collections.abc import Iterable, Iterator

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = special_tokens or []

        # add any special token to vocab
        existing = set(self.vocab.values())
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in existing:
                self.vocab[len(self.vocab)] = token_bytes
                existing.add(token_bytes)

        # token bytes -> id lookup table
        self.token_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}

        # pair -> rank lookup table
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

        if self.special_tokens:
            escaped = [re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)]
            self.special_pattern = "(" + "|".join(escaped) + ")"
        else:
            self.special_pattern = None

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if self.special_pattern:
            chunks = re.split(self.special_pattern, text)
        else:
            chunks = [text]

        ids = []
        for chunk in chunks:
            if chunk in self.special_tokens:
                ids.append(self.token_to_id[chunk.encode("utf-8")])
            else:
                # step 1 pre-tokenize
                for match in re.finditer(PAT, chunk):
                    # step 2 apply the merges in order of creation
                    tokens = [bytes([b]) for b in match.group().encode("utf-8")]
                    while len(tokens) > 1:
                        # of the adjacent pairs in this pre-token, the one that was merged earliest in training
                        pair = min(zip(tokens, tokens[1:]), key=lambda p: self.merge_ranks.get(p, float("inf")))
                        if pair not in self.merge_ranks:
                            break
                        tokens = merge_word(tokens, pair, pair[0] + pair[1])
                    ids.extend(self.token_to_id[token] for token in tokens)
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for text in iterable:
            buffer += text
            # find the last non-whitespace character
            cut = len(buffer.rstrip())

            if 0 < cut < len(buffer):
                yield from self.encode(buffer[:cut])
                buffer = buffer[cut:]
        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        text_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return text_bytes.decode("utf-8", errors="replace")
