import os

import pytest

from transformer.bpe import train_bpe
from transformer.tokenizer import Tokenizer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "tinystories")
EOT = "<|endoftext|>"


@pytest.fixture(scope="module")
def tinystories_tokenizer():
    return Tokenizer.from_files(os.path.join(MODEL_DIR, "vocab.pkl"), os.path.join(MODEL_DIR, "merges.pkl"), [EOT])


def test_train_bpe_learns_frequent_merges(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(f"low lower lowest {EOT} newer newest wider {EOT} low low low {EOT}" * 20)
    vocab, merges = train_bpe(str(corpus), vocab_size=265, special_tokens=[EOT])

    assert len(vocab) == 265
    assert EOT.encode() in vocab.values()
    assert all(len(v) == 1 for k, v in vocab.items() if k < 256)
    # " low" is the most common pre-token, so its pieces merge early
    assert b" low" in vocab.values()
    # special tokens are never split or merged across
    assert not any(b"<|" in a + b and (a + b) != EOT.encode() for a, b in merges)


def test_roundtrip(tinystories_tokenizer):
    for text in ["Once upon a time, there was a dog.", "  odd   spacing\n\nand newlines ", "émojis 🙂 and ünïcode", ""]:
        assert tinystories_tokenizer.decode(tinystories_tokenizer.encode(text)) == text


def test_special_token_is_one_id(tinystories_tokenizer):
    ids = tinystories_tokenizer.encode(f"The end.{EOT}Once")
    eot_id = tinystories_tokenizer.token_to_id[EOT.encode()]
    assert ids.count(eot_id) == 1


def test_common_words_are_single_tokens(tinystories_tokenizer):
    for word in [" the", " Once", " happy", " dog"]:
        assert len(tinystories_tokenizer.encode(word)) == 1


def test_encode_iterable_matches_encode(tinystories_tokenizer):
    lines = ["Lily went to the park.\n", "She saw a big red ball.\n", f"{EOT}\n", "The end.\n"]
    assert list(tinystories_tokenizer.encode_iterable(iter(lines))) == tinystories_tokenizer.encode("".join(lines))
