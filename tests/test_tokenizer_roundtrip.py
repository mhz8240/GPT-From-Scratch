from pathlib import Path

from minigpt.tokenizer import decode, encode, load_tokenizer, train_bpe_tokenizer


def test_tokenizer_roundtrip(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\nhello tiny transformer\n", encoding="utf-8")
    tokenizer_path = train_bpe_tokenizer([corpus], tmp_path / "tok", vocab_size=64, min_frequency=1)
    tokenizer = load_tokenizer(tokenizer_path)
    text = "hello world"
    ids = encode(tokenizer, text)
    assert ids
    assert decode(tokenizer, ids) == text
