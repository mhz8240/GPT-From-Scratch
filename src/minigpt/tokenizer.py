from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def train_bpe_tokenizer(
    input_files: list[str | Path],
    output_dir: str | Path,
    vocab_size: int = 4096,
    min_frequency: int = 2,
) -> Path:
    files = [str(Path(path)) for path in input_files]
    missing = [path for path in files if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Tokenizer input files not found: {missing}")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train(files, trainer)

    output_path = Path(output_dir) / "tokenizer.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return output_path


def load_tokenizer(path: str | Path) -> Tokenizer:
    tokenizer_path = Path(path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    return Tokenizer.from_file(str(tokenizer_path))


def token_id(tokenizer: Tokenizer, token: str) -> int | None:
    return tokenizer.token_to_id(token)


def encode(
    tokenizer: Tokenizer,
    text: str,
    add_bos: bool = False,
    add_eos: bool = False,
) -> list[int]:
    ids = tokenizer.encode(text).ids
    if add_bos:
        bos_id = token_id(tokenizer, "<bos>")
        if bos_id is not None:
            ids = [bos_id] + ids
    if add_eos:
        eos_id = token_id(tokenizer, "<eos>")
        if eos_id is not None:
            ids = ids + [eos_id]
    return ids


def decode(tokenizer: Tokenizer, ids: list[int], skip_special_tokens: bool = True) -> str:
    return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
