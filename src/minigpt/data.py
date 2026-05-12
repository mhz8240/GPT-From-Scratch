from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from minigpt.config import resolve_project_path
from minigpt.tokenizer import encode, load_tokenizer, train_bpe_tokenizer

DEBUG_TEXT = """Once upon a time there was a small model learning to tell stories.
The model read tiny sentences, guessed the next token, and improved one step at a time.
Every batch was small, every run was quick, and the training loop stayed easy to inspect.
"""


def write_debug_corpus(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(DEBUG_TEXT * 200, encoding="utf-8")
    return path


def _dtype_for_vocab(vocab_size: int) -> np.dtype:
    return np.uint16 if vocab_size <= np.iinfo(np.uint16).max else np.uint32


def read_local_text_files(paths: Iterable[str | Path]) -> list[str]:
    texts = []
    for path in paths:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")
        texts.append(file_path.read_text(encoding="utf-8"))
    return texts


def tokenize_and_split(
    texts: list[str],
    tokenizer_path: str | Path,
    train_bin: str | Path,
    val_bin: str | Path,
    vocab_size: int,
    val_fraction: float = 0.1,
) -> dict[str, int]:
    tokenizer = load_tokenizer(tokenizer_path)
    all_ids: list[int] = []
    for text in texts:
        all_ids.extend(encode(tokenizer, text, add_eos=True))

    if len(all_ids) < 4:
        raise ValueError("Need at least 4 tokens to create a train/val split.")

    split_idx = max(1, int(len(all_ids) * (1.0 - val_fraction)))
    train_ids = np.array(all_ids[:split_idx], dtype=_dtype_for_vocab(vocab_size))
    val_ids = np.array(all_ids[split_idx:], dtype=_dtype_for_vocab(vocab_size))

    train_path = resolve_project_path(train_bin)
    val_path = resolve_project_path(val_bin)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)
    return {"train_tokens": int(train_ids.size), "val_tokens": int(val_ids.size)}


def tokenize_to_bins(
    train_texts: list[str],
    val_texts: list[str],
    tokenizer_path: str | Path,
    train_bin: str | Path,
    val_bin: str | Path,
    vocab_size: int,
) -> dict[str, int]:
    tokenizer = load_tokenizer(tokenizer_path)
    train_ids: list[int] = []
    val_ids: list[int] = []
    for text in train_texts:
        train_ids.extend(encode(tokenizer, text, add_eos=True))
    for text in val_texts:
        val_ids.extend(encode(tokenizer, text, add_eos=True))

    if len(train_ids) < 2 or len(val_ids) < 2:
        raise ValueError("Need at least 2 train and 2 validation tokens.")

    train_arr = np.array(train_ids, dtype=_dtype_for_vocab(vocab_size))
    val_arr = np.array(val_ids, dtype=_dtype_for_vocab(vocab_size))
    train_path = resolve_project_path(train_bin)
    val_path = resolve_project_path(val_bin)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    train_arr.tofile(train_path)
    val_arr.tofile(val_path)
    return {"train_tokens": int(train_arr.size), "val_tokens": int(val_arr.size)}


def load_token_array(path: str | Path) -> np.ndarray:
    file_path = resolve_project_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Tokenized data file not found: {file_path}")
    dtype = np.uint16 if file_path.stat().st_size % np.dtype(np.uint16).itemsize == 0 else np.uint32
    return np.fromfile(file_path, dtype=dtype)


def get_batch(
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size:
        raise ValueError(f"Dataset has {len(data)} tokens but block_size is {block_size}.")
    indices = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in indices])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in indices])
    return x.to(device), y.to(device)


def load_tinystories(max_train_examples: int | None, max_val_examples: int | None) -> tuple[list[str], list[str]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets to prepare TinyStories: pip install datasets") from exc

    train_split = "train"
    val_split = "validation"
    if max_train_examples:
        train_split = f"train[:{max_train_examples}]"
    if max_val_examples:
        val_split = f"validation[:{max_val_examples}]"

    dataset = load_dataset("roneneldan/TinyStories", split={"train": train_split, "val": val_split})
    return list(dataset["train"]["text"]), list(dataset["val"]["text"])


def prepare_dataset(config: dict, input_files: list[str | Path] | None = None) -> dict:
    data_cfg = config["data"]
    tok_cfg = config["tokenizer"]
    model_cfg = config["model"]

    tokenizer_path = resolve_project_path(data_cfg["tokenizer_path"])
    if data_cfg["dataset"] == "local_text":
        if input_files:
            files = [resolve_project_path(path) for path in input_files]
        else:
            files = [write_debug_corpus(resolve_project_path(data_cfg["data_dir"]) / "debug_corpus.txt")]
        texts = read_local_text_files(files)
        if not tokenizer_path.exists():
            train_bpe_tokenizer(files, tokenizer_path.parent, tok_cfg["vocab_size"], tok_cfg["min_frequency"])
        stats = tokenize_and_split(
            texts,
            tokenizer_path,
            data_cfg["train_bin"],
            data_cfg["val_bin"],
            model_cfg["vocab_size"],
        )
    elif data_cfg["dataset"] == "tinystories":
        train_texts, val_texts = load_tinystories(
            data_cfg.get("max_train_examples"),
            data_cfg.get("max_val_examples"),
        )
        corpus_dir = resolve_project_path(data_cfg["data_dir"]) / "raw"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        train_txt = corpus_dir / "train.txt"
        val_txt = corpus_dir / "val.txt"
        train_txt.write_text("\n".join(train_texts), encoding="utf-8")
        val_txt.write_text("\n".join(val_texts), encoding="utf-8")
        if not tokenizer_path.exists():
            train_bpe_tokenizer([train_txt], tokenizer_path.parent, tok_cfg["vocab_size"], tok_cfg["min_frequency"])
        stats = tokenize_to_bins(
            train_texts,
            val_texts,
            tokenizer_path,
            data_cfg["train_bin"],
            data_cfg["val_bin"],
            model_cfg["vocab_size"],
        )
    else:
        raise ValueError(f"Unsupported dataset mode: {data_cfg['dataset']}")

    metadata = {
        "dataset": data_cfg["dataset"],
        "tokenizer_path": str(tokenizer_path),
        "vocab_size": model_cfg["vocab_size"],
        "train_tokens": stats["train_tokens"],
        "val_tokens": stats["val_tokens"],
        "date_prepared": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = resolve_project_path(data_cfg["data_dir"]) / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
