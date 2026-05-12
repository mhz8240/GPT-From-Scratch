from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from minigpt.config import load_config, resolve_project_path
from minigpt.tokenizer import train_bpe_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocab-size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    vocab_size = args.vocab_size or config["tokenizer"]["vocab_size"]
    output = train_bpe_tokenizer(
        [resolve_project_path(path) for path in args.input],
        resolve_project_path(args.output_dir),
        vocab_size=vocab_size,
        min_frequency=config["tokenizer"]["min_frequency"],
    )
    print(f"Saved tokenizer to {output}")


if __name__ == "__main__":
    main()
