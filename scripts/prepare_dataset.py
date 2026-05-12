from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from minigpt.config import load_config
from minigpt.data import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", nargs="*", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    metadata = prepare_dataset(config, input_files=args.input)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
