from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from minigpt.config import load_config
from minigpt.train import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    result = train(load_config(args.config), resume=args.resume)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
