from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from minigpt.evaluate import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    print(json.dumps(evaluate_checkpoint(args.checkpoint, args.config), indent=2))


if __name__ == "__main__":
    main()
