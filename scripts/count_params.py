from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from minigpt.config import ModelConfig, load_config
from minigpt.model import GPT
from minigpt.utils import count_parameters, parameter_breakdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--breakdown", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    model = GPT(ModelConfig.from_dict(config["model"]))
    print(f"total_parameters: {count_parameters(model):,}")
    print(f"trainable_parameters: {count_parameters(model, trainable_only=True):,}")
    if args.breakdown:
        for name, count in parameter_breakdown(model).items():
            print(f"{name}: {count:,}")


if __name__ == "__main__":
    main()
