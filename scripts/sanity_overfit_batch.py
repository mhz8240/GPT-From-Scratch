from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import torch

from minigpt.config import ModelConfig, load_config
from minigpt.model import GPT
from minigpt.utils import get_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=25)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    device = get_device()
    model = GPT(ModelConfig.from_dict(config["model"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    block_size = config["model"]["block_size"]
    batch_size = config["training"]["batch_size"]
    vocab_size = config["model"]["vocab_size"]
    x = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
    y = torch.roll(x, shifts=-1, dims=1)

    first_loss = None
    last_loss = None
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        assert loss is not None
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
    print(f"first_loss={first_loss:.4f} last_loss={last_loss:.4f} device={device}")


if __name__ == "__main__":
    main()
