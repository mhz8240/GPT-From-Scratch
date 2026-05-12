from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch

from minigpt.checkpointing import load_checkpoint
from minigpt.config import ModelConfig, load_config, resolve_project_path
from minigpt.data import get_batch, load_token_array
from minigpt.model import GPT
from minigpt.utils import count_parameters, get_device


@torch.no_grad()
def estimate_loss(
    model: GPT,
    data,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, batch_size, block_size, device)
        _, loss = model(x, y)
        assert loss is not None
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / len(losses))


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    output_dir: str | Path = "reports/evals",
) -> dict:
    device = get_device()
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    config = checkpoint["config"] if config_path is None else load_config(config_path)
    model = GPT(ModelConfig.from_dict(config["model"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    val_data = load_token_array(config["data"]["val_bin"])
    training = config["training"]
    val_loss = estimate_loss(
        model,
        val_data,
        training["batch_size"],
        config["model"]["block_size"],
        training.get("eval_iters", 50),
        device,
    )
    result = {
        "checkpoint": str(resolve_project_path(checkpoint_path)),
        "run_name": config["project"]["run_name"],
        "val_loss": val_loss,
        "val_perplexity": math.exp(min(20, val_loss)),
        "parameters": count_parameters(model),
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "tokens_evaluated": int(training["batch_size"] * config["model"]["block_size"] * training.get("eval_iters", 50)),
        "device": str(device),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    eval_dir = resolve_project_path(output_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_path = eval_dir / f"{config['project']['run_name']}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["output_path"] = str(output_path)
    return result
