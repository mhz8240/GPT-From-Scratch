from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def maybe_init_wandb(config: dict):
    if not config.get("logging", {}).get("use_wandb", False):
        return None
    try:
        import wandb
    except ImportError:
        print("W&B logging requested but wandb is not installed; continuing without it.")
        return None
    return wandb.init(
        project=config["logging"].get("wandb_project", "gpt-from-scratch"),
        name=config["project"]["run_name"],
        config=config,
    )
