from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from minigpt.config import resolve_project_path
from minigpt.utils import git_commit_hash


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: dict[str, Any],
    step: int,
    best_val_loss: float,
    tokenizer_path: str | Path | None = None,
) -> Path:
    checkpoint_path = resolve_project_path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": config,
        "step": step,
        "best_val_loss": best_val_loss,
        "tokenizer_path": str(tokenizer_path) if tokenizer_path is not None else None,
        "git_commit": git_commit_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint_path = resolve_project_path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def find_latest_checkpoint(directory: str | Path) -> Path | None:
    checkpoint_dir = resolve_project_path(directory)
    latest = checkpoint_dir / "latest.pt"
    if latest.exists():
        return latest
    checkpoints = sorted(checkpoint_dir.glob("*.pt"), key=lambda path: path.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None
