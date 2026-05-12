from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def parameter_breakdown(model: torch.nn.Module) -> dict[str, int]:
    groups = {"embedding": 0, "attention": 0, "mlp": 0, "lm_head": 0, "other": 0}
    for name, param in model.named_parameters():
        if "token_embedding" in name or "position_embedding" in name:
            groups["embedding"] += param.numel()
        elif ".attn." in name:
            groups["attention"] += param.numel()
        elif ".mlp." in name:
            groups["mlp"] += param.numel()
        elif "lm_head" in name:
            groups["lm_head"] += param.numel()
        else:
            groups["other"] += param.numel()
    return groups


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}
