from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from minigpt.checkpointing import load_checkpoint, save_checkpoint
from minigpt.config import ModelConfig, resolve_project_path
from minigpt.data import get_batch, load_token_array
from minigpt.evaluate import estimate_loss
from minigpt.generate import generate_text
from minigpt.logging_utils import maybe_init_wandb
from minigpt.model import GPT
from minigpt.tokenizer import load_tokenizer
from minigpt.utils import count_parameters, get_device, set_seed


def get_lr(step: int, max_steps: int, learning_rate: float, min_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return learning_rate * (step + 1) / max(1, warmup_steps)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def train(config: dict, resume: str | Path | None = None) -> dict:
    training = config["training"]
    logging = config.get("logging", {})
    model_cfg = ModelConfig.from_dict(config["model"])
    set_seed(training["seed"])
    device = get_device()

    train_data = load_token_array(config["data"]["train_bin"])
    val_data = load_token_array(config["data"]["val_bin"])
    model = GPT(model_cfg).to(device)

    if training.get("compile_model", False) and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        betas=(training["beta1"], training["beta2"]),
        weight_decay=training["weight_decay"],
    )

    step = 0
    best_val_loss = float("inf")
    if resume is not None:
        checkpoint = load_checkpoint(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        step = int(checkpoint.get("step", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))

    use_amp = bool(training.get("mixed_precision", False) and device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)
    wandb_run = maybe_init_wandb(config)

    run_name = config["project"]["run_name"]
    output_dir = resolve_project_path(config["project"]["output_dir"])
    reports_dir = resolve_project_path("reports/runs") / run_name
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict] = []
    tokenizer = None
    tokenizer_path = resolve_project_path(config["data"]["tokenizer_path"])
    if tokenizer_path.exists():
        tokenizer = load_tokenizer(tokenizer_path)

    start_time = time.time()
    tokens_per_step = training["batch_size"] * model_cfg.block_size * training["gradient_accumulation_steps"]
    pbar = tqdm(range(step, training["max_steps"]), initial=step, total=training["max_steps"])
    for current_step in pbar:
        lr = get_lr(
            current_step,
            training["max_steps"],
            training["learning_rate"],
            training["min_lr"],
            training["warmup_steps"],
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(training["gradient_accumulation_steps"]):
            x, y = get_batch(train_data, training["batch_size"], model_cfg.block_size, device)
            with autocast(device_type=device.type, enabled=use_amp):
                _, loss = model(x, y)
                assert loss is not None
                loss = loss / training["gradient_accumulation_steps"]
            scaler.scale(loss).backward()
            total_loss += loss.item()

        if training["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        step = current_step + 1
        row = {"step": step, "train_loss": total_loss, "lr": lr}

        if step % training["eval_interval"] == 0 or step == training["max_steps"]:
            val_loss = estimate_loss(
                model,
                val_data,
                training["batch_size"],
                model_cfg.block_size,
                training["eval_iters"],
                device,
            )
            row["val_loss"] = val_loss
            row["val_perplexity"] = math.exp(min(20, val_loss))
            latest_path = output_dir / "latest.pt"
            save_checkpoint(
                latest_path,
                model,
                optimizer,
                config,
                step,
                best_val_loss,
                tokenizer_path=tokenizer_path,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                row["best_val_loss"] = best_val_loss
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    config,
                    step,
                    best_val_loss,
                    tokenizer_path=tokenizer_path,
                )
            if tokenizer is not None:
                sample = generate_text(
                    model,
                    tokenizer,
                    config["generation"]["prompt"],
                    max_new_tokens=config["generation"]["max_new_tokens"],
                    temperature=config["generation"]["temperature"],
                    top_k=config["generation"]["top_k"],
                    top_p=config["generation"]["top_p"],
                    seed=training["seed"],
                )
                with (reports_dir / "samples.txt").open("a", encoding="utf-8") as f:
                    f.write(f"\n--- step {step} ---\n{sample}\n")

        elapsed = max(time.time() - start_time, 1e-9)
        row["tokens_per_sec"] = step * tokens_per_step / elapsed
        metrics.append(row)
        if wandb_run is not None:
            wandb_run.log(row)
        if step % logging.get("log_interval", 10) == 0:
            pbar.set_description(f"loss {total_loss:.4f} lr {lr:.2e}")

    metrics_path = reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    final = {
        "run_name": run_name,
        "steps": step,
        "best_val_loss": best_val_loss,
        "parameters": count_parameters(model),
        "metrics_path": str(metrics_path),
        "checkpoint_dir": str(output_dir),
        "device": str(device),
    }
    if wandb_run is not None:
        wandb_run.finish()
    return final
