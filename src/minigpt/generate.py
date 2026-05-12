from __future__ import annotations

from pathlib import Path

import torch

from minigpt.checkpointing import load_checkpoint
from minigpt.config import ModelConfig, resolve_project_path
from minigpt.model import GPT
from minigpt.tokenizer import decode, encode, load_tokenizer, token_id
from minigpt.utils import get_device, set_seed


def load_model_for_generation(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> tuple[GPT, object, dict]:
    device = device or get_device()
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = GPT(ModelConfig.from_dict(config["model"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer_path = checkpoint.get("tokenizer_path") or config["data"]["tokenizer_path"]
    tokenizer = load_tokenizer(resolve_project_path(tokenizer_path))
    return model, tokenizer, checkpoint


@torch.no_grad()
def generate_text(
    model: GPT,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = 0.95,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
    stop_at_eos: bool = True,
) -> str:
    if seed is not None:
        set_seed(seed)
    device = next(model.parameters()).device
    ids = encode(tokenizer, prompt, add_bos=False, add_eos=False)
    if not ids:
        bos = token_id(tokenizer, "<bos>")
        ids = [bos if bos is not None else 0]
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    eos_id = token_id(tokenizer, "<eos>") if stop_at_eos else None
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=eos_id,
    )
    return decode(tokenizer, output[0].tolist())
