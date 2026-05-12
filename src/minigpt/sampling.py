from __future__ import annotations

import torch
from torch.nn import functional as F


def apply_repetition_penalty(
    logits: torch.Tensor,
    previous_tokens: torch.Tensor | None,
    penalty: float,
) -> torch.Tensor:
    if previous_tokens is None or penalty == 1.0:
        return logits
    logits = logits.clone()
    for batch_idx in range(logits.size(0)):
        seen = torch.unique(previous_tokens[batch_idx])
        logits[batch_idx, seen] = torch.where(
            logits[batch_idx, seen] < 0,
            logits[batch_idx, seen] * penalty,
            logits[batch_idx, seen] / penalty,
        )
    return logits


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    filtered = logits.clone()
    if top_k is not None and top_k > 0:
        top_k = min(top_k, filtered.size(-1))
        threshold = torch.topk(filtered, top_k, dim=-1).values[:, -1, None]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
        sorted_mask[:, 0] = False
        indices_to_remove = torch.zeros_like(sorted_mask).scatter(1, sorted_indices, sorted_mask)
        filtered = filtered.masked_fill(indices_to_remove, float("-inf"))
    return filtered


def sample_next_token(
    logits: torch.Tensor,
    previous_tokens: torch.Tensor | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = apply_repetition_penalty(logits, previous_tokens, repetition_penalty)
    logits = logits / temperature
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
