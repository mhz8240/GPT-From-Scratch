import torch

from minigpt.sampling import sample_next_token, top_k_top_p_filtering


def test_top_k_filter_keeps_expected_number_of_logits():
    logits = torch.arange(10, dtype=torch.float32).unsqueeze(0)
    filtered = top_k_top_p_filtering(logits, top_k=3)
    assert torch.isfinite(filtered).sum().item() == 3


def test_sample_next_token_returns_token_column():
    logits = torch.randn(2, 20)
    next_id = sample_next_token(logits, temperature=1.0, top_k=5)
    assert next_id.shape == (2, 1)
