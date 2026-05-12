import torch

from minigpt.attention import CausalSelfAttention
from minigpt.config import ModelConfig


def test_causal_mask_prevents_future_attention():
    config = ModelConfig(vocab_size=32, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    attn = CausalSelfAttention(config)
    attn.eval()
    x = torch.randn(1, 5, 16)
    _, weights = attn(x, return_attn=True)
    future_weights = torch.triu(weights[0, 0], diagonal=1)
    assert torch.allclose(future_weights, torch.zeros_like(future_weights))
