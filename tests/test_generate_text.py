import torch

from minigpt.config import ModelConfig
from minigpt.model import GPT


def test_model_generate_returns_longer_tensor():
    config = ModelConfig(vocab_size=20, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    model = GPT(config)
    idx = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = model.generate(idx, max_new_tokens=3, temperature=0.0)
    assert out.shape == (1, 6)
