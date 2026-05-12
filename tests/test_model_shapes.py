import torch

from minigpt.config import ModelConfig
from minigpt.model import GPT


def test_model_forward_shapes_and_loss():
    config = ModelConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = GPT(config)
    x = torch.randint(0, config.vocab_size, (3, config.block_size))
    y = torch.randint(0, config.vocab_size, (3, config.block_size))
    logits, loss = model(x, y)
    assert logits.shape == (3, config.block_size, config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0
    assert torch.isfinite(loss)
