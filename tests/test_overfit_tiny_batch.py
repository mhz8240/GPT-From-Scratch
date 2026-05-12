import torch

from minigpt.config import ModelConfig
from minigpt.model import GPT


class Encoded:
    def __init__(self, ids):
        self.ids = ids


class DummyTokenizer:
    def encode(self, text):
        ids = [int(token) % 32 for token in text.split() if token.isdigit()]
        return Encoded(ids or [1])

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)

    def token_to_id(self, token):
        return None


def test_generation_returns_non_empty_string():
    from minigpt.generate import generate_text

    config = ModelConfig(vocab_size=32, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    model = GPT(config)
    text = generate_text(model, DummyTokenizer(), "1 2", max_new_tokens=4, temperature=1.0, seed=123)
    assert isinstance(text, str)
    assert text


def test_tiny_batch_overfit_smoke_reduces_loss():
    torch.manual_seed(0)
    config = ModelConfig(vocab_size=16, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    x = torch.randint(0, config.vocab_size, (4, config.block_size))
    y = torch.roll(x, shifts=-1, dims=1)
    losses = []
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        assert loss is not None
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0]
