import torch

from minigpt.checkpointing import load_checkpoint, save_checkpoint
from minigpt.config import ModelConfig
from minigpt.model import GPT


def test_checkpoint_save_load_restores_model_weights(tmp_path):
    config = {"model": ModelConfig(32, 8, 1, 2, 16, 0.0).to_dict(), "data": {}}
    model = GPT(ModelConfig.from_dict(config["model"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, config, step=3, best_val_loss=1.23)

    for param in model.parameters():
        param.data.add_(1.0)

    payload = load_checkpoint(path)
    model.load_state_dict(payload["model_state_dict"])
    reloaded = GPT(ModelConfig.from_dict(config["model"]))
    reloaded.load_state_dict(payload["model_state_dict"])
    for left, right in zip(model.parameters(), reloaded.parameters(), strict=True):
        assert torch.allclose(left, right)
