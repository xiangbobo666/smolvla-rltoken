"""CPU test for frozen Stage 1 encoder loading."""

from __future__ import annotations

from pathlib import Path

import torch

from smolvla_rltoken.rlt.config import RLTokenConfig
from smolvla_rltoken.rlt.load import load_frozen_encoder
from smolvla_rltoken.rlt.module import RLTokenModule


def test_load_frozen_encoder_eval_and_no_grad(tmp_path: Path):
    cfg = RLTokenConfig(
        vla_width=8,
        d_model=16,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        max_recon_tokens=8,
    )
    module = RLTokenModule(cfg)
    path = tmp_path / "rl_token.pt"
    torch.save({"rl_token": module.state_dict(), "config": cfg.to_dict(), "step": 1}, path)
    loaded = load_frozen_encoder(path, device="cpu")
    assert loaded.training is False
    for param in loaded.parameters():
        assert param.requires_grad is False
    z = torch.randn(2, 5, 8)
    token = loaded.rl_token(z)
    assert token.shape == (2, 16)
    assert token.requires_grad is False
