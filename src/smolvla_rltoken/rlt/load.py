"""Load a frozen Stage 1 RL Token encoder for Stage 2."""

from __future__ import annotations

from pathlib import Path

import torch

from smolvla_rltoken.rlt.config import RLTokenConfig
from smolvla_rltoken.rlt.module import RLTokenModule


def load_frozen_encoder(path: str | Path, device: str | torch.device = "cpu") -> RLTokenModule:
    """Load ``rl_token.pt`` and freeze the module. Stage 2 only calls ``encode`` / ``rl_token``."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = RLTokenConfig.from_dict(ckpt["config"])
    module = RLTokenModule(cfg)
    module.load_state_dict(ckpt["rl_token"])
    module.to(device)
    module.eval()
    module.requires_grad_(False)
    return module
