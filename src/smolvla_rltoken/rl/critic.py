"""Twin Q critic over (x, executed chunk). No target-action smoothing in V1."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from smolvla_rltoken.rl.actor import mlp
from smolvla_rltoken.rl.config import ActorCriticConfig


class ChunkCritic(nn.Module):
    """Q_psi(x, a_{1:C}); ``forward`` returns ``[n_critics, B]``."""

    def __init__(self, cfg: ActorCriticConfig):
        super().__init__()
        self.cfg = cfg
        chunk_dim = cfg.chunk_len * cfg.action_dim
        in_dim = cfg.x_dim + chunk_dim
        self.qs = nn.ModuleList(
            [mlp(in_dim, cfg.hidden_dim, 1, cfg.n_layers) for _ in range(cfg.n_critics)]
        )

    def forward(self, x: Tensor, action_chunk: Tensor) -> Tensor:
        a_flat = action_chunk.reshape(action_chunk.shape[0], -1)
        inp = torch.cat([x, a_flat], dim=-1)
        return torch.stack([head(inp).squeeze(-1) for head in self.qs], dim=0)

    def min_q(self, x: Tensor, action_chunk: Tensor) -> Tensor:
        return self.forward(x, action_chunk).min(dim=0).values
