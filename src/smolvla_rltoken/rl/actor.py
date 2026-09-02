"""Gaussian chunk Actor over the VLA reference chunk.

Two parameterizations, selected by ``ActorCriticConfig.use_residual_actor``:

- residual (default): ``mu = a_tilde + Delta(x, a_tilde)`` with the last layer
  zero-initialized, so a freshly built Actor reproduces the frozen VLA chunk
  bit for bit and control handover cannot regress below the SFT baseline.
- non-residual: ``mu = mu_theta(x, a_tilde)``. Kept for ablation. A randomly
  initialized non-residual Actor outputs roughly the dataset-mean action
  regardless of the reference, which destroys the SFT prior on handover.

Reference dropout is applied *inside* the Actor and only to the network input.
The residual add always uses the undropped chunk, so dropout never turns the
residual Actor into a non-residual one.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from smolvla_rltoken.rl.config import ActorCriticConfig


def mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class ChunkActor(nn.Module):
    """pi(a | x, a_tilde) = N(mu(x, a_tilde), sigma^2 I). Output is the final chunk."""

    def __init__(self, cfg: ActorCriticConfig):
        super().__init__()
        self.cfg = cfg
        chunk_dim = cfg.chunk_len * cfg.action_dim
        self.net = mlp(cfg.x_dim + chunk_dim, cfg.hidden_dim, chunk_dim, cfg.n_layers)
        if cfg.use_residual_actor:
            # Zero the residual head so mu == a_tilde before any gradient step.
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    @property
    def is_residual(self) -> bool:
        return bool(self.cfg.use_residual_actor)

    def apply_ref_dropout(self, ref_chunk: Tensor) -> Tensor:
        """Zero the whole reference chunk for a random subset of the batch."""
        if self.cfg.ref_dropout <= 0:
            return ref_chunk
        keep = (
            torch.rand(ref_chunk.shape[0], 1, 1, device=ref_chunk.device) >= self.cfg.ref_dropout
        ).to(dtype=ref_chunk.dtype)
        return ref_chunk * keep

    def delta(self, x: Tensor, ref_chunk: Tensor, apply_dropout: bool = False) -> Tensor:
        """Raw network output, shaped like a chunk."""
        ref_input = self.apply_ref_dropout(ref_chunk) if apply_dropout else ref_chunk
        ref_flat = ref_input.reshape(ref_input.shape[0], -1)
        out = self.net(torch.cat([x, ref_flat], dim=-1))
        return out.reshape(-1, self.cfg.chunk_len, self.cfg.action_dim)

    def mu(self, x: Tensor, ref_chunk: Tensor, apply_dropout: bool = False) -> Tensor:
        out = self.delta(x, ref_chunk, apply_dropout=apply_dropout)
        if self.cfg.use_residual_actor:
            return ref_chunk + out
        return out

    def sample(
        self,
        x: Tensor,
        ref_chunk: Tensor,
        deterministic: bool = False,
        std: float | None = None,
        apply_dropout: bool = False,
    ) -> Tensor:
        """``std=None`` uses the training noise ``cfg.action_std``."""
        mean = self.mu(x, ref_chunk, apply_dropout=apply_dropout)
        sigma = self.cfg.action_std if std is None else float(std)
        if deterministic or sigma <= 0:
            return mean
        return mean + sigma * torch.randn_like(mean)
