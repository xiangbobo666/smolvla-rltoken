"""Non-residual Gaussian chunk Actor: a = mu_theta(x, ã), not ã + Δ."""

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
    """pi(a | x, ã) = N(mu(x, ã), sigma^2 I). Output is the final action chunk."""

    def __init__(self, cfg: ActorCriticConfig):
        super().__init__()
        self.cfg = cfg
        chunk_dim = cfg.chunk_len * cfg.action_dim
        self.net = mlp(cfg.x_dim + chunk_dim, cfg.hidden_dim, chunk_dim, cfg.n_layers)

    def mu(self, x: Tensor, ref_chunk: Tensor) -> Tensor:
        ref_flat = ref_chunk.reshape(ref_chunk.shape[0], -1)
        out = self.net(torch.cat([x, ref_flat], dim=-1))
        return out.reshape(-1, self.cfg.chunk_len, self.cfg.action_dim)

    def sample(self, x: Tensor, ref_chunk: Tensor, deterministic: bool = False) -> Tensor:
        mean = self.mu(x, ref_chunk)
        if deterministic:
            return mean
        return mean + self.cfg.action_std * torch.randn_like(mean)

    def apply_ref_dropout(self, ref_chunk: Tensor) -> Tensor:
        """Zero the Actor *input* reference for a random subset of the batch."""
        if self.cfg.ref_dropout <= 0:
            return ref_chunk
        keep = (
            torch.rand(ref_chunk.shape[0], 1, 1, device=ref_chunk.device) >= self.cfg.ref_dropout
        ).to(dtype=ref_chunk.dtype)
        return ref_chunk * keep
