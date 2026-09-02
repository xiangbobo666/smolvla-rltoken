"""Twin Q critic over (x, executed chunk). No target-action smoothing in V1.

``critic_residual_input`` (default) hands the action to the network as
``(a - a_tilde) / critic_residual_scale`` next to the reference chunk itself,
rather than as the raw ``a``. Since ``a = a_tilde + Delta`` this is strictly
more information than the raw form, but the point is the rescaling: the rollout
action is the reference plus an ``explore_std``-sized perturbation, so the raw
channel is an O(1)-O(5) MEAN_STD value whose variation is only ~0.02. Against a
521-dim state that 250:1 dynamic range is quantized away by the first
``Linear -> LayerNorm``, and Q can reach a 3e-4 TD loss as a pure V(x) -- which
is what ``run_20260902_115329`` measured: ``q_gap=0.60`` discriminating states
against ``q_adv_det=0.0013`` discriminating actions, the latter 14x below the
Critic's own TD residual. An action-blind Q hands the Actor a pure-noise
gradient, which pins it at the BC stationary point. See stage2_survey.md 13.5.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from smolvla_rltoken.rl.actor import mlp
from smolvla_rltoken.rl.config import ActorCriticConfig


class ChunkCritic(nn.Module):
    """Q_psi(x, a_{1:C}); ``forward`` returns ``[n_critics, B]``.

    ``reference_chunk`` is required even when ``critic_residual_input`` is off,
    so an omitted reference is a signature error rather than a silent residual
    of ``a / scale``.
    """

    def __init__(self, cfg: ActorCriticConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.critic_residual_input and cfg.critic_residual_scale <= 0:
            raise ValueError(
                f"critic_residual_scale must be > 0, got {cfg.critic_residual_scale!r}"
            )
        chunk_dim = cfg.chunk_len * cfg.action_dim
        in_dim = cfg.x_dim + chunk_dim
        if cfg.critic_residual_input:
            in_dim += chunk_dim
        self.qs = nn.ModuleList(
            [mlp(in_dim, cfg.hidden_dim, 1, cfg.n_layers) for _ in range(cfg.n_critics)]
        )

    def input_features(
        self, x: Tensor, action_chunk: Tensor, reference_chunk: Tensor
    ) -> Tensor:
        a_flat = action_chunk.reshape(action_chunk.shape[0], -1)
        if not self.cfg.critic_residual_input:
            return torch.cat([x, a_flat], dim=-1)
        ref_flat = reference_chunk.reshape(reference_chunk.shape[0], -1)
        residual = (a_flat - ref_flat) / self.cfg.critic_residual_scale
        return torch.cat([x, ref_flat, residual], dim=-1)

    def forward(
        self, x: Tensor, action_chunk: Tensor, reference_chunk: Tensor
    ) -> Tensor:
        inp = self.input_features(x, action_chunk, reference_chunk)
        return torch.stack([head(inp).squeeze(-1) for head in self.qs], dim=0)

    def min_q(
        self, x: Tensor, action_chunk: Tensor, reference_chunk: Tensor
    ) -> Tensor:
        return self.forward(x, action_chunk, reference_chunk).min(dim=0).values
