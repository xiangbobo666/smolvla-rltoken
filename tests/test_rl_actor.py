"""CPU tests for the non-residual chunk Actor."""

from __future__ import annotations

import torch
from torch import nn

from smolvla_rltoken.rl.actor import ChunkActor
from smolvla_rltoken.rl.config import ActorCriticConfig, OnlineRLConfig


def _ac(**overrides) -> ActorCriticConfig:
    payload = dict(
        rl_token_dim=4,
        proprio_dim=3,
        chunk_len=4,
        action_dim=2,
        hidden_dim=16,
        n_layers=2,
        action_std=0.05,
        ref_dropout=0.5,
    )
    payload.update(overrides)
    return ActorCriticConfig(**payload)


def test_actor_mlp_is_layernorm_relu():
    actor = ChunkActor(_ac())
    names = [type(mod).__name__ for mod in actor.net]
    assert names.count("LayerNorm") == 2
    assert names.count("ReLU") == 2
    assert names[-1] == "Linear"


def test_mu_is_not_residual_add():
    actor = ChunkActor(_ac())
    nn.init.zeros_(actor.net[-1].weight)
    nn.init.zeros_(actor.net[-1].bias)
    x = torch.randn(3, 7)
    ref = torch.randn(3, 4, 2)
    mu = actor.mu(x, ref)
    torch.testing.assert_close(mu, torch.zeros_like(mu))
    assert not torch.allclose(mu, ref)


def test_dropout_only_affects_input_not_bc_target():
    actor = ChunkActor(_ac(ref_dropout=1.0))
    ref = torch.randn(5, 4, 2)
    dropped = actor.apply_ref_dropout(ref)
    torch.testing.assert_close(dropped, torch.zeros_like(ref))
    torch.testing.assert_close(ref, ref.clone())
    x = torch.zeros(5, 7)
    mu_dropped = actor.mu(x, dropped)
    mu_full = actor.mu(x, ref)
    assert not torch.allclose(mu_dropped, mu_full)


def test_online_rl_config_rejects_residual_flag_via_agent():
    from smolvla_rltoken.rl.agent import RLTAgent

    cfg = OnlineRLConfig(
        rl_token_dim=4,
        proprio_dim=3,
        chunk_len=4,
        action_dim=2,
        hidden_dim=8,
        use_residual_actor=True,
    )
    try:
        RLTAgent(cfg, device="cpu")
    except ValueError as exc:
        assert "non-residual" in str(exc)
    else:
        raise AssertionError("residual actor must be rejected")
