"""CPU tests for the chunk Actor (residual by default, non-residual ablation)."""

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
        use_residual_actor=True,
    )
    payload.update(overrides)
    return ActorCriticConfig(**payload)


def test_actor_mlp_is_layernorm_relu():
    actor = ChunkActor(_ac())
    names = [type(mod).__name__ for mod in actor.net]
    assert names.count("LayerNorm") == 2
    assert names.count("ReLU") == 2
    assert names[-1] == "Linear"


def test_residual_actor_reproduces_the_reference_at_init():
    # The whole point of the residual head: before any gradient step the Actor
    # is the frozen VLA, so control handover cannot regress below the SFT
    # baseline. A random non-residual Actor measured 0.9-2.4% against 15%.
    actor = ChunkActor(_ac())
    assert actor.is_residual
    x = torch.randn(3, 7)
    ref = torch.randn(3, 4, 2)
    torch.testing.assert_close(actor.mu(x, ref), ref)
    torch.testing.assert_close(actor.delta(x, ref), torch.zeros_like(ref))
    torch.testing.assert_close(actor.sample(x, ref, deterministic=True), ref)


def test_non_residual_actor_ignores_the_reference_at_init():
    actor = ChunkActor(_ac(use_residual_actor=False))
    assert not actor.is_residual
    x = torch.randn(3, 7)
    ref = torch.randn(3, 4, 2)
    mu = actor.mu(x, ref)
    assert not torch.allclose(mu, ref)


def test_dropout_hits_the_network_input_not_the_residual_add():
    actor = ChunkActor(_ac(ref_dropout=1.0))
    # Undo the zero-init so the head actually depends on its input.
    nn.init.normal_(actor.net[-1].weight)
    x = torch.zeros(5, 7)
    ref = torch.randn(5, 4, 2)
    # Dropout zeroes what the network sees, so delta changes...
    assert not torch.allclose(
        actor.delta(x, ref, apply_dropout=True), actor.delta(x, ref, apply_dropout=False)
    )
    # ...but the residual is always added to the undropped reference, so a
    # dropped batch never degenerates into a non-residual Actor.
    mu = actor.mu(x, ref, apply_dropout=True)
    torch.testing.assert_close(mu - ref, actor.delta(x, ref, apply_dropout=True))
    torch.testing.assert_close(ref, ref.clone())


def test_apply_ref_dropout_zeroes_whole_chunks():
    actor = ChunkActor(_ac(ref_dropout=1.0))
    ref = torch.randn(5, 4, 2)
    torch.testing.assert_close(actor.apply_ref_dropout(ref), torch.zeros_like(ref))
    keep = ChunkActor(_ac(ref_dropout=0.0))
    torch.testing.assert_close(keep.apply_ref_dropout(ref), ref)


def test_sample_std_overrides_action_std():
    actor = ChunkActor(_ac(action_std=10.0))
    x = torch.randn(64, 7)
    ref = torch.zeros(64, 4, 2)
    torch.testing.assert_close(actor.sample(x, ref, std=0.0), ref)
    noisy = actor.sample(x, ref, std=1.0).detach()
    assert 0.5 < float((noisy - ref).std()) < 2.0


def test_agent_accepts_both_parameterizations():
    from smolvla_rltoken.rl.agent import RLTAgent

    for residual in (True, False):
        cfg = OnlineRLConfig(
            rl_token_dim=4,
            proprio_dim=3,
            chunk_len=4,
            action_dim=2,
            hidden_dim=8,
            use_residual_actor=residual,
        )
        agent = RLTAgent(cfg, device="cpu")
        assert agent.actor.is_residual is residual
