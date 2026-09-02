"""CPU tests for Actor/Critic update order, dropout BC target, and offline warm-start."""

from __future__ import annotations

import copy

import pytest
import torch

from smolvla_rltoken.rl.agent import RLTAgent
from smolvla_rltoken.rl.config import OnlineRLConfig
from smolvla_rltoken.rl.replay import ChunkReplayBuffer
from smolvla_rltoken.rollout.transition import ChunkTransition, cat_state


def _cfg(**overrides) -> OnlineRLConfig:
    payload = dict(
        rl_token_dim=4,
        proprio_dim=3,
        chunk_len=4,
        action_dim=2,
        hidden_dim=32,
        n_layers=2,
        gamma=0.99,
        tau=0.005,
        actor_lr=1e-3,
        critic_lr=1e-3,
        ref_dropout=0.5,
        action_std=0.05,
        bc_beta=1.0,
        batch_size=8,
        critic_updates_per_actor=2,
        use_residual_actor=False,
        device="cpu",
    )
    payload.update(overrides)
    return OnlineRLConfig(**payload)


def _batch(cfg: OnlineRLConfig, batch_size: int = 8) -> dict:
    z = torch.randn(batch_size, cfg.rl_token_dim)
    p = torch.randn(batch_size, cfg.proprio_dim)
    return {
        "x": cat_state(z, p),
        "x_next": cat_state(torch.randn(batch_size, cfg.rl_token_dim), torch.randn(batch_size, cfg.proprio_dim)),
        "executed_action": torch.randn(batch_size, cfg.chunk_len, cfg.action_dim),
        "reference_action": torch.randn(batch_size, cfg.chunk_len, cfg.action_dim),
        "next_reference_action": torch.randn(batch_size, cfg.chunk_len, cfg.action_dim),
        "reward_sequence": torch.zeros(batch_size, cfg.chunk_len),
        "n_steps": torch.full((batch_size,), cfg.chunk_len, dtype=torch.long),
        "terminated": torch.zeros(batch_size),
        "truncated": torch.zeros(batch_size),
    }


def test_actor_update_changes_actor_not_critic():
    agent = RLTAgent(_cfg(), device="cpu")
    actor_before = copy.deepcopy(agent.actor.state_dict())
    critic_before = copy.deepcopy(agent.critic.state_dict())
    agent.update_actor(_batch(agent.cfg))
    actor_changed = False
    for key, value in agent.actor.state_dict().items():
        if not torch.allclose(value, actor_before[key]):
            actor_changed = True
            break
    assert actor_changed
    for key, value in agent.critic.state_dict().items():
        torch.testing.assert_close(value, critic_before[key])


def test_no_grad_around_q_blocks_actor_gradients():
    agent = RLTAgent(_cfg(ref_dropout=0.0, action_std=0.0), device="cpu")
    batch = _batch(agent.cfg)
    agent.actor.train()
    ref_in = agent.actor.apply_ref_dropout(batch["reference_action"])
    new_action = agent.actor.sample(batch["x"], ref_in)
    with torch.no_grad():
        q_detached = agent.critic.min_q(batch["x"], new_action)
    loss_wrong = -q_detached.mean()
    assert not loss_wrong.requires_grad

    agent.actor.zero_grad(set_to_none=True)
    q = agent.critic.min_q(batch["x"], new_action)
    (-q.mean()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.actor.parameters())


def test_dropout_does_not_zero_bc_target():
    agent = RLTAgent(_cfg(ref_dropout=1.0), device="cpu")
    batch = _batch(agent.cfg, batch_size=6)
    captured = {}

    orig_sample = agent.actor.sample

    def wrapped(x, ref_chunk, deterministic=False):
        captured["ref_in"] = ref_chunk.detach().clone()
        return orig_sample(x, ref_chunk, deterministic=deterministic)

    agent.actor.sample = wrapped
    agent.update_actor(batch)
    torch.testing.assert_close(captured["ref_in"], torch.zeros_like(batch["reference_action"]))
    assert batch["reference_action"].abs().sum() > 0


def test_update_runs_actor_every_two_critic_steps():
    agent = RLTAgent(_cfg(critic_updates_per_actor=2), device="cpu")
    batch = _batch(agent.cfg)
    first = agent.update(batch)
    assert "actor_loss" not in first
    second = agent.update(batch)
    assert "actor_loss" in second


def test_offline_updates_skip_when_buffer_too_small():
    cfg = _cfg(batch_size=8)
    agent = RLTAgent(cfg, device="cpu")
    replay = ChunkReplayBuffer(
        16,
        rl_token_dim=cfg.rl_token_dim,
        proprio_dim=cfg.proprio_dim,
        chunk_len=cfg.chunk_len,
        action_dim=cfg.action_dim,
    )
    replay.add(
        ChunkTransition(
            z_rl=torch.zeros(cfg.rl_token_dim),
            proprio=torch.zeros(cfg.proprio_dim),
            reference_action=torch.zeros(cfg.chunk_len, cfg.action_dim),
            executed_action=torch.zeros(cfg.chunk_len, cfg.action_dim),
            reward_sequence=torch.zeros(cfg.chunk_len),
            n_steps=cfg.chunk_len,
            next_z_rl=torch.zeros(cfg.rl_token_dim),
            next_proprio=torch.zeros(cfg.proprio_dim),
            next_reference_action=torch.zeros(cfg.chunk_len, cfg.action_dim),
            terminated=0.0,
            truncated=0.0,
            episode_id=0,
            chunk_id=0,
        )
    )
    before = copy.deepcopy(agent.actor.state_dict())
    metrics = agent.run_offline_updates(replay, n_updates=5)
    # The caller must be able to see that nothing ran, not just an empty dict.
    assert metrics["offline_updates"] == 0.0
    assert "critic_loss" not in metrics
    for key, value in agent.actor.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_offline_updates_run_when_buffer_is_full_enough():
    cfg = _cfg(batch_size=4)
    agent = RLTAgent(cfg, device="cpu")
    replay = ChunkReplayBuffer(
        32,
        rl_token_dim=cfg.rl_token_dim,
        proprio_dim=cfg.proprio_dim,
        chunk_len=cfg.chunk_len,
        action_dim=cfg.action_dim,
    )
    for index in range(8):
        replay.add(
            ChunkTransition(
                z_rl=torch.randn(cfg.rl_token_dim),
                proprio=torch.randn(cfg.proprio_dim),
                reference_action=torch.randn(cfg.chunk_len, cfg.action_dim),
                executed_action=torch.randn(cfg.chunk_len, cfg.action_dim),
                reward_sequence=torch.zeros(cfg.chunk_len),
                n_steps=cfg.chunk_len,
                next_z_rl=torch.randn(cfg.rl_token_dim),
                next_proprio=torch.randn(cfg.proprio_dim),
                next_reference_action=torch.randn(cfg.chunk_len, cfg.action_dim),
                terminated=0.0,
                truncated=0.0,
                episode_id=0,
                chunk_id=index,
            )
        )
    before = copy.deepcopy(agent.critic.state_dict())
    metrics = agent.run_offline_updates(replay, n_updates=4)
    assert "critic_loss" in metrics
    assert metrics["offline_updates"] == 4.0
    changed = any(
        not torch.allclose(agent.critic.state_dict()[key], before[key]) for key in before
    )
    assert changed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_actor_critic_tiny_cuda_forward():
    cfg = _cfg()
    agent = RLTAgent(cfg, device="cuda")
    batch = {key: value.cuda() if torch.is_tensor(value) else value for key, value in _batch(cfg).items()}
    metrics = agent.update_critic(batch)
    assert metrics["critic_loss"] >= 0.0
    q = agent.critic.min_q(batch["x"], batch["executed_action"])
    assert q.device.type == "cuda"
