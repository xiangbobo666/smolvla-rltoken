"""CPU tests for Actor/Critic update order, dropout BC target, and offline warm-start."""

from __future__ import annotations

import copy
import math

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
        use_residual_actor=True,
        bc_reduction="sum",
        device="cpu",
    )
    payload.update(overrides)
    return OnlineRLConfig(**payload)


def _filled_replay(cfg: OnlineRLConfig, *, n: int) -> ChunkReplayBuffer:
    replay = ChunkReplayBuffer(
        max(n, 1),
        rl_token_dim=cfg.rl_token_dim,
        proprio_dim=cfg.proprio_dim,
        chunk_len=cfg.chunk_len,
        action_dim=cfg.action_dim,
    )
    for index in range(n):
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
    return replay


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
        q_detached = agent.critic.min_q(batch["x"], new_action, batch["reference_action"])
    loss_wrong = -q_detached.mean()
    assert not loss_wrong.requires_grad

    agent.actor.zero_grad(set_to_none=True)
    q = agent.critic.min_q(batch["x"], new_action, batch["reference_action"])
    (-q.mean()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.actor.parameters())


def test_dropout_does_not_zero_bc_target():
    agent = RLTAgent(_cfg(ref_dropout=1.0, use_residual_actor=False), device="cpu")
    batch = _batch(agent.cfg, batch_size=6)
    captured = {}

    orig = agent.actor.delta

    def wrapped(x, ref_chunk, apply_dropout=False):
        out = orig(x, ref_chunk, apply_dropout=apply_dropout)
        captured["ref_in"] = agent.actor.apply_ref_dropout(ref_chunk).detach().clone()
        return out

    agent.actor.delta = wrapped
    metrics = agent.update_actor(batch)
    torch.testing.assert_close(captured["ref_in"], torch.zeros_like(batch["reference_action"]))
    assert batch["reference_action"].abs().sum() > 0
    # The BC target is the undropped reference, so the reported distance is
    # non-zero even though the network saw nothing of the reference.
    assert metrics["bc_dist"] > 0


def test_bc_reduction_sum_is_chunk_elements_times_mean():
    agent = RLTAgent(_cfg(bc_reduction="sum"), device="cpu")
    batch = _batch(agent.cfg, batch_size=6)
    summed = agent.update_actor(batch)
    elements = agent.cfg.chunk_len * agent.cfg.action_dim
    assert summed["bc_dist"] == pytest.approx(summed["bc_dist_det"] * elements, rel=1e-5)


def test_bc_uses_the_deterministic_mean_not_the_noisy_sample():
    # With mean reduction and a residual Actor at init, mu == reference exactly,
    # so the BC term must be 0 rather than the action_std^2 noise floor.
    agent = RLTAgent(
        _cfg(bc_reduction="mean", action_std=1.0, ref_dropout=0.0), device="cpu"
    )
    metrics = agent.update_actor(_batch(agent.cfg, batch_size=6))
    assert metrics["bc_dist"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["bc_dist_det"] == pytest.approx(0.0, abs=1e-9)


def test_grad_bc_rms_is_the_analytic_bc_gradient():
    # d/dmu of beta * sum (mu - ref)^2 is 2 * beta * (mu - ref), so the reported
    # norm must equal 2 * beta * sqrt(bc_dist_det). This is what makes bc_beta
    # tunable from a log line instead of by trial and error.
    beta = 0.25
    agent = RLTAgent(
        _cfg(bc_beta=beta, use_residual_actor=False, ref_dropout=0.0), device="cpu"
    )
    metrics = agent.update_actor(_batch(agent.cfg, batch_size=6))
    expected = 2.0 * beta * math.sqrt(metrics["bc_dist_det"])
    assert metrics["grad_bc_rms"] == pytest.approx(expected, rel=1e-5)
    assert metrics["bc_dist_det"] > 0


def test_grad_ratio_is_the_q_gradient_over_the_bc_gradient():
    agent = RLTAgent(_cfg(use_residual_actor=False, ref_dropout=0.0), device="cpu")
    metrics = agent.update_actor(_batch(agent.cfg, batch_size=6))
    assert metrics["grad_ratio"] == pytest.approx(
        metrics["grad_q_rms"] / metrics["grad_bc_rms"], rel=1e-5
    )


def test_grad_q_rms_matches_a_linear_critic_gradient():
    # Q(x, a) = sum(w * a) has dQ/da = w everywhere, so the captured norm must be
    # the RMS of w. Without this the hook could be off by the batch-size factor
    # that backward() introduces via .mean().
    agent = RLTAgent(_cfg(action_std=0.0, ref_dropout=0.0), device="cpu")
    weight = torch.randn(agent.cfg.chunk_len, agent.cfg.action_dim)

    def linear_min_q(x, action, reference):
        return (action * weight).flatten(1).sum(dim=-1)

    agent.critic.min_q = linear_min_q
    metrics = agent.update_actor(_batch(agent.cfg, batch_size=6))
    assert metrics["grad_q_rms"] == pytest.approx(
        float(weight.pow(2).mean().sqrt()), rel=1e-5
    )


def test_q_adv_det_is_zero_while_the_residual_actor_sits_on_the_reference():
    # A zero-init residual Actor emits the reference exactly, so the Critic
    # cannot prefer one over the other. A non-zero q_adv_det here would mean the
    # metric is comparing different states, not different actions.
    agent = RLTAgent(_cfg(ref_dropout=0.0), device="cpu")
    metrics = agent.update_actor(_batch(agent.cfg, batch_size=6))
    assert metrics["q_adv_det"] == pytest.approx(0.0, abs=1e-7)
    assert "q_ref_mean" in metrics
    assert "q_adv_exec" in metrics


def test_q_adv_exec_compares_the_executed_action_with_the_reference():
    agent = RLTAgent(_cfg(ref_dropout=0.0), device="cpu")
    batch = _batch(agent.cfg, batch_size=6)
    metrics = agent.update_actor(batch)
    with torch.no_grad():
        ref = batch["reference_action"]
        q_ref = agent.critic.min_q(batch["x"], ref, ref)
        q_exec = agent.critic.min_q(batch["x"], batch["executed_action"], ref)
    assert metrics["q_adv_exec"] == pytest.approx(
        float((q_exec - q_ref).mean()), abs=1e-5
    )


def test_bc_pretrain_moves_a_non_residual_actor_toward_the_reference():
    cfg = _cfg(batch_size=4, use_residual_actor=False, actor_lr=3e-3)
    agent = RLTAgent(cfg, device="cpu")
    replay = _filled_replay(cfg, n=8)
    before = agent.reference_fidelity(replay)
    metrics = agent.bc_pretrain(replay, 200)
    after = agent.reference_fidelity(replay)
    assert metrics["bc_updates"] == 200.0
    assert after < before


def test_bc_pretrain_leaves_a_residual_actor_exactly_on_the_reference():
    cfg = _cfg(batch_size=4, use_residual_actor=True)
    agent = RLTAgent(cfg, device="cpu")
    replay = _filled_replay(cfg, n=8)
    assert agent.reference_fidelity(replay) == 0.0
    agent.bc_pretrain(replay, 10)
    assert agent.reference_fidelity(replay) == 0.0


def test_reference_fidelity_is_infinite_on_an_empty_buffer():
    cfg = _cfg()
    agent = RLTAgent(cfg, device="cpu")
    replay = ChunkReplayBuffer(
        4,
        rl_token_dim=cfg.rl_token_dim,
        proprio_dim=cfg.proprio_dim,
        chunk_len=cfg.chunk_len,
        action_dim=cfg.action_dim,
    )
    assert agent.reference_fidelity(replay) == float("inf")


def test_actor_and_critic_can_use_different_batches():
    agent = RLTAgent(_cfg(critic_updates_per_actor=1), device="cpu")
    critic_batch = _batch(agent.cfg)
    actor_batch = _batch(agent.cfg)
    seen: list[int] = []

    orig = agent.update_actor

    def wrapped(batch):
        seen.append(id(batch))
        return orig(batch)

    agent.update_actor = wrapped
    agent.update(lambda: critic_batch, lambda: actor_batch)
    assert seen == [id(actor_batch)]


def test_q_discrimination_metrics_split_success_slots():
    agent = RLTAgent(_cfg(), device="cpu")
    batch = _batch(agent.cfg, batch_size=8)
    flags = torch.zeros(8)
    flags[:3] = 1.0
    batch["success_slot"] = flags
    metrics = agent.update_critic(batch)
    assert "q_success_mean" in metrics
    assert "q_rest_mean" in metrics
    assert metrics["q_gap"] == pytest.approx(
        metrics["q_success_mean"] - metrics["q_rest_mean"], rel=1e-6
    )
    # A Critic collapsed to a constant is only visible through the spread.
    assert metrics["q_std"] >= 0.0


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
    q = agent.critic.min_q(
        batch["x"], batch["executed_action"], batch["reference_action"]
    )
    assert q.device.type == "cuda"
