"""CPU tests for twin Q / gamma^n TD (Actor is not optimized)."""

from __future__ import annotations

import copy

import pytest
import torch

from smolvla_rltoken.rl.agent import RLTAgent
from smolvla_rltoken.rl.config import OnlineRLConfig
from smolvla_rltoken.rl.critic import ChunkCritic
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
        gamma=0.5,
        tau=0.05,
        actor_lr=1e-3,
        critic_lr=1e-3,
        ref_dropout=0.0,
        action_std=0.0,
        bc_beta=1.0,
        batch_size=8,
        use_residual_actor=False,
        device="cpu",
    )
    payload.update(overrides)
    return OnlineRLConfig(**payload)


def _batch(cfg: OnlineRLConfig, *, n_steps: int, terminated: float, reward_at: int | None = None):
    b = 4
    z = torch.randn(b, cfg.rl_token_dim)
    p = torch.randn(b, cfg.proprio_dim)
    z_n = torch.randn(b, cfg.rl_token_dim)
    p_n = torch.randn(b, cfg.proprio_dim)
    rewards = torch.zeros(b, cfg.chunk_len)
    if reward_at is not None:
        rewards[:, reward_at] = 1.0
    return {
        "x": cat_state(z, p),
        "x_next": cat_state(z_n, p_n),
        "executed_action": torch.randn(b, cfg.chunk_len, cfg.action_dim),
        "reference_action": torch.randn(b, cfg.chunk_len, cfg.action_dim),
        "next_reference_action": torch.randn(b, cfg.chunk_len, cfg.action_dim),
        "reward_sequence": rewards,
        "n_steps": torch.full((b,), n_steps, dtype=torch.long),
        "terminated": torch.full((b,), terminated),
        "truncated": torch.zeros(b),
    }


def test_critic_has_twin_heads_and_ln_relu():
    agent = RLTAgent(_cfg(), device="cpu")
    assert len(agent.critic.qs) == 2
    names = [type(mod).__name__ for mod in agent.critic.qs[0]]
    assert names.count("LayerNorm") == 2
    assert names.count("ReLU") == 2


def test_q_has_grad_wrt_executed_action():
    agent = RLTAgent(_cfg(), device="cpu")
    batch = _batch(agent.cfg, n_steps=4, terminated=0.0)
    executed = batch["executed_action"].detach().clone().requires_grad_(True)
    q = agent.critic.min_q(batch["x"], executed, batch["reference_action"])
    q.sum().backward()
    assert executed.grad is not None
    assert executed.grad.abs().sum() > 0


def test_residual_input_rescales_the_action_channel():
    agent = RLTAgent(_cfg(), device="cpu")
    batch = _batch(agent.cfg, n_steps=4, terminated=0.0)
    cfg = agent.cfg
    chunk_dim = cfg.chunk_len * cfg.action_dim
    ref, executed = batch["reference_action"], batch["executed_action"]
    features = agent.critic.input_features(batch["x"], executed, ref)

    assert features.shape[-1] == batch["x"].shape[-1] + 2 * chunk_dim
    x_dim = batch["x"].shape[-1]
    torch.testing.assert_close(features[:, x_dim : x_dim + chunk_dim], ref.flatten(1))
    scale = cfg.resolved_critic_residual_scale()
    torch.testing.assert_close(
        features[:, x_dim + chunk_dim :], (executed - ref).flatten(1) / scale
    )
    # A one-explore_std perturbation must land at magnitude 1 on that channel,
    # which is the whole point of the rescaling.
    nudged = ref + cfg.explore_std
    residual = agent.critic.input_features(batch["x"], nudged, ref)[:, x_dim + chunk_dim :]
    torch.testing.assert_close(residual, torch.ones_like(residual))


def test_critic_residual_scale_defaults_to_explore_std():
    cfg = _cfg(explore_std=0.03, critic_residual_scale=0.0)
    assert cfg.resolved_critic_residual_scale() == pytest.approx(0.03)
    assert cfg.to_actor_critic().critic_residual_scale == pytest.approx(0.03)
    explicit = _cfg(explore_std=0.03, critic_residual_scale=0.5)
    assert explicit.resolved_critic_residual_scale() == pytest.approx(0.5)
    # explore_std=0 is degenerate (no exploration at all); 1.0 keeps the divisor
    # valid instead of raising deep inside ChunkCritic.
    assert _cfg(explore_std=0.0).resolved_critic_residual_scale() == pytest.approx(1.0)


def test_residual_input_rejects_a_non_positive_scale():
    ac = _cfg().to_actor_critic()
    ac.critic_residual_scale = 0.0
    with pytest.raises(ValueError, match="critic_residual_scale"):
        ChunkCritic(ac)


def test_raw_action_input_is_blind_to_explore_std_sized_actions():
    """The failure that pinned the Actor in run_20260902_115329, as a unit test.

    The regression target depends *only* on which side of the reference the
    action sits, at exactly one explore_std. With the raw ``(x, a)`` input that
    is a 0.02 variation on an O(2) MEAN_STD value next to a 521-dim state, and
    the Critic regresses to the mean: loss stays at the target variance and
    ``Q(ref + d) - Q(ref - d)`` is 0 to four decimals. Residual coordinates
    recover the separation. See stage2_survey.md 13.5.
    """

    def fit(residual_input: bool) -> tuple[float, float]:
        torch.manual_seed(0)
        cfg = _cfg(
            rl_token_dim=512,
            proprio_dim=9,
            hidden_dim=128,
            explore_std=0.02,
            critic_residual_input=residual_input,
        )
        critic = ChunkCritic(cfg.to_actor_critic())
        opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
        direction = torch.randn(cfg.chunk_len, cfg.action_dim)
        direction /= direction.norm()
        x_dim = cfg.rl_token_dim + cfg.proprio_dim
        loss = torch.zeros(())
        for _ in range(150):
            x = torch.randn(128, x_dim)
            ref = torch.randn(128, cfg.chunk_len, cfg.action_dim) * 2.0
            side = torch.where(torch.rand(128) < 0.5, -1.0, 1.0)
            action = ref + cfg.explore_std * side.view(-1, 1, 1) * direction
            q = critic(x, action, ref)
            loss = torch.nn.functional.mse_loss(q, side.unsqueeze(0).expand_as(q))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            x = torch.randn(256, x_dim)
            ref = torch.randn(256, cfg.chunk_len, cfg.action_dim) * 2.0
            step = cfg.explore_std * direction
            gap = critic.min_q(x, ref + step, ref) - critic.min_q(x, ref - step, ref)
        return float(loss.detach()), float(gap.mean())

    raw_loss, raw_gap = fit(residual_input=False)
    res_loss, res_gap = fit(residual_input=True)

    assert raw_loss > 0.9  # target variance is 1.0: nothing was learned
    assert abs(raw_gap) < 0.01
    assert res_loss < 0.1
    assert res_gap > 1.5  # the true separation is 2.0


def test_critic_update_does_not_change_actor():
    agent = RLTAgent(_cfg(), device="cpu")
    before = {name: param.detach().clone() for name, param in agent.actor.named_parameters()}
    agent.update_critic(_batch(agent.cfg, n_steps=4, terminated=0.0, reward_at=0))
    for name, param in agent.actor.named_parameters():
        torch.testing.assert_close(param, before[name])
    assert agent._update_count == 0  # update_critic alone does not bump the actor schedule


def test_td_uses_next_reference_not_current():
    agent = RLTAgent(_cfg(chunk_len=4), device="cpu")
    batch = _batch(agent.cfg, n_steps=4, terminated=0.0)
    called = {}

    orig = agent.actor.sample

    def wrapped(x, ref_chunk, **kwargs):
        called["ref"] = ref_chunk.detach().clone()
        return orig(x, ref_chunk, **kwargs)

    agent.actor.sample = wrapped
    agent.compute_td_target(batch)
    torch.testing.assert_close(called["ref"], batch["next_reference_action"])
    assert not torch.equal(called["ref"], batch["reference_action"])


def test_terminated_zeros_bootstrap():
    agent = RLTAgent(_cfg(chunk_len=10, gamma=0.5), device="cpu")
    agent.critic_target.min_q = lambda x, a, ref: torch.ones(x.shape[0], device=x.device)
    batch = _batch(agent.cfg, n_steps=6, terminated=1.0)
    target = agent.compute_td_target(batch)
    torch.testing.assert_close(target, torch.zeros(4))


def test_n_steps_six_uses_gamma_six_not_chunk_len():
    agent = RLTAgent(_cfg(chunk_len=10, gamma=0.5), device="cpu")
    agent.critic_target.min_q = lambda x, a, ref: torch.ones(x.shape[0], device=x.device)
    y6 = agent.compute_td_target(_batch(agent.cfg, n_steps=6, terminated=0.0))
    y10 = agent.compute_td_target(_batch(agent.cfg, n_steps=10, terminated=0.0))
    torch.testing.assert_close(y6, torch.full((4,), 0.5**6))
    torch.testing.assert_close(y10, torch.full((4,), 0.5**10))
    assert y6[0].item() != y10[0].item()


def test_prior_chunk_q_rises_after_success_td():
    cfg = _cfg(chunk_len=4, gamma=0.9, tau=0.1, critic_lr=3e-3, ref_dropout=0.0, action_std=0.0)
    agent = RLTAgent(cfg, device="cpu")
    x0 = torch.ones(cfg.rl_token_dim + cfg.proprio_dim)
    x1 = -torch.ones(cfg.rl_token_dim + cfg.proprio_dim)
    a0 = torch.zeros(cfg.chunk_len, cfg.action_dim)
    a1 = torch.ones(cfg.chunk_len, cfg.action_dim)
    replay = ChunkReplayBuffer(
        64,
        rl_token_dim=cfg.rl_token_dim,
        proprio_dim=cfg.proprio_dim,
        chunk_len=cfg.chunk_len,
        action_dim=cfg.action_dim,
    )
    z0, p0 = x0[: cfg.rl_token_dim], x0[cfg.rl_token_dim :]
    z1, p1 = x1[: cfg.rl_token_dim], x1[cfg.rl_token_dim :]
    prior = ChunkTransition(
        z_rl=z0,
        proprio=p0,
        reference_action=a0,
        executed_action=a0,
        reward_sequence=torch.zeros(cfg.chunk_len),
        n_steps=cfg.chunk_len,
        next_z_rl=z1,
        next_proprio=p1,
        next_reference_action=a1,
        terminated=0.0,
        truncated=0.0,
        episode_id=0,
        chunk_id=0,
    )
    success = ChunkTransition(
        z_rl=z1,
        proprio=p1,
        reference_action=a1,
        executed_action=a1,
        reward_sequence=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        n_steps=1,
        next_z_rl=torch.zeros_like(z1),
        next_proprio=torch.zeros_like(p1),
        next_reference_action=torch.zeros_like(a1),
        terminated=1.0,
        truncated=0.0,
        episode_id=0,
        chunk_id=1,
    )
    for _ in range(16):
        replay.add(prior)
        replay.add(success)

    def q_prior() -> float:
        with torch.no_grad():
            return float(
                agent.critic.min_q(
                    x0.unsqueeze(0), a0.unsqueeze(0), a0.unsqueeze(0)
                ).item()
            )

    start = q_prior()
    actor_before = copy.deepcopy(agent.actor.state_dict())
    for _ in range(80):
        agent.update_critic(replay.sample(16))
    end = q_prior()
    assert end > start
    for key, value in agent.actor.state_dict().items():
        torch.testing.assert_close(value, actor_before[key])
