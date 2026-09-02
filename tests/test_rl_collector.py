"""CPU tests for late-write chunk collection."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from smolvla_rltoken.envs.chunk_env import MockChunkEnv, reset_env_indices
from smolvla_rltoken.rl.replay import ChunkReplayBuffer
from smolvla_rltoken.rollout import collector as collector_module
from smolvla_rltoken.rollout.collector import BatchedChunkCollector, ChunkCollector
from smolvla_rltoken.rollout.planner import MockPlanner


def _make(chunk_len=4, action_dim=2, success_at=None, max_episode_steps=20):
    env = MockChunkEnv(
        success_at=success_at,
        max_episode_steps=max_episode_steps,
        action_dim=action_dim,
    )
    planner = MockPlanner(
        chunk_len=chunk_len,
        action_dim=action_dim,
        rl_token_dim=4,
        proprio_dim=3,
    )
    replay = ChunkReplayBuffer(
        32,
        rl_token_dim=4,
        proprio_dim=3,
        chunk_len=chunk_len,
        action_dim=action_dim,
    )
    collector = ChunkCollector(env, planner, replay, chunk_len=chunk_len, action_dim=action_dim)
    collector.reset()
    return collector, planner, replay


def test_collector_source_has_no_intervention():
    assert "get_intervention" not in inspect.getsource(collector_module)


def test_warmup_executed_equals_reference():
    collector, planner, replay = _make()
    result = collector.run_chunk(use_actor=False)
    torch.testing.assert_close(result.executed_action, result.reference_action)
    assert result.executed_matches_reference is True
    assert result.use_actor is False
    assert planner.use_actor_calls == [False]
    assert result.n_steps == 4
    assert len(replay) == 0  # late-write waits for the next observe


def test_early_success_still_counts_as_reference_match():
    # The unexecuted tail is zero-padded, so only the executed prefix may be
    # compared; otherwise the warmup self-check fires on every early success.
    collector, _planner, _replay = _make(success_at=2)
    result = collector.run_chunk(use_actor=False)
    assert result.n_steps == 2
    assert result.executed_matches_reference is True
    assert not torch.equal(result.executed_action, result.reference_action)


def test_actor_chunk_does_not_match_reference():
    collector, _planner, _replay = _make()
    result = collector.run_chunk(use_actor=True)
    assert result.executed_matches_reference is False


def test_late_write_adds_one_and_stores_next_reference():
    collector, planner, replay = _make()
    first = collector.run_chunk(use_actor=False)
    assert first.added == 0
    second = collector.run_chunk(use_actor=False)
    assert second.added == 1
    assert len(replay) == 1
    batch = replay.sample(1)
    torch.testing.assert_close(batch["executed_action"][0], first.executed_action)
    torch.testing.assert_close(batch["reference_action"][0], first.reference_action)
    torch.testing.assert_close(batch["next_reference_action"][0], second.reference_action)
    assert not torch.equal(batch["next_reference_action"], batch["reference_action"])
    assert batch["terminated"].item() == 0.0
    assert batch["truncated"].item() == 0.0


def test_flush_pending_writes_the_trailing_chunk():
    collector, planner, replay = _make()
    collector.run_chunk(use_actor=False)
    assert len(replay) == 0
    assert collector.flush_pending() == 1
    assert len(replay) == 1
    batch = replay.sample(1)
    assert batch["terminated"].item() == 0.0
    assert batch["truncated"].item() == 0.0
    # The next state is a real encoding of the current observation.
    assert planner.observe_count == 2
    assert not torch.equal(batch["next_z_rl"][0], torch.zeros(4))
    assert collector.flush_pending() == 0
    assert len(replay) == 1


def test_finished_counters_track_episode_outcomes():
    collector, _planner, _replay = _make(success_at=2, max_episode_steps=20)
    result = collector.run_chunk(use_actor=False)
    assert result.finished_episodes == 1
    assert result.finished_successes == 1
    outcome = result.finished[0]
    assert outcome.episode_id == 0
    assert outcome.steps == 2
    assert outcome.success is True
    assert outcome.terminated is True
    assert outcome.truncated is False
    assert outcome.use_actor is False


def test_finished_episode_accumulates_steps_and_actor_flag():
    collector, _planner, _replay = _make(success_at=6, max_episode_steps=20)
    collector.run_chunk(use_actor=False)
    result = collector.run_chunk(use_actor=True)
    assert result.finished_episodes == 1
    outcome = result.finished[0]
    assert outcome.steps == 6
    # A mixed episode counts as Actor-controlled so the VLA rate stays clean.
    assert outcome.use_actor is True


def test_timeout_episode_is_recorded_as_failure():
    collector, _planner, _replay = _make(success_at=None, max_episode_steps=3)
    result = collector.run_chunk(use_actor=False)
    assert result.finished_episodes == 1
    assert result.finished_successes == 0
    outcome = result.finished[0]
    assert outcome.steps == 3
    assert outcome.truncated is True
    assert outcome.terminated is False


def test_success_mid_chunk_pads_and_sets_terminated():
    collector, _planner, replay = _make(success_at=2, max_episode_steps=20)
    result = collector.run_chunk(use_actor=False)
    assert result.n_steps == 2
    assert result.terminated is True
    assert result.truncated is False
    assert result.added == 1
    assert len(replay) == 1
    batch = replay.sample(1)
    assert batch["n_steps"].item() == 2
    assert batch["terminated"].item() == 1.0
    assert batch["truncated"].item() == 0.0
    rewards = batch["reward_sequence"][0]
    torch.testing.assert_close(rewards, torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert torch.count_nonzero(batch["executed_action"][0][2:]) == 0


def test_timeout_is_truncated_and_encodes_real_next():
    collector, planner, replay = _make(success_at=None, max_episode_steps=3)
    result = collector.run_chunk(use_actor=False)
    assert result.n_steps == 3
    assert result.truncated is True
    assert result.terminated is False
    assert result.added == 1
    batch = replay.sample(1)
    assert batch["terminated"].item() == 0.0
    assert batch["truncated"].item() == 1.0
    # Truncation still stores a real next reference (observe after last step).
    assert planner.observe_count == 2
    assert not torch.equal(batch["next_z_rl"][0], torch.zeros(4))


def test_actor_flag_changes_executed():
    collector, planner, _replay = _make()
    warm = collector.run_chunk(use_actor=False)
    actor = collector.run_chunk(use_actor=True)
    torch.testing.assert_close(warm.executed_action, warm.reference_action)
    torch.testing.assert_close(actor.executed_action, actor.reference_action + 1.0)
    assert planner.use_actor_calls == [False, True]


def _make_batched(num_envs=2, chunk_len=4, action_dim=2, success_at=None, max_episode_steps=20):
    env = MockChunkEnv(
        success_at=success_at,
        max_episode_steps=max_episode_steps,
        action_dim=action_dim,
        num_envs=num_envs,
    )
    planner = MockPlanner(
        chunk_len=chunk_len,
        action_dim=action_dim,
        rl_token_dim=4,
        proprio_dim=3,
    )
    replay = ChunkReplayBuffer(
        32,
        rl_token_dim=4,
        proprio_dim=3,
        chunk_len=chunk_len,
        action_dim=action_dim,
    )
    collector = BatchedChunkCollector(
        env, planner, replay, chunk_len=chunk_len, action_dim=action_dim, num_envs=num_envs
    )
    collector.reset(seed=0)
    return collector, planner, replay


def test_batched_late_write_adds_one_per_env():
    collector, _planner, replay = _make_batched()
    first = collector.run_chunk(use_actor=False)
    assert first.added == 0
    assert first.n_steps == 8
    second = collector.run_chunk(use_actor=False)
    assert second.added == 2
    assert len(replay) == 2


def test_batched_success_mid_chunk_pads_and_sets_terminated():
    collector, _planner, replay = _make_batched(success_at=2)
    result = collector.run_chunk(use_actor=False)
    assert result.n_steps == 4
    assert result.terminated is True
    assert result.added == 2
    assert len(replay) == 2
    batch = replay.sample(2)
    assert torch.all(batch["n_steps"] == 2)
    assert torch.all(batch["terminated"] == 1.0)


class _StaggeredSuccessEnv:
    """Batched mock where each env succeeds at its own step.

    Mirrors ManiSkill: a finished env keeps reporting ``info['success']=True``
    until it is reset, and ``options['env_idx']`` performs a partial reset.
    """

    def __init__(self, success_at: list[int], *, action_dim=2, max_episode_steps=20):
        self.success_at = np.asarray(success_at, dtype=np.int32)
        self.num_envs = len(success_at)
        self.action_dim = action_dim
        self.max_episode_steps = max_episode_steps
        self.t = np.zeros(self.num_envs, dtype=np.int32)
        self.reconfigure_count = 0
        self.step_calls = 0

    def reset(self, **kwargs):
        options = kwargs.get("options") or {}
        self.reconfigure_count += bool(options.get("reconfigure", False))
        env_idx = reset_env_indices(options, self.num_envs)
        if env_idx is None:
            self.t = np.zeros(self.num_envs, dtype=np.int32)
        else:
            self.t = self.t.copy()
            self.t[env_idx] = 0
        return {"t": self.t.copy()}, {"success": np.zeros(self.num_envs, dtype=bool)}

    def step(self, action):
        np.asarray(action, dtype=np.float32).reshape(self.num_envs, self.action_dim)
        self.step_calls += 1
        self.t = self.t + 1
        success = self.t >= self.success_at
        truncated = (self.t >= self.max_episode_steps) & ~success
        return (
            {"t": self.t.copy()},
            success.astype(np.float32),
            success.copy(),
            truncated,
            {"success": success.copy()},
        )


def _make_staggered(
    success_at,
    chunk_len=4,
    action_dim=2,
    max_episode_steps=20,
    reconfigure_every_episodes=0,
):
    env = _StaggeredSuccessEnv(
        success_at, action_dim=action_dim, max_episode_steps=max_episode_steps
    )
    planner = MockPlanner(
        chunk_len=chunk_len, action_dim=action_dim, rl_token_dim=4, proprio_dim=3
    )
    replay = ChunkReplayBuffer(
        32, rl_token_dim=4, proprio_dim=3, chunk_len=chunk_len, action_dim=action_dim
    )
    collector = BatchedChunkCollector(
        env,
        planner,
        replay,
        chunk_len=chunk_len,
        action_dim=action_dim,
        num_envs=len(success_at),
        reconfigure_every_episodes=reconfigure_every_episodes,
    )
    collector.reset()
    return collector, env, replay


def test_batched_finished_env_restarts_without_external_reset():
    collector, env, replay = _make_staggered([2, 6])
    first = collector.run_chunk(use_actor=False)
    assert first.finished_episodes == 1
    assert first.finished_successes == 1
    assert first.added == 1  # only the finished env's terminal transition
    assert first.n_steps == 2 + 4  # env0 stopped at step 2, env1 ran the chunk
    # The collector restarts finished envs itself; the caller must not reset.
    assert first.episode_done is False
    assert first.finished[0].env_index == 0
    # env0 was partially reset, env1 kept its progress.
    assert env.t.tolist() == [0, 4]

    second = collector.run_chunk(use_actor=False)
    # env0 ran a fresh episode and succeeded again; env1 finished its first.
    assert second.finished_episodes == 2
    by_env = {outcome.env_index: outcome for outcome in second.finished}
    assert by_env[0].steps == 2  # fresh episode, not accumulated
    assert by_env[1].steps == 6
    assert len(replay) == 4
    batch = replay._gather(torch.arange(len(replay)))
    assert int(batch["terminated"].sum()) == 3


def test_batched_episode_ids_are_globally_unique():
    collector, _env, _replay = _make_staggered([2, 6])
    seen: list[int] = []
    for _ in range(4):
        result = collector.run_chunk(use_actor=False)
        seen.extend(outcome.episode_id for outcome in result.finished)
    assert len(seen) == len(set(seen)), seen
    assert len(seen) >= 3


def test_batched_periodic_reconfigure_refreshes_geometry_and_drops_pending():
    # ManiSkill freezes peg geometry for num_envs > 1, so a full reconfigure has
    # to be forced periodically; it invalidates in-flight next states.
    collector, env, replay = _make_staggered([2, 6], reconfigure_every_episodes=1)
    assert env.reconfigure_count == 0
    result = collector.run_chunk(use_actor=False)
    assert result.reconfigured is True
    assert env.reconfigure_count == 1
    # env1 was mid-episode: its pending chunk is dropped, not bootstrapped across.
    assert all(pending is None for pending in collector._pending)
    assert env.t.tolist() == [0, 0]
    assert len(replay) == 1  # only env0's terminal transition


def test_batched_reconfigure_disabled_keeps_partial_reset():
    collector, env, _replay = _make_staggered([2, 6], reconfigure_every_episodes=0)
    result = collector.run_chunk(use_actor=False)
    assert result.reconfigured is False
    assert env.reconfigure_count == 0


def test_batched_flush_pending_writes_every_env():
    collector, _env, replay = _make_staggered([50, 50])
    collector.run_chunk(use_actor=False)
    assert len(replay) == 0
    assert collector.flush_pending() == 2
    assert len(replay) == 2
    assert collector.flush_pending() == 0


class _TorchInfoEnv:
    def __init__(self, num_envs=2, action_dim=2):
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.t = np.zeros(num_envs, dtype=np.int32)

    def reset(self, **kwargs):
        self.t = np.zeros(self.num_envs, dtype=np.int32)
        return {"t": self.t.copy()}, {"success": torch.zeros(self.num_envs)}

    def step(self, action):
        del action
        self.t = self.t + 1
        reward = torch.zeros(self.num_envs)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        success = torch.zeros(self.num_envs)
        return {"t": self.t.copy()}, reward, terminated, truncated, {"success": success}


def test_batched_collector_accepts_torch_step_outputs():
    env = _TorchInfoEnv()
    planner = MockPlanner(chunk_len=2, action_dim=2, rl_token_dim=4, proprio_dim=3)
    replay = ChunkReplayBuffer(8, rl_token_dim=4, proprio_dim=3, chunk_len=2, action_dim=2)
    collector = BatchedChunkCollector(
        env, planner, replay, chunk_len=2, action_dim=2, num_envs=2
    )
    collector.reset()
    result = collector.run_chunk(use_actor=False)
    assert result.n_steps == 4
    assert result.terminated is False
