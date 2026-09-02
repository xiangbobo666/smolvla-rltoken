"""CPU tests for sparse success mapping (no ManiSkill)."""

from __future__ import annotations

import numpy as np
import pytest

from smolvla_rltoken.envs.chunk_env import (
    MockChunkEnv,
    SparseSuccessChunkEnv,
    reset_env_indices,
    sparse_success_reward,
)


class _DenseTrapEnv:
    def __init__(self):
        self.t = 0

    def reset(self, **kwargs):
        self.t = 0
        return {"t": 0}, {"success": np.array([False])}

    def step(self, action):
        del action
        self.t += 1
        success = self.t >= 2
        return (
            {"t": self.t},
            np.array([99.0], dtype=np.float32),
            np.array([success]),
            np.array([False]),
            {"success": np.array([success])},
        )


def test_sparse_success_reward_ignores_missing_and_false():
    assert sparse_success_reward({}) == 0.0
    assert sparse_success_reward({"success": False}) == 0.0
    assert sparse_success_reward({"success": True}) == 1.0
    assert sparse_success_reward({"success": np.array([True])}) == 1.0


def test_wrapper_writes_sparse_not_dense_and_splits_term_trunc():
    env = SparseSuccessChunkEnv(_DenseTrapEnv(), action_dim=8, max_episode_steps=10)
    env.reset()
    _obs, reward, terminated, truncated, info = env.step(np.zeros(8, dtype=np.float32))
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    _obs, reward, terminated, truncated, info = env.step(np.zeros(8, dtype=np.float32))
    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert bool(np.asarray(info["success"]).reshape(-1)[0]) is True


class _NeverSuccessEnv:
    def reset(self, **kwargs):
        return {"t": 0}, {"success": False}

    def step(self, action):
        del action
        return {"t": 1}, np.array([99.0], dtype=np.float32), np.array([False]), np.array([False]), {
            "success": False
        }


def test_wrapper_timeout_is_truncated_not_terminated():
    env = SparseSuccessChunkEnv(_NeverSuccessEnv(), action_dim=8, max_episode_steps=3)
    env.reset()
    for _ in range(2):
        _obs, reward, terminated, truncated, _info = env.step(np.zeros(8, dtype=np.float32))
        assert reward == 0.0
        assert terminated is False
        assert truncated is False
    _obs, reward, terminated, truncated, _info = env.step(np.zeros(8, dtype=np.float32))
    assert reward == 0.0
    assert terminated is False
    assert truncated is True


def test_mock_env_success_at_step():
    env = MockChunkEnv(success_at=3, max_episode_steps=10, action_dim=2)
    env.reset()
    for _ in range(2):
        _obs, reward, terminated, truncated, _info = env.step(np.zeros(2, dtype=np.float32))
        assert reward == 0.0
        assert not terminated
        assert not truncated
    _obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert reward == 1.0
    assert terminated
    assert not truncated
    assert info["success"] is True


def test_mock_env_batched_step_shapes():
    env = MockChunkEnv(success_at=None, max_episode_steps=5, action_dim=2, num_envs=3)
    obs, info = env.reset()
    assert obs["t"].shape == (3,)
    assert info["success"].shape == (3,)
    obs, reward, terminated, truncated, info = env.step(np.zeros((3, 2), dtype=np.float32))
    assert reward.shape == (3,)
    assert terminated.shape == (3,)
    assert truncated.shape == (3,)
    assert not terminated.any()
    assert not truncated.any()


class _BatchedNeverSuccessEnv:
    def __init__(self, num_envs=2):
        self.num_envs = num_envs

    def reset(self, **kwargs):
        del kwargs
        return {"t": 0}, {"success": np.zeros(self.num_envs, dtype=bool)}

    def step(self, action):
        del action
        zeros = np.zeros(self.num_envs, dtype=bool)
        return (
            {"t": 0},
            np.zeros(self.num_envs, dtype=np.float32),
            zeros.copy(),
            zeros.copy(),
            {"success": zeros.copy()},
        )


class _BatchedDenseTrapEnv:
    def __init__(self, num_envs=2):
        self.num_envs = num_envs
        self.t = np.zeros(num_envs, dtype=np.int32)

    def reset(self, **kwargs):
        self.t = np.zeros(self.num_envs, dtype=np.int32)
        return {"t": self.t.copy()}, {"success": np.zeros(self.num_envs, dtype=bool)}

    def step(self, action):
        del action
        self.t = self.t + 1
        success = self.t >= 2
        return (
            {"t": self.t.copy()},
            np.full(self.num_envs, 99.0, dtype=np.float32),
            success,
            np.zeros(self.num_envs, dtype=bool),
            {"success": success},
        )


def test_reset_env_indices_selects_partial_or_all():
    assert reset_env_indices(None, 4) is None
    assert reset_env_indices({}, 4) is None
    assert reset_env_indices({"reconfigure": True}, 4) is None
    # Selecting every env is a full reset.
    assert reset_env_indices({"env_idx": [0, 1]}, 2) is None
    np.testing.assert_array_equal(reset_env_indices({"env_idx": [2]}, 4), np.array([2]))
    np.testing.assert_array_equal(reset_env_indices({"env_idx": np.array([1, 3])}, 4), [1, 3])
    with pytest.raises(ValueError, match="at least one"):
        reset_env_indices({"env_idx": []}, 4)
    with pytest.raises(ValueError, match="out of range"):
        reset_env_indices({"env_idx": [4]}, 4)


def test_reset_env_indices_accepts_torch_tensor():
    torch = pytest.importorskip("torch")
    np.testing.assert_array_equal(
        reset_env_indices({"env_idx": torch.tensor([1, 2])}, 4), [1, 2]
    )


def test_partial_reset_only_restarts_selected_env_time_limit():
    env = SparseSuccessChunkEnv(
        _BatchedNeverSuccessEnv(num_envs=3), action_dim=8, max_episode_steps=4, num_envs=3
    )
    env.reset()
    for _ in range(3):
        _obs, _r, _term, trunc, _info = env.step(np.zeros((3, 8), dtype=np.float32))
        assert not trunc.any()
    env.reset(options={"env_idx": [1]})
    np.testing.assert_array_equal(np.asarray(env._elapsed), [3, 0, 3])
    _obs, _r, _term, trunc, _info = env.step(np.zeros((3, 8), dtype=np.float32))
    # Only the envs that kept their progress hit the 4-step limit.
    np.testing.assert_array_equal(trunc, [True, False, True])


def test_mock_env_partial_reset_and_reconfigure_count():
    env = MockChunkEnv(success_at=None, max_episode_steps=10, action_dim=2, num_envs=3)
    env.reset()
    for _ in range(2):
        env.step(np.zeros((3, 2), dtype=np.float32))
    assert env.t.tolist() == [2, 2, 2]
    env.reset(options={"env_idx": [0, 2]})
    assert env.t.tolist() == [0, 2, 0]
    assert env.reconfigure_count == 0
    env.reset(options={"reconfigure": True})
    assert env.t.tolist() == [0, 0, 0]
    assert env.reconfigure_count == 1


def test_batched_wrapper_writes_sparse_not_dense():
    env = SparseSuccessChunkEnv(
        _BatchedDenseTrapEnv(), action_dim=8, max_episode_steps=10, num_envs=2
    )
    env.reset()
    _obs, reward, terminated, truncated, _info = env.step(np.zeros((2, 8), dtype=np.float32))
    assert np.allclose(reward, 0.0)
    assert not terminated.any()
    _obs, reward, terminated, truncated, _info = env.step(np.zeros((2, 8), dtype=np.float32))
    assert np.allclose(reward, 1.0)
    assert terminated.all()
    assert not truncated.any()
