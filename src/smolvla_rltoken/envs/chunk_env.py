"""Chunk-level env wrapper: sparse success reward, separate terminated/truncated.

V1 does not use dense ManiSkill shaping as the RL target. ``get_intervention``
is intentionally absent.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from smolvla_rltoken.vla.evaluation import EXPECTED_ACTION_DIM, scalar_bool


def sparse_success_reward(info: Mapping[str, Any] | None) -> float:
    """r_t = 1 iff info['success'] else 0. Ignores env.reward."""
    if not info:
        return 0.0
    return float(scalar_bool(info.get("success", False), name="info.success"))


def reset_env_indices(options: Mapping[str, Any] | None, num_envs: int) -> np.ndarray | None:
    """Env indices a reset touches. ``None`` means all of them.

    ManiSkill performs a partial reset when ``options["env_idx"]`` is given;
    everything else in the reset path stays batched, so the wrapper has to mirror
    that selection in its own per-env bookkeeping.
    """
    if not options or "env_idx" not in options:
        return None
    raw = options["env_idx"]
    if hasattr(raw, "detach"):
        raw = raw.detach()
    if hasattr(raw, "cpu"):
        raw = raw.cpu()
    idx = np.asarray(raw, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        raise ValueError("options['env_idx'] must select at least one environment")
    if idx.min() < 0 or idx.max() >= num_envs:
        raise ValueError(f"options['env_idx'] out of range for num_envs={num_envs}: {idx.tolist()}")
    if idx.size == num_envs:
        return None
    return idx


class SparseSuccessChunkEnv:
    """8-D actions in, sparse success reward out. ``num_envs=1`` stays scalar."""

    def __init__(
        self,
        env: Any,
        *,
        action_dim: int = EXPECTED_ACTION_DIM,
        max_episode_steps: int = 200,
        dense_reward_debug: bool = False,
        num_envs: int = 1,
    ):
        if num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        self.env = env
        self.action_dim = action_dim
        self.max_episode_steps = max_episode_steps
        self.dense_reward_debug = dense_reward_debug
        self.num_envs = int(num_envs)
        self._elapsed: int | np.ndarray = 0 if self.num_envs == 1 else np.zeros(self.num_envs, dtype=np.int32)

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Full reset, or a ManiSkill partial reset when ``options['env_idx']`` is set."""
        env_idx = reset_env_indices(kwargs.get("options"), self.num_envs)
        obs, info = self.env.reset(**kwargs)
        if self.num_envs == 1 or env_idx is None:
            self._elapsed = 0 if self.num_envs == 1 else np.zeros(self.num_envs, dtype=np.int32)
        else:
            # Partial reset: only the selected envs restart their time limit,
            # matching ManiSkill's per-env ``_elapsed_steps``.
            elapsed = np.asarray(self._elapsed, dtype=np.int32).reshape(self.num_envs).copy()
            elapsed[env_idx] = 0
            self._elapsed = elapsed
        return obs, info

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        if self.num_envs == 1:
            return self._step_one(action)
        return self._step_batch(action)

    def _step_one(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.shape != (self.action_dim,):
            raise ValueError(f"expected action shape ({self.action_dim},), got {action_array.shape}")
        obs, env_reward, terminated, truncated, info = self.env.step(action_array)
        self._elapsed = int(self._elapsed) + 1
        success = sparse_success_reward(info) >= 0.5
        if self.dense_reward_debug:
            reward = float(np.asarray(env_reward).reshape(-1)[0])
        else:
            reward = 1.0 if success else 0.0
        terminated_flag = bool(np.asarray(terminated).reshape(-1)[0]) or success
        truncated_flag = bool(np.asarray(truncated).reshape(-1)[0])
        if self._elapsed >= self.max_episode_steps and not success:
            truncated_flag = True
        if success:
            # Success zeros the TD bootstrap; do not treat timeout as the outcome.
            truncated_flag = False
            terminated_flag = True
        return obs, reward, terminated_flag, truncated_flag, info

    def _step_batch(self, action: Any) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        from smolvla_rltoken.vla.evaluation import vector_bool, vector_float

        action_array = np.asarray(action, dtype=np.float32).reshape(self.num_envs, self.action_dim)
        obs, env_reward, terminated, truncated, info = self.env.step(action_array)
        elapsed = np.asarray(self._elapsed, dtype=np.int32).reshape(self.num_envs) + 1
        self._elapsed = elapsed
        success = vector_bool(
            info.get("success", np.zeros(self.num_envs, dtype=bool)),
            size=self.num_envs,
            name="info.success",
        )
        if self.dense_reward_debug:
            reward = vector_float(env_reward, size=self.num_envs, name="reward")
        else:
            reward = success.astype(np.float64)
        terminated_flag = vector_bool(terminated, size=self.num_envs, name="terminated") | success
        truncated_flag = vector_bool(truncated, size=self.num_envs, name="truncated")
        truncated_flag = truncated_flag | ((elapsed >= self.max_episode_steps) & ~success)
        truncated_flag = truncated_flag & ~success
        terminated_flag = terminated_flag | success
        return obs, reward, terminated_flag, truncated_flag, info


class MockChunkEnv:
    """CPU stand-in: optional success step, dense env.reward is a trap (99)."""

    def __init__(
        self,
        *,
        success_at: int | None = None,
        max_episode_steps: int = 20,
        action_dim: int = EXPECTED_ACTION_DIM,
        num_envs: int = 1,
    ):
        if num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        self.success_at = success_at
        self.max_episode_steps = max_episode_steps
        self.action_dim = action_dim
        self.num_envs = int(num_envs)
        self.t: int | np.ndarray = 0
        self.last_action: np.ndarray | None = None
        self.reconfigure_count = 0

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Mirrors ManiSkill: ``options['env_idx']`` resets only those envs."""
        env_idx = reset_env_indices(kwargs.get("options"), self.num_envs)
        self.reconfigure_count += bool((kwargs.get("options") or {}).get("reconfigure", False))
        if self.num_envs == 1:
            self.t = 0
            return {"t": 0}, {"success": False}
        if env_idx is None:
            self.t = np.zeros(self.num_envs, dtype=np.int32)
        else:
            ticks = np.asarray(self.t, dtype=np.int32).reshape(self.num_envs).copy()
            ticks[env_idx] = 0
            self.t = ticks
        return {"t": self.t.copy()}, {"success": np.zeros(self.num_envs, dtype=bool)}

    def step(self, action: Any) -> tuple[dict[str, Any], Any, Any, Any, dict[str, Any]]:
        if self.num_envs == 1:
            action_array = np.asarray(action, dtype=np.float32).reshape(-1)
            if action_array.shape != (self.action_dim,):
                raise ValueError(f"expected action shape ({self.action_dim},), got {action_array.shape}")
            self.last_action = action_array
            self.t = int(self.t) + 1
            success = self.success_at is not None and self.t >= self.success_at
            terminated = success
            truncated = self.t >= self.max_episode_steps and not success
            reward = 1.0 if success else 0.0
            return {"t": self.t}, reward, terminated, truncated, {"success": success}

        action_array = np.asarray(action, dtype=np.float32).reshape(self.num_envs, self.action_dim)
        self.last_action = action_array
        ticks = np.asarray(self.t, dtype=np.int32).reshape(self.num_envs) + 1
        self.t = ticks
        if self.success_at is None:
            success = np.zeros(self.num_envs, dtype=bool)
        else:
            success = ticks >= int(self.success_at)
        terminated = success.copy()
        truncated = (ticks >= self.max_episode_steps) & ~success
        reward = success.astype(np.float32)
        return {"t": ticks.copy()}, reward, terminated, truncated, {"success": success}


def make_chunk_env(
    camera_specs: Mapping[str, Any],
    *,
    sim_backend: str = "physx_cuda",
    max_episode_steps: int = 200,
    dense_reward_debug: bool = False,
    action_dim: int = EXPECTED_ACTION_DIM,
    num_envs: int = 1,
) -> SparseSuccessChunkEnv:
    """Real PegInsertion wrapper. Imports ManiSkill only when called."""
    from smolvla_rltoken.vla.evaluation import make_maniskill_env

    env = make_maniskill_env(
        camera_specs,
        sim_backend=sim_backend,
        max_episode_steps=max_episode_steps,
        num_envs=num_envs,
    )
    return SparseSuccessChunkEnv(
        env,
        action_dim=action_dim,
        max_episode_steps=max_episode_steps,
        dense_reward_debug=dense_reward_debug,
        num_envs=num_envs,
    )
