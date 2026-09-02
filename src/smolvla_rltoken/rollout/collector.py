"""Execute C env steps and late-write one replay transition.

Pending non-terminal chunks wait for the next ``observe`` so ``next_*`` is the
real VLA encoding of obs', not a copy of the current reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from smolvla_rltoken.rl.replay import ChunkReplayBuffer
from smolvla_rltoken.rollout.planner import ObserveResult, Planner
from smolvla_rltoken.rollout.transition import ChunkTransition


@dataclass(frozen=True)
class EpisodeOutcome:
    """One finished episode. ``use_actor`` is true if any chunk came from the Actor.

    ``episode_id`` is unique across the whole run, including parallel envs, so
    ``episodes.jsonl`` and the replay's ``episode_id`` stay unambiguous.
    """

    episode_id: int
    steps: int
    success: bool
    terminated: bool
    truncated: bool
    use_actor: bool
    env_index: int = 0


@dataclass
class CollectResult:
    n_steps: int
    terminated: bool
    truncated: bool
    success: bool
    use_actor: bool
    executed_action: Tensor
    reference_action: Tensor
    # True when the caller must reset the collector before the next chunk.
    # ``BatchedChunkCollector`` restarts finished envs itself, so it reports False.
    episode_done: bool
    added: int
    finished: tuple[EpisodeOutcome, ...] = ()
    # True when every *executed* step equalled the VLA reference. Compared before
    # the unexecuted tail is zero-padded, so an early success still counts.
    executed_matches_reference: bool = True
    reconfigured: bool = False

    @property
    def finished_episodes(self) -> int:
        return len(self.finished)

    @property
    def finished_successes(self) -> int:
        return sum(1 for outcome in self.finished if outcome.success)


@dataclass
class _PendingChunk:
    features: ObserveResult
    executed_action: Tensor
    reward_sequence: Tensor
    n_steps: int
    episode_id: int
    chunk_id: int


@dataclass
class _EpisodeState:
    """Per-env accumulator for the episode currently in progress."""

    steps: int = 0
    used_actor: bool = False

    def reset(self) -> None:
        self.steps = 0
        self.used_actor = False


def _zeros_like_state(features: ObserveResult) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.zeros_like(features.z_rl),
        torch.zeros_like(features.proprio),
        torch.zeros_like(features.reference_action),
    )


class ChunkCollector:
    """Single-env collector. Does not call human intervention."""

    def __init__(
        self,
        env: Any,
        planner: Planner,
        replay: ChunkReplayBuffer,
        *,
        chunk_len: int,
        action_dim: int,
    ):
        self.env = env
        self.planner = planner
        self.replay = replay
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.obs: Any = None
        self.episode_id = 0
        self.chunk_id = 0
        self._pending: _PendingChunk | None = None
        self._episode = _EpisodeState()

    def reset(self, **kwargs: Any) -> None:
        self.obs, _info = self.env.reset(**kwargs)
        self.chunk_id = 0
        self._pending = None
        self._episode.reset()

    def _observe(self) -> ObserveResult:
        return self.planner.observe(self.obs)

    def _flush_pending(self, next_features: ObserveResult) -> None:
        pending = self._pending
        if pending is None:
            return
        self.replay.add(
            ChunkTransition(
                z_rl=pending.features.z_rl,
                proprio=pending.features.proprio,
                reference_action=pending.features.reference_action,
                executed_action=pending.executed_action,
                reward_sequence=pending.reward_sequence,
                n_steps=pending.n_steps,
                next_z_rl=next_features.z_rl,
                next_proprio=next_features.proprio,
                next_reference_action=next_features.reference_action,
                terminated=0.0,
                truncated=0.0,
                episode_id=pending.episode_id,
                chunk_id=pending.chunk_id,
            )
        )
        self._pending = None

    def flush_pending(self) -> int:
        """Write the trailing mid-episode chunk. One extra observe of the current obs."""
        if self._pending is None:
            return 0
        self._flush_pending(self._observe())
        return 1

    def run_chunk(self, *, use_actor: bool, deterministic: bool = False) -> CollectResult:
        features = self._observe()
        added = 0
        if self._pending is not None:
            self._flush_pending(features)
            added += 1

        executed = self.planner.act(features, use_actor=use_actor, deterministic=deterministic)
        rewards = torch.zeros(self.chunk_len)
        terminated = truncated = success = False
        n_steps = self.chunk_len
        for step_index in range(self.chunk_len):
            env_action = self.planner.to_env_action(executed[step_index])
            self.obs, reward, terminated, truncated, info = self.env.step(env_action)
            rewards[step_index] = float(reward)
            success = success or (float(reward) >= 0.5) or bool(info.get("success", False))
            if terminated or truncated:
                n_steps = step_index + 1
                break

        raw_exec = executed.detach().cpu()
        reference = features.reference_action.detach().cpu()
        matches = bool(torch.equal(raw_exec[:n_steps], reference[:n_steps]))
        pad_exec = raw_exec.clone()
        if n_steps < self.chunk_len:
            pad_exec[n_steps:] = 0
            rewards[n_steps:] = 0

        self._episode.steps += n_steps
        self._episode.used_actor = self._episode.used_actor or use_actor
        finished: tuple[EpisodeOutcome, ...] = ()

        if terminated:
            next_z, next_p, next_ref = _zeros_like_state(features)
            self.replay.add(
                ChunkTransition(
                    z_rl=features.z_rl,
                    proprio=features.proprio,
                    reference_action=features.reference_action,
                    executed_action=pad_exec,
                    reward_sequence=rewards,
                    n_steps=n_steps,
                    next_z_rl=next_z,
                    next_proprio=next_p,
                    next_reference_action=next_ref,
                    terminated=1.0,
                    truncated=0.0,
                    episode_id=self.episode_id,
                    chunk_id=self.chunk_id,
                )
            )
            added += 1
            finished = (self._close_episode(success, terminated=True, truncated=False),)
        elif truncated:
            next_features = self.planner.observe(self.obs)
            self.replay.add(
                ChunkTransition(
                    z_rl=features.z_rl,
                    proprio=features.proprio,
                    reference_action=features.reference_action,
                    executed_action=pad_exec,
                    reward_sequence=rewards,
                    n_steps=n_steps,
                    next_z_rl=next_features.z_rl,
                    next_proprio=next_features.proprio,
                    next_reference_action=next_features.reference_action,
                    terminated=0.0,
                    truncated=1.0,
                    episode_id=self.episode_id,
                    chunk_id=self.chunk_id,
                )
            )
            added += 1
            finished = (self._close_episode(success, terminated=False, truncated=True),)
        else:
            self._pending = _PendingChunk(
                features=features,
                executed_action=pad_exec,
                reward_sequence=rewards,
                n_steps=n_steps,
                episode_id=self.episode_id,
                chunk_id=self.chunk_id,
            )

        self.chunk_id += 1
        return CollectResult(
            n_steps=n_steps,
            terminated=bool(terminated),
            truncated=bool(truncated),
            success=bool(success),
            use_actor=use_actor,
            executed_action=pad_exec,
            reference_action=reference,
            episode_done=bool(terminated or truncated),
            added=added,
            finished=finished,
            executed_matches_reference=matches,
        )

    def _close_episode(self, success: bool, *, terminated: bool, truncated: bool) -> EpisodeOutcome:
        outcome = EpisodeOutcome(
            episode_id=self.episode_id,
            steps=self._episode.steps,
            success=bool(success),
            terminated=terminated,
            truncated=truncated,
            use_actor=self._episode.used_actor,
            env_index=0,
        )
        self.episode_id += 1
        self._episode.reset()
        return outcome


def _row(features: ObserveResult, index: int) -> ObserveResult:
    return ObserveResult(
        z_rl=features.z_rl[index],
        proprio=features.proprio[index],
        reference_action=features.reference_action[index],
    )


class BatchedChunkCollector:
    """Vectorized collector. Finished envs restart immediately via partial reset.

    A finished env is written to the replay, closed, and then restarted with
    ManiSkill's ``options={"env_idx": ...}`` partial reset at the end of the
    current chunk, so it idles for at most ``chunk_len - 1`` steps instead of
    waiting for the slowest member of the batch.

    Partial resets do **not** resample peg/box geometry: ManiSkill sets
    ``reconfiguration_freq=0`` for ``num_envs > 1``, so the geometry drawn at the
    first reconfigure would otherwise stay fixed for the entire run (only
    ``num_envs`` distinct pegs). ``reconfigure_every_episodes`` therefore forces a
    periodic full reset with ``reconfigure=True``. In-flight episodes are dropped
    at that point because a reconfigure invalidates their next state.
    """

    def __init__(
        self,
        env: Any,
        planner: Planner,
        replay: ChunkReplayBuffer,
        *,
        chunk_len: int,
        action_dim: int,
        num_envs: int,
        reconfigure_every_episodes: int = 0,
    ):
        if num_envs < 2:
            raise ValueError("BatchedChunkCollector requires num_envs >= 2")
        self.env = env
        self.planner = planner
        self.replay = replay
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.num_envs = int(num_envs)
        self.reconfigure_every_episodes = int(reconfigure_every_episodes)
        self.obs: Any = None
        self.chunk_id = np.zeros(self.num_envs, dtype=np.int64)
        self._pending: list[_PendingChunk | None] = [None] * self.num_envs
        self._episodes = [_EpisodeState() for _ in range(self.num_envs)]
        self._episode_ids = np.zeros(self.num_envs, dtype=np.int64)
        self._next_episode_id = 0
        self._episodes_since_reconfigure = 0
        self.reconfigure_count = 0

    @property
    def episode_id(self) -> np.ndarray:
        """Currently active episode id per env (globally unique across the run)."""
        return self._episode_ids

    def _assign_episode_ids(self, indices: np.ndarray | None) -> None:
        targets = range(self.num_envs) if indices is None else [int(i) for i in indices]
        for index in targets:
            self._episode_ids[index] = self._next_episode_id
            self._next_episode_id += 1

    def reset(self, **kwargs: Any) -> None:
        seed = kwargs.pop("seed", None)
        if seed is not None:
            seeds = [int(seed) + index for index in range(self.num_envs)]
            kwargs.setdefault("options", {"reconfigure": True})
            self.obs, _info = self.env.reset(seed=seeds, **kwargs)
        else:
            self.obs, _info = self.env.reset(**kwargs)
        if (kwargs.get("options") or {}).get("reconfigure", False):
            self.reconfigure_count += 1
        self.chunk_id[:] = 0
        self._pending = [None] * self.num_envs
        self._episodes_since_reconfigure = 0
        for state in self._episodes:
            state.reset()
        self._assign_episode_ids(None)

    def _restart_envs(self, indices: list[int]) -> bool:
        """Partially reset finished envs, or reconfigure the whole batch if due.

        Returns True when a full reconfigure happened.
        """
        if not indices:
            return False
        due = (
            self.reconfigure_every_episodes > 0
            and self._episodes_since_reconfigure >= self.reconfigure_every_episodes
        )
        if due:
            # A reconfigure rebuilds every scene, so no env keeps a valid next
            # state; drop in-flight chunks instead of bootstrapping across it.
            self.reset(options={"reconfigure": True})
            return True
        selected = np.asarray(sorted(set(indices)), dtype=np.int64)
        self.obs, _info = self.env.reset(options={"env_idx": selected})
        for index in selected:
            self.chunk_id[index] = 0
            self._pending[int(index)] = None
            self._episodes[int(index)].reset()
        self._assign_episode_ids(selected)
        return False

    def _flush_one(self, index: int, next_features: ObserveResult) -> None:
        pending = self._pending[index]
        if pending is None:
            return
        self.replay.add(
            ChunkTransition(
                z_rl=pending.features.z_rl,
                proprio=pending.features.proprio,
                reference_action=pending.features.reference_action,
                executed_action=pending.executed_action,
                reward_sequence=pending.reward_sequence,
                n_steps=pending.n_steps,
                next_z_rl=next_features.z_rl,
                next_proprio=next_features.proprio,
                next_reference_action=next_features.reference_action,
                terminated=0.0,
                truncated=0.0,
                episode_id=pending.episode_id,
                chunk_id=pending.chunk_id,
            )
        )
        self._pending[index] = None

    def flush_pending(self) -> int:
        """Write every trailing mid-episode chunk. One extra batched observe."""
        if all(pending is None for pending in self._pending):
            return 0
        features = self._as_batch(self.planner.observe_batch(self.obs))
        added = 0
        for index in range(self.num_envs):
            if self._pending[index] is None:
                continue
            self._flush_one(index, _row(features, index))
            added += 1
        return added

    def _as_batch(self, features: ObserveResult) -> ObserveResult:
        if features.z_rl.ndim == 1:
            features = ObserveResult(
                z_rl=features.z_rl.unsqueeze(0),
                proprio=features.proprio.unsqueeze(0),
                reference_action=features.reference_action.unsqueeze(0),
            )
        if features.z_rl.shape[0] != self.num_envs:
            raise ValueError(
                f"planner batch {features.z_rl.shape[0]} does not match num_envs={self.num_envs}"
            )
        return features

    def run_chunk(self, *, use_actor: bool, deterministic: bool = False) -> CollectResult:
        from smolvla_rltoken.vla.evaluation import vector_bool, vector_float

        features = self._as_batch(self.planner.observe_batch(self.obs))
        added = 0
        for index in range(self.num_envs):
            if self._pending[index] is not None:
                self._flush_one(index, _row(features, index))
                added += 1

        executed = self.planner.act_batch(features, use_actor=use_actor, deterministic=deterministic)
        if executed.ndim == 2:
            executed = executed.unsqueeze(0)
        rewards = torch.zeros(self.num_envs, self.chunk_len)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        success = np.zeros(self.num_envs, dtype=bool)
        n_steps = np.full(self.num_envs, self.chunk_len, dtype=np.int32)
        # An env that finishes mid-chunk stops contributing rewards and steps.
        # It is restarted by a partial reset once the chunk ends, so it idles for
        # at most chunk_len - 1 steps rather than for the rest of the batch wave.
        active = np.ones(self.num_envs, dtype=bool)

        for step_index in range(self.chunk_len):
            if not active.any():
                break
            env_action = self.planner.to_env_actions(executed[:, step_index])
            self.obs, reward, term, trunc, info = self.env.step(env_action)
            reward_v = vector_float(reward, size=self.num_envs, name="reward")
            term_v = vector_bool(term, size=self.num_envs, name="terminated")
            trunc_v = vector_bool(trunc, size=self.num_envs, name="truncated")
            succ_v = vector_bool(
                info.get("success", np.zeros(self.num_envs, dtype=bool)),
                size=self.num_envs,
                name="info.success",
            )
            for index in range(self.num_envs):
                if not active[index]:
                    continue
                rewards[index, step_index] = float(reward_v[index])
                success[index] = success[index] or (float(reward_v[index]) >= 0.5) or bool(succ_v[index])
                if term_v[index] or trunc_v[index]:
                    terminated[index] = bool(term_v[index])
                    truncated[index] = bool(trunc_v[index]) and not bool(term_v[index])
                    n_steps[index] = step_index + 1
                    active[index] = False

        raw_exec = executed.detach().cpu()
        reference = features.reference_action.detach().cpu()
        pad_exec = raw_exec.clone()
        next_features = None
        if truncated.any() and not terminated.all():
            next_features = self._as_batch(self.planner.observe_batch(self.obs))
        finished: list[EpisodeOutcome] = []
        matches = True
        for index in range(self.num_envs):
            executed_len = int(n_steps[index])
            if not torch.equal(
                raw_exec[index, :executed_len], reference[index, :executed_len]
            ):
                matches = False
            if n_steps[index] < self.chunk_len:
                pad_exec[index, n_steps[index] :] = 0
                rewards[index, n_steps[index] :] = 0
            self._episodes[index].steps += int(n_steps[index])
            self._episodes[index].used_actor = self._episodes[index].used_actor or use_actor
            if terminated[index]:
                next_z, next_p, next_ref = _zeros_like_state(_row(features, index))
                self.replay.add(
                    ChunkTransition(
                        z_rl=features.z_rl[index],
                        proprio=features.proprio[index],
                        reference_action=features.reference_action[index],
                        executed_action=pad_exec[index],
                        reward_sequence=rewards[index],
                        n_steps=int(n_steps[index]),
                        next_z_rl=next_z,
                        next_proprio=next_p,
                        next_reference_action=next_ref,
                        terminated=1.0,
                        truncated=0.0,
                        episode_id=int(self.episode_id[index]),
                        chunk_id=int(self.chunk_id[index]),
                    )
                )
                added += 1
                self._pending[index] = None
                finished.append(
                    self._close_episode(index, bool(success[index]), terminated=True, truncated=False)
                )
            elif truncated[index]:
                if next_features is None:
                    next_features = self._as_batch(self.planner.observe_batch(self.obs))
                nxt = _row(next_features, index)
                self.replay.add(
                    ChunkTransition(
                        z_rl=features.z_rl[index],
                        proprio=features.proprio[index],
                        reference_action=features.reference_action[index],
                        executed_action=pad_exec[index],
                        reward_sequence=rewards[index],
                        n_steps=int(n_steps[index]),
                        next_z_rl=nxt.z_rl,
                        next_proprio=nxt.proprio,
                        next_reference_action=nxt.reference_action,
                        terminated=0.0,
                        truncated=1.0,
                        episode_id=int(self.episode_id[index]),
                        chunk_id=int(self.chunk_id[index]),
                    )
                )
                added += 1
                self._pending[index] = None
                finished.append(
                    self._close_episode(index, bool(success[index]), terminated=False, truncated=True)
                )
            else:
                self._pending[index] = _PendingChunk(
                    features=_row(features, index),
                    executed_action=pad_exec[index],
                    reward_sequence=rewards[index],
                    n_steps=int(n_steps[index]),
                    episode_id=int(self.episode_id[index]),
                    chunk_id=int(self.chunk_id[index]),
                )
            self.chunk_id[index] += 1

        self._episodes_since_reconfigure += len(finished)
        reconfigured = self._restart_envs(
            [index for index in range(self.num_envs) if terminated[index] or truncated[index]]
        )

        return CollectResult(
            n_steps=int(n_steps.sum()),
            terminated=bool(terminated.any()),
            truncated=bool(truncated.any()),
            success=bool(success.any()),
            use_actor=use_actor,
            executed_action=pad_exec,
            reference_action=reference,
            # Finished envs are restarted here, so the caller never has to.
            episode_done=False,
            added=added,
            finished=tuple(finished),
            executed_matches_reference=matches,
            reconfigured=reconfigured,
        )

    def _close_episode(
        self, index: int, success: bool, *, terminated: bool, truncated: bool
    ) -> EpisodeOutcome:
        state = self._episodes[index]
        return EpisodeOutcome(
            episode_id=int(self._episode_ids[index]),
            steps=state.steps,
            success=bool(success),
            terminated=terminated,
            truncated=truncated,
            use_actor=state.used_actor,
            env_index=index,
        )
