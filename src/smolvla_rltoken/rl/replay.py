"""Replay of executed chunks. V1: no stride; optional success/reward upsampling."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor

from smolvla_rltoken.rollout.transition import ChunkTransition, cat_state


class ChunkReplayBuffer:
    """Ring buffer. ``add`` stores exactly one transition; stride is not implemented.

    ``sample`` is uniform by default. Passing ``success_frac`` / ``reward_frac``
    upsamples slots from episodes that later terminated and slots with a
    positive reward. There is no PER.
    """

    def __init__(
        self,
        capacity: int,
        *,
        rl_token_dim: int,
        proprio_dim: int,
        chunk_len: int,
        action_dim: int,
        device: str | torch.device = "cpu",
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.rl_token_dim = rl_token_dim
        self.proprio_dim = proprio_dim
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.device = torch.device(device)
        # Documented unused: V1 does not subsample action chunks.
        self.stride = 1

        self.z_rl = torch.zeros(capacity, rl_token_dim)
        self.proprio = torch.zeros(capacity, proprio_dim)
        self.reference_action = torch.zeros(capacity, chunk_len, action_dim)
        self.executed_action = torch.zeros(capacity, chunk_len, action_dim)
        self.reward_sequence = torch.zeros(capacity, chunk_len)
        self.n_steps = torch.zeros(capacity, dtype=torch.long)
        self.next_z_rl = torch.zeros(capacity, rl_token_dim)
        self.next_proprio = torch.zeros(capacity, proprio_dim)
        self.next_reference_action = torch.zeros(capacity, chunk_len, action_dim)
        self.terminated = torch.zeros(capacity)
        self.truncated = torch.zeros(capacity)
        self.episode_id = torch.zeros(capacity, dtype=torch.long)
        self.chunk_id = torch.zeros(capacity, dtype=torch.long)

        self.size = 0
        self._ptr = 0
        self._episode_slots: dict[int, set[int]] = defaultdict(set)
        self._success_episodes: set[int] = set()
        self._success_slots: set[int] = set()
        self._reward_slots: set[int] = set()

    @property
    def n_success_slots(self) -> int:
        return len(self._success_slots)

    @property
    def n_reward_slots(self) -> int:
        return len(self._reward_slots)

    def add(self, transition: ChunkTransition) -> None:
        if getattr(transition, "offset", None) is not None:
            raise ValueError("stride/offset transitions are not enabled in V1")
        n = int(transition.n_steps)
        if n < 1 or n > self.chunk_len:
            raise ValueError(f"n_steps must be in [1, {self.chunk_len}], got {n}")
        p = self._ptr
        if self.size == self.capacity:
            self._forget_slot(p)
        self.z_rl[p] = transition.z_rl.detach().cpu().reshape(self.rl_token_dim)
        self.proprio[p] = transition.proprio.detach().cpu().reshape(self.proprio_dim)
        self.reference_action[p] = transition.reference_action.detach().cpu()
        self.executed_action[p] = transition.executed_action.detach().cpu()
        self.reward_sequence[p] = transition.reward_sequence.detach().cpu()
        self.n_steps[p] = n
        self.next_z_rl[p] = transition.next_z_rl.detach().cpu().reshape(self.rl_token_dim)
        self.next_proprio[p] = transition.next_proprio.detach().cpu().reshape(self.proprio_dim)
        self.next_reference_action[p] = transition.next_reference_action.detach().cpu()
        self.terminated[p] = float(transition.terminated)
        self.truncated[p] = float(transition.truncated)
        self.episode_id[p] = int(transition.episode_id)
        self.chunk_id[p] = int(transition.chunk_id)
        self._register_slot(p, transition)
        self._ptr = (p + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _forget_slot(self, slot: int) -> None:
        episode = int(self.episode_id[slot].item())
        slots = self._episode_slots.get(episode)
        if slots is not None:
            slots.discard(slot)
            if not slots:
                del self._episode_slots[episode]
                self._success_episodes.discard(episode)
        self._success_slots.discard(slot)
        self._reward_slots.discard(slot)

    def _register_slot(self, slot: int, transition: ChunkTransition) -> None:
        episode = int(transition.episode_id)
        self._episode_slots[episode].add(slot)
        if float(transition.reward_sequence.sum()) > 0:
            self._reward_slots.add(slot)
        if float(transition.terminated) >= 0.5:
            self._success_episodes.add(episode)
            self._success_slots.update(self._episode_slots[episode])
        elif episode in self._success_episodes:
            self._success_slots.add(slot)

    def sample(
        self,
        batch_size: int,
        *,
        success_frac: float = 0.0,
        reward_frac: float = 0.0,
    ) -> dict[str, Tensor]:
        if self.size < 1:
            raise RuntimeError("cannot sample from an empty replay buffer")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if success_frac < 0 or reward_frac < 0:
            raise ValueError("success_frac and reward_frac must be non-negative")
        if not self._success_slots or (success_frac <= 0 and reward_frac <= 0):
            idx = torch.randint(0, self.size, (batch_size,))
            return self._gather(idx)

        n_reward = min(batch_size, max(0, int(round(batch_size * float(reward_frac)))))
        n_success = min(
            batch_size - n_reward, max(0, int(round(batch_size * float(success_frac))))
        )
        reward_idx = self._sample_from(self._reward_slots, n_reward)
        if reward_idx.numel() < n_reward:
            n_success = min(
                batch_size - int(reward_idx.numel()),
                n_success + (n_reward - int(reward_idx.numel())),
            )
        picked = set(int(i) for i in reward_idx.tolist())
        success_remain = self._success_slots - picked
        success_idx = self._sample_from(success_remain, n_success)
        picked.update(int(i) for i in success_idx.tolist())
        n_rest = batch_size - int(reward_idx.numel()) - int(success_idx.numel())
        rest_idx = self._sample_rest(n_rest, picked)
        idx = torch.cat([reward_idx, success_idx, rest_idx], dim=0)
        if idx.numel() != batch_size:
            raise RuntimeError(
                f"stratified sample built {idx.numel()} indices, expected {batch_size}"
            )
        return self._gather(idx[torch.randperm(idx.numel())])

    def _sample_from(self, pool: set[int], count: int) -> Tensor:
        if count <= 0 or not pool:
            return torch.empty(0, dtype=torch.long)
        values = torch.tensor(sorted(pool), dtype=torch.long)
        n = values.numel()
        if count <= n:
            return values[torch.randperm(n)[:count]]
        extra = torch.randint(0, n, (count - n,))
        return torch.cat([values, values[extra]], dim=0)

    def _sample_rest(self, count: int, exclude: set[int]) -> Tensor:
        if count <= 0:
            return torch.empty(0, dtype=torch.long)
        occupied = torch.arange(self.size, dtype=torch.long)
        success_mask = torch.zeros(self.size, dtype=torch.bool)
        if self._success_slots:
            success_idx = torch.tensor(sorted(self._success_slots), dtype=torch.long)
            success_mask[success_idx] = True
        rest = occupied[~success_mask]
        if rest.numel() == 0:
            keep = torch.ones(self.size, dtype=torch.bool)
            for slot in exclude:
                if 0 <= slot < self.size:
                    keep[slot] = False
            remaining = occupied[keep]
            rest = remaining if remaining.numel() else occupied
        return self._sample_from(set(int(i) for i in rest.tolist()), count)

    def _gather(self, idx: Tensor) -> dict[str, Tensor]:
        dev = self.device
        z_rl = self.z_rl[idx].to(dev)
        proprio = self.proprio[idx].to(dev)
        next_z_rl = self.next_z_rl[idx].to(dev)
        next_proprio = self.next_proprio[idx].to(dev)
        return {
            "z_rl": z_rl,
            "proprio": proprio,
            "x": cat_state(z_rl, proprio),
            "reference_action": self.reference_action[idx].to(dev),
            "executed_action": self.executed_action[idx].to(dev),
            "reward_sequence": self.reward_sequence[idx].to(dev),
            "n_steps": self.n_steps[idx].to(dev),
            "next_z_rl": next_z_rl,
            "next_proprio": next_proprio,
            "x_next": cat_state(next_z_rl, next_proprio),
            "next_reference_action": self.next_reference_action[idx].to(dev),
            "terminated": self.terminated[idx].to(dev),
            "truncated": self.truncated[idx].to(dev),
            "episode_id": self.episode_id[idx].to(dev),
            "chunk_id": self.chunk_id[idx].to(dev),
        }

    def __len__(self) -> int:
        return self.size
