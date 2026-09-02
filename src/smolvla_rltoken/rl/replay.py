"""Uniform replay of executed chunks. V1: no stride, no offset windows."""

from __future__ import annotations

import torch
from torch import Tensor

from smolvla_rltoken.rollout.transition import ChunkTransition, cat_state


class ChunkReplayBuffer:
    """Ring buffer. ``add`` stores exactly one transition; stride is not implemented."""

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

    def add(self, transition: ChunkTransition) -> None:
        if getattr(transition, "offset", None) is not None:
            raise ValueError("stride/offset transitions are not enabled in V1")
        n = int(transition.n_steps)
        if n < 1 or n > self.chunk_len:
            raise ValueError(f"n_steps must be in [1, {self.chunk_len}], got {n}")
        p = self._ptr
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
        self._ptr = (p + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, Tensor]:
        if self.size < 1:
            raise RuntimeError("cannot sample from an empty replay buffer")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        idx = torch.randint(0, self.size, (batch_size,))
        return self._gather(idx)

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
