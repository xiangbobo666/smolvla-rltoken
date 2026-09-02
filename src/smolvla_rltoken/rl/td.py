"""Chunk-level TD helpers. V1 uses gamma^n_steps, not a fixed gamma^C."""

from __future__ import annotations

import torch
from torch import Tensor


def discounted_return(reward_sequence: Tensor, n_steps: Tensor, gamma: float) -> Tensor:
    """R = sum_{i=0}^{n-1} gamma^i r_i. ``reward_sequence`` is ``[B, C]``, ``n_steps`` is ``[B]``."""
    chunk_len = reward_sequence.shape[-1]
    idx = torch.arange(chunk_len, device=reward_sequence.device)
    valid = idx.unsqueeze(0) < n_steps.reshape(-1, 1)
    discounts = gamma ** idx.to(dtype=reward_sequence.dtype)
    return (reward_sequence * discounts.unsqueeze(0) * valid.to(reward_sequence.dtype)).sum(dim=-1)


def bootstrap_coeff(n_steps: Tensor, terminated: Tensor, gamma: float) -> Tensor:
    """(1 - terminated) * gamma^n. Truncation must pass terminated=0."""
    n = n_steps.to(dtype=torch.float32)
    done = terminated.to(dtype=torch.float32)
    return (1.0 - done) * (gamma ** n)
