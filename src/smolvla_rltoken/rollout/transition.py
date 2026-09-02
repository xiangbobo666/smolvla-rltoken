"""One executed action chunk as a replay transition. V1 has no stride/offset."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor


@dataclass
class ChunkTransition:
    """Late-write chunk transition. Unused executed/reward slots are zeros."""

    z_rl: Tensor
    proprio: Tensor
    reference_action: Tensor
    executed_action: Tensor
    reward_sequence: Tensor
    n_steps: int
    next_z_rl: Tensor
    next_proprio: Tensor
    next_reference_action: Tensor
    terminated: float
    truncated: float
    episode_id: int
    chunk_id: int

    def __post_init__(self) -> None:
        if hasattr(self, "offset") or hasattr(self, "stride"):
            raise ValueError("V1 transitions have no stride/offset fields")


TRANSITION_FIELD_NAMES = tuple(f.name for f in fields(ChunkTransition))


def cat_state(z_rl: Tensor, proprio: Tensor) -> Tensor:
    """x = (z_rl, s^p). Accepts a single row or a batch."""
    if z_rl.ndim == 1:
        return torch.cat([z_rl, proprio], dim=-1)
    return torch.cat([z_rl, proprio], dim=-1)
