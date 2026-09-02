"""CPU tests for the frozen-VLA planner's action handling (no real VLA/sim)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from smolvla_rltoken.rollout.planner import FrozenVLAPlanner, ObserveResult

ACTION_DIM = 4
CHUNK_LEN = 3


class _FarAwayActor(nn.Module):
    """Ignores its inputs and returns a chunk far outside any sane bound."""

    def __init__(self, value: float = 50.0):
        super().__init__()
        self.value = value
        self.dummy = nn.Parameter(torch.zeros(1))

    def sample(self, x, ref_chunk, deterministic: bool = False, std=None, apply_dropout=False):
        del x, deterministic, std, apply_dropout
        return torch.full_like(ref_chunk, self.value)


def _bounds() -> tuple[torch.Tensor, torch.Tensor]:
    lo = torch.tensor([-2.0, -3.0, -1.0, -4.0])
    hi = torch.tensor([2.0, 4.0, 1.5, 4.0])
    return lo, hi


def _planner(actor=None, bounds=None) -> FrozenVLAPlanner:
    return FrozenVLAPlanner(
        extractor=None,
        encoder=None,
        actor=actor,
        chunk_len=CHUNK_LEN,
        action_dim=ACTION_DIM,
        action_bounds=bounds,
    )


def _features(scale: float = 1.0) -> ObserveResult:
    reference = torch.full((1, CHUNK_LEN, ACTION_DIM), scale)
    return ObserveResult(
        z_rl=torch.zeros(1, 5),
        proprio=torch.zeros(1, 2),
        reference_action=reference,
    )


def test_warmup_reference_passes_through_unclipped():
    planner = _planner(bounds=_bounds())
    # A reference well outside the bound must still be executed verbatim so that
    # warmup matches the SFT evaluation chain.
    features = _features(scale=9.0)
    chunk = planner.act_batch(features, use_actor=False)
    torch.testing.assert_close(chunk, features.reference_action)
    assert planner.reference_out_of_bounds == 1


def test_actor_output_is_clipped_per_dimension():
    lo, hi = _bounds()
    planner = _planner(actor=_FarAwayActor(), bounds=(lo, hi))
    chunk = planner.act_batch(_features(), use_actor=True)
    assert chunk.shape == (1, CHUNK_LEN, ACTION_DIM)
    torch.testing.assert_close(chunk[0, 0], hi)
    planner_low = _planner(actor=_FarAwayActor(-50.0), bounds=(lo, hi))
    chunk_low = planner_low.act_batch(_features(), use_actor=True)
    torch.testing.assert_close(chunk_low[0, 0], lo)


def test_actor_output_within_bounds_is_untouched():
    planner = _planner(actor=_FarAwayActor(0.5), bounds=_bounds())
    chunk = planner.act_batch(_features(), use_actor=True)
    torch.testing.assert_close(chunk, torch.full_like(chunk, 0.5))


def test_no_bounds_means_no_clip():
    planner = _planner(actor=_FarAwayActor(), bounds=None)
    chunk = planner.act_batch(_features(), use_actor=True)
    torch.testing.assert_close(chunk, torch.full_like(chunk, 50.0))
    assert planner.reference_out_of_bounds == 0


def test_to_env_actions_does_not_clip():
    planner = _planner(bounds=_bounds())
    steps = torch.tensor([[9.0, -9.0, 9.0, -9.0]])
    array = planner.to_env_actions(steps)
    assert isinstance(array, np.ndarray)
    np.testing.assert_allclose(array, steps.numpy())


def test_bounds_are_validated():
    lo, hi = _bounds()
    with pytest.raises(ValueError, match="must have 4 dims"):
        _planner(bounds=(torch.zeros(8), torch.ones(8)))
    with pytest.raises(ValueError, match="hi > lo"):
        _planner(bounds=(hi, lo))
