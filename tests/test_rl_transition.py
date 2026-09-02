"""CPU tests for chunk transitions (no stride)."""

from __future__ import annotations

import torch

from smolvla_rltoken.rollout.transition import TRANSITION_FIELD_NAMES, ChunkTransition, cat_state


def test_transition_fields_have_no_stride_offset():
    assert "offset" not in TRANSITION_FIELD_NAMES
    assert "stride" not in TRANSITION_FIELD_NAMES
    required = {
        "z_rl",
        "proprio",
        "reference_action",
        "executed_action",
        "reward_sequence",
        "n_steps",
        "next_z_rl",
        "next_proprio",
        "next_reference_action",
        "terminated",
        "truncated",
        "episode_id",
        "chunk_id",
    }
    assert required.issubset(set(TRANSITION_FIELD_NAMES))


def test_cat_state_concatenates_token_and_proprio():
    z = torch.arange(4.0)
    p = torch.arange(3.0) + 10
    x = cat_state(z, p)
    assert x.tolist() == [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0]
