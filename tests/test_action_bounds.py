"""CPU tests for per-dim action bounds in the VLA normalized space (no VLA load)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from smolvla_rltoken.paths import SFT_LAST_PRETRAINED
from smolvla_rltoken.vla.action_bounds import (
    POSTPROCESSOR_FILENAME,
    describe_action_bounds,
    normalized_action_bounds,
    read_action_norm_stats,
)

ACTION_DIM = 8


def _write_checkpoint(
    root: Path,
    *,
    mode: str,
    stats: dict[str, list[float]],
    state_file: str = "unnorm.safetensors",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / POSTPROCESSOR_FILENAME).write_text(
        json.dumps(
            {
                "name": "policy_postprocessor",
                "steps": [
                    {
                        "registry_name": "unnormalizer_processor",
                        "config": {"eps": 1e-8, "norm_map": {"ACTION": mode}},
                        "state_file": state_file,
                    },
                    {"registry_name": "device_processor", "config": {"device": "cpu"}},
                ],
            }
        )
    )
    save_file(
        {f"action.{name}": torch.tensor(value) for name, value in stats.items()},
        root / state_file,
    )
    return root


@pytest.mark.skipif(not SFT_LAST_PRETRAINED.is_dir(), reason="SFT checkpoint not available")
def test_real_checkpoint_is_mean_std_and_not_unit_bounded():
    stats = read_action_norm_stats(SFT_LAST_PRETRAINED)
    assert stats.mode == "MEAN_STD"
    lo, hi = normalized_action_bounds(SFT_LAST_PRETRAINED, margin=1.0, action_dim=ACTION_DIM)
    assert lo.shape == (ACTION_DIM,)
    assert hi.shape == (ACTION_DIM,)
    # A +/-1 clip would cut most of the joint travel; the arm joints reach past 2.
    assert float(hi[:7].min()) > 2.0
    assert float(lo[:7].max()) < -2.0
    assert bool((hi > lo).all())


@pytest.mark.skipif(not SFT_LAST_PRETRAINED.is_dir(), reason="SFT checkpoint not available")
def test_real_checkpoint_bounds_unnormalize_to_demo_range():
    stats = read_action_norm_stats(SFT_LAST_PRETRAINED)
    lo, hi = normalized_action_bounds(SFT_LAST_PRETRAINED, margin=1.0, action_dim=ACTION_DIM)
    mean, std = stats.stats["mean"], stats.stats["std"]
    torch.testing.assert_close(lo * std + mean, stats.stats["min"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(hi * std + mean, stats.stats["max"], rtol=1e-4, atol=1e-5)


def test_min_max_checkpoint_bounds_are_unit(tmp_path: Path):
    root = _write_checkpoint(
        tmp_path / "ckpt",
        mode="MIN_MAX",
        stats={"min": [-3.0, 0.5], "max": [1.0, 4.5]},
    )
    lo, hi = normalized_action_bounds(root, margin=1.0)
    torch.testing.assert_close(lo, torch.tensor([-1.0, -1.0]))
    torch.testing.assert_close(hi, torch.tensor([1.0, 1.0]))


def test_quantile_checkpoint_normalizes_extremes(tmp_path: Path):
    root = _write_checkpoint(
        tmp_path / "ckpt",
        mode="QUANTILES",
        stats={"min": [-2.0], "max": [2.0], "q01": [-1.0], "q99": [1.0]},
    )
    lo, hi = normalized_action_bounds(root, margin=1.0)
    torch.testing.assert_close(lo, torch.tensor([-2.0]))
    torch.testing.assert_close(hi, torch.tensor([2.0]))


def test_margin_expands_around_the_center(tmp_path: Path):
    root = _write_checkpoint(
        tmp_path / "ckpt",
        mode="MEAN_STD",
        stats={"min": [-1.0], "max": [3.0], "mean": [0.0], "std": [1.0]},
    )
    lo1, hi1 = normalized_action_bounds(root, margin=1.0)
    lo2, hi2 = normalized_action_bounds(root, margin=2.0)
    torch.testing.assert_close(lo1, torch.tensor([-1.0]))
    torch.testing.assert_close(hi1, torch.tensor([3.0]))
    # center 1.0, half-range 2.0 -> margin 2 gives [-3, 5]
    torch.testing.assert_close(lo2, torch.tensor([-3.0]))
    torch.testing.assert_close(hi2, torch.tensor([5.0]))


def test_margin_must_be_positive(tmp_path: Path):
    root = _write_checkpoint(
        tmp_path / "ckpt",
        mode="MEAN_STD",
        stats={"min": [-1.0], "max": [1.0], "mean": [0.0], "std": [1.0]},
    )
    with pytest.raises(ValueError, match="margin must be positive"):
        normalized_action_bounds(root, margin=0.0)


def test_missing_stats_and_files_report_clearly(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="postprocessor metadata"):
        read_action_norm_stats(tmp_path / "absent")
    root = _write_checkpoint(
        tmp_path / "ckpt",
        mode="MEAN_STD",
        stats={"min": [-1.0], "max": [1.0]},
    )
    with pytest.raises(ValueError, match="action.mean"):
        read_action_norm_stats(root)


def test_action_dim_slice_and_overflow(tmp_path: Path):
    root = _write_checkpoint(
        tmp_path / "ckpt",
        mode="MEAN_STD",
        stats={
            "min": [-1.0, -2.0, -3.0],
            "max": [1.0, 2.0, 3.0],
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        },
    )
    lo, hi = normalized_action_bounds(root, margin=1.0, action_dim=2)
    assert lo.shape == (2,)
    assert hi.shape == (2,)
    with pytest.raises(ValueError, match="need 5"):
        normalized_action_bounds(root, margin=1.0, action_dim=5)


@pytest.mark.skipif(not SFT_LAST_PRETRAINED.is_dir(), reason="SFT checkpoint not available")
def test_describe_is_json_serializable():
    payload = describe_action_bounds(SFT_LAST_PRETRAINED, margin=1.5, action_dim=ACTION_DIM)
    assert payload["mode"] == "MEAN_STD"
    assert payload["margin"] == 1.5
    assert len(payload["lo"]) == ACTION_DIM
    assert len(payload["hi"]) == ACTION_DIM
    json.dumps(payload)
