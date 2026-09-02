"""Per-dimension action bounds in the SmolVLA normalized space.

Stage 2 clips the Actor chunk before the saved postprocessor unnormalizes it.
That bound must not be hard-coded to +/-1: this checkpoint normalizes ACTION
with ``MEAN_STD``, so the normalized space is a z-score whose demonstrated span
reaches about +/-5, and a +/-1 clip would cut roughly 70% of the joint travel.
The bounds are therefore derived from the same statistics the postprocessor
uses, which also makes a future MIN_MAX/quantile checkpoint come out at +/-1
automatically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

ACTION_KEY = "action"
UNNORMALIZER_REGISTRY_NAME = "unnormalizer_processor"
POSTPROCESSOR_FILENAME = "policy_postprocessor.json"
DEFAULT_EPS = 1e-8
# How far past the demonstrated range the Actor may go on each side.
DEFAULT_BOUND_MARGIN = 1.5

# Modes that map [low, high] onto [-1, 1]; MEAN_STD and IDENTITY are special-cased.
_RANGE_STATS = {
    "MIN_MAX": ("min", "max"),
    "QUANTILES": ("q01", "q99"),
    "QUANTILE10": ("q10", "q90"),
}


@dataclass(frozen=True)
class ActionNormStats:
    """ACTION normalization mode plus the tensors needed to reproduce it."""

    mode: str
    eps: float
    stats: dict[str, Tensor]
    state_file: str

    @property
    def action_dim(self) -> int:
        return int(self.stats["min"].numel())


def _required_stat_names(mode: str) -> tuple[str, ...]:
    # min/max define the demonstrated range; the mode stats normalize it.
    if mode in {"MEAN_STD", "IDENTITY"}:
        extra = ("mean", "std") if mode == "MEAN_STD" else ()
    elif mode in _RANGE_STATS:
        extra = _RANGE_STATS[mode]
    else:
        raise ValueError(f"unsupported ACTION normalization mode {mode!r}")
    return tuple(dict.fromkeys(("min", "max", *extra)))


def read_action_norm_stats(checkpoint: str | Path) -> ActionNormStats:
    """Read the ACTION norm mode and stats from a saved LeRobot postprocessor."""
    root = Path(checkpoint)
    meta_path = root / POSTPROCESSOR_FILENAME
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing postprocessor metadata: {meta_path}")
    meta = json.loads(meta_path.read_text())
    steps = [
        step
        for step in meta.get("steps", [])
        if step.get("registry_name") == UNNORMALIZER_REGISTRY_NAME
    ]
    if not steps:
        raise ValueError(f"{meta_path} has no {UNNORMALIZER_REGISTRY_NAME} step")
    step = steps[0]
    config = step.get("config") or {}
    mode = str((config.get("norm_map") or {}).get("ACTION", "IDENTITY"))
    eps = float(config.get("eps", DEFAULT_EPS))

    state_file = step.get("state_file")
    if not state_file:
        raise ValueError(f"{meta_path} {UNNORMALIZER_REGISTRY_NAME} step has no state_file")
    state_path = root / state_file
    if not state_path.is_file():
        raise FileNotFoundError(f"missing normalization stats: {state_path}")

    from safetensors.torch import load_file

    tensors = load_file(state_path)
    stats: dict[str, Tensor] = {}
    for name in _required_stat_names(mode):
        key = f"{ACTION_KEY}.{name}"
        if key not in tensors:
            raise ValueError(
                f"{state_path} is missing {key}, required for ACTION mode {mode}"
            )
        stats[name] = tensors[key].to(torch.float32).reshape(-1)
    return ActionNormStats(mode=mode, eps=eps, stats=stats, state_file=state_file)


def normalize_action(values: Tensor, stats: ActionNormStats) -> Tensor:
    """Forward normalization, mirroring LeRobot's ``_NormalizationMixin._normalize``."""
    if stats.mode == "IDENTITY":
        return values
    if stats.mode == "MEAN_STD":
        return (values - stats.stats["mean"]) / (stats.stats["std"] + stats.eps)
    low_name, high_name = _RANGE_STATS[stats.mode]
    low, high = stats.stats[low_name], stats.stats[high_name]
    denom = high - low
    denom = torch.where(denom == 0, torch.full_like(denom, stats.eps), denom)
    return 2.0 * (values - low) / denom - 1.0


def normalized_action_bounds(
    checkpoint: str | Path,
    *,
    margin: float = DEFAULT_BOUND_MARGIN,
    action_dim: int | None = None,
    stats: ActionNormStats | None = None,
) -> tuple[Tensor, Tensor]:
    """Return ``(lo, hi)`` of shape ``[action_dim]`` in the VLA normalized space."""
    if margin <= 0:
        raise ValueError(f"action bound margin must be positive, got {margin}")
    stats = stats or read_action_norm_stats(checkpoint)
    lo = normalize_action(stats.stats["min"], stats)
    hi = normalize_action(stats.stats["max"], stats)
    lo, hi = torch.minimum(lo, hi), torch.maximum(lo, hi)
    center, half = (lo + hi) / 2.0, (hi - lo) / 2.0
    lo, hi = center - margin * half, center + margin * half
    if action_dim is not None:
        if lo.numel() < action_dim:
            raise ValueError(
                f"checkpoint ACTION stats have {lo.numel()} dims, need {action_dim}"
            )
        lo, hi = lo[:action_dim], hi[:action_dim]
    return lo.contiguous(), hi.contiguous()


def describe_action_bounds(
    checkpoint: str | Path,
    *,
    margin: float = DEFAULT_BOUND_MARGIN,
    action_dim: int | None = None,
) -> dict[str, Any]:
    """JSON-friendly summary for the Stage 2 CPU preflight."""
    stats = read_action_norm_stats(checkpoint)
    lo, hi = normalized_action_bounds(
        checkpoint, margin=margin, action_dim=action_dim, stats=stats
    )
    return {
        "mode": stats.mode,
        "margin": float(margin),
        "state_file": stats.state_file,
        "lo": [round(value, 4) for value in lo.tolist()],
        "hi": [round(value, 4) for value in hi.tolist()],
    }
