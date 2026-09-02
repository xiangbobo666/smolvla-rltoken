"""VLA loading and prefix extraction. Submodules import LeRobot lazily."""

from __future__ import annotations

__all__ = [
    "SmolVLAPrefixExtractor",
    "apply_camera_rename",
    "build_dataset_and_processors",
    "dataset_delta_timestamps",
    "describe_action_bounds",
    "load_smolvla_policy",
    "normalized_action_bounds",
    "read_action_norm_stats",
]


def __getattr__(name: str):
    if name == "SmolVLAPrefixExtractor":
        from smolvla_rltoken.vla.extractor import SmolVLAPrefixExtractor

        return SmolVLAPrefixExtractor
    if name in {"describe_action_bounds", "normalized_action_bounds", "read_action_norm_stats"}:
        from smolvla_rltoken.vla import action_bounds as action_bounds_mod

        return getattr(action_bounds_mod, name)
    if name in {"apply_camera_rename", "build_dataset_and_processors", "dataset_delta_timestamps"}:
        from smolvla_rltoken.vla import dataset as dataset_mod

        return getattr(dataset_mod, name)
    if name == "load_smolvla_policy":
        from smolvla_rltoken.vla.load import load_smolvla_policy

        return load_smolvla_policy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
