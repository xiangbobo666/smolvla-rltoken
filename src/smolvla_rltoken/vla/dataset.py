"""LeRobot dataset + SmolVLA preprocessor for Stage 1.

Keeps the policy on smolvla_base / SFT keys (``camera1/2/3``). Dataset keys
(``environment_camera`` / ``hand_camera`` / ``insertion_camera``) are renamed
with the same map as SFT before ``prepare_images``.
"""

from __future__ import annotations

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.processor.rename_processor import RenameObservationsProcessorStep, rename_stats

from smolvla_rltoken.paths import SFT_IMAGE_RENAME_MAP


def apply_camera_rename(preprocessor, rename_map: dict[str, str] | None = None) -> None:
    """Write the SFT camera rename map onto an existing SmolVLA preprocessor."""
    mapping = dict(rename_map or SFT_IMAGE_RENAME_MAP)
    found = False
    for step in preprocessor.steps:
        if isinstance(step, RenameObservationsProcessorStep):
            step.rename_map = mapping
            found = True
    if not found:
        raise RuntimeError("preprocessor has no RenameObservationsProcessorStep")


def dataset_delta_timestamps(policy, meta: LeRobotDatasetMetadata) -> dict[str, list[float]]:
    """Delta timestamps keyed by *dataset* feature names, not policy camera1/2/3."""
    delta: dict[str, list[float]] = {}
    for key in meta.features:
        if key.startswith("observation."):
            delta[key] = [0.0]
        elif key.startswith("action"):
            delta[key] = [i / meta.fps for i in policy.config.action_delta_indices]
    return delta


def build_dataset_and_processors(
    policy,
    dataset_repo: str,
    dataset_root: str | None,
    rename_map: dict[str, str] | None = None,
    video_backend: str = "torchcodec",
    episodes: list[int] | None = None,
):
    """LeRobot demo set plus the official SmolVLA preprocessor.

    Does not rewrite ``policy.config.input_features``. The batch after the
    preprocessor uses ``camera1/2/3``, matching SFT and ``prepare_images``.
    ``episodes`` selects a subset by ``episode_index`` (None = all).
    """
    mapping = dict(rename_map or SFT_IMAGE_RENAME_MAP)
    meta = LeRobotDatasetMetadata(dataset_repo, root=dataset_root)
    stats = rename_stats(meta.stats, mapping)
    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        policy.config, dataset_stats=stats
    )
    apply_camera_rename(preprocessor, mapping)

    dataset = LeRobotDataset(
        dataset_repo,
        root=dataset_root,
        episodes=episodes,
        delta_timestamps=dataset_delta_timestamps(policy, meta),
        video_backend=video_backend,
    )
    return dataset, preprocessor, postprocessor
