"""LeRobot dataset + SmolVLA preprocessor for Stage 1."""

from __future__ import annotations

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors


def apply_dataset_features(policy, meta: LeRobotDatasetMetadata) -> None:
    """Replace pretrained camera/state keys with this dataset's features.

    smolvla_base uses observation.images.camera{1,2,3}; PegInsertion uses
    environment_camera + hand_camera + insertion_camera. Stage 1 reads the
    batch after the preprocessor, so the policy must look up the dataset keys.
    """
    features = dataset_to_policy_features(meta.features)
    output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    input_features = {k: f for k, f in features.items() if k not in output_features}
    policy.config.input_features = input_features
    policy.config.output_features = output_features
    policy.config.validate_features()


def build_dataset_and_processors(policy, dataset_repo: str, dataset_root: str | None):
    """LeRobot demo set plus the official SmolVLA preprocessor."""
    meta = LeRobotDatasetMetadata(dataset_repo, root=dataset_root)
    apply_dataset_features(policy, meta)
    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        policy.config, dataset_stats=meta.stats
    )

    delta_timestamps = {
        k: [0.0] for k in policy.config.input_features if k.startswith("observation.")
    }
    for k in policy.config.output_features:
        if k.startswith("action"):
            delta_timestamps[k] = [i / meta.fps for i in policy.config.action_delta_indices]

    dataset = LeRobotDataset(
        dataset_repo,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    return dataset, preprocessor, postprocessor
