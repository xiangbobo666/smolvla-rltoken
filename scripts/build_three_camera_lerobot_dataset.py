#!/usr/bin/env python3
"""Rebuild PegInsertion LeRobot data with current 512px three-camera observations.

The source ``trajectory.h5`` is authoritative for state and action. Episodes
are replayed in small GPU batches using their recorded seeds; this recreates
the randomized geometry before the saved state is applied. RGB is streamed to
LeRobot-compatible H.264 videos, without an intermediate RGB HDF5 file.

The output directory must not exist. Build and validate there first, then move
it into the training dataset path only after a successful complete run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

import mani_skill.envs  # Registers stock environments used by the parent task.
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.video_utils import StreamingVideoEncoder
from mani_skill.trajectory import utils as trajectory_utils

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Importing the module registers PegInsertionSideThreeCamera-v1.
import smolvla_rltoken.envs.maniskill_env  # noqa: F401, E402


DEFAULT_TRAJECTORY = (
    REPO_ROOT
    / "data/maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.h5"
)
DEFAULT_CAMERA_CONFIG = REPO_ROOT / "configs/vla/peg_insertion_three_cameras.json"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "data/lerobot/PegInsertionSide-v1/motionplanning_rgb_pd_joint_pos.rebuild"
)
LEGACY_DATASET_ROOT = (
    REPO_ROOT / "data/lerobot/PegInsertionSide-v1/motionplanning_rgb_pd_joint_pos"
)
CAMERA_NAMES = ("environment_camera", "hand_camera", "insertion_camera")
TASK = "Insert the peg into the hole from the side."
FPS = 20
CHUNKS_SIZE = 1000
# Articulation state layout: root pose (7), root velocity (6), qpos (9), qvel (9).
PANDA_QPOS_SLICE = slice(13, 22)
STATS_KEYS = ("min", "max", "mean", "std", "count")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        help="Optional explicit source episodes, useful for a smoke build; default is all 1,000.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--vcodec",
        choices=("h264_nvenc", "h264"),
        default="h264",
        help="H.264 software encoding is the reliable default; h264_nvenc is optional when PyAV supports the active CUDA context.",
    )
    parser.add_argument("--encoder-queue-maxsize", type=int, default=512)
    return parser.parse_args()


def read_episode_seeds(trajectory_path: Path) -> dict[int, int]:
    with trajectory_path.with_suffix(".json").open() as metadata_file:
        return {
            int(item["episode_id"]): int(item["episode_seed"])
            for item in json.load(metadata_file)["episodes"]
        }


def select_episodes(args: argparse.Namespace, seeds: dict[int, int]) -> list[int]:
    if args.episodes is None:
        return list(seeds)
    episode_ids = list(dict.fromkeys(args.episodes))
    missing = sorted(set(episode_ids) - set(seeds))
    if missing:
        raise ValueError(f"Source episode(s) not found: {missing}")
    return episode_ids


def load_episode(
    h5_file: h5py.File, episode_id: int
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    trajectory = h5_file[f"traj_{episode_id}"]
    states = trajectory_utils.dict_to_list_of_dicts(trajectory["env_states"])
    actions = np.asarray(trajectory["actions"], dtype=np.float32)
    articulation = np.asarray(
        trajectory["env_states"]["articulations"]["panda_wristcam"], dtype=np.float32
    )
    qpos = articulation[:, PANDA_QPOS_SLICE]
    if len(states) < len(actions) or len(qpos) < len(actions):
        raise ValueError(f"traj_{episode_id} has inconsistent state/action lengths")
    if actions.shape[1:] != (8,) or qpos.shape[1:] != (9,):
        raise ValueError(f"traj_{episode_id} has unexpected action/state shapes: {actions.shape}, {qpos.shape}")
    return states[: len(actions)], qpos[: len(actions)], actions


def array_stats(array: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(array)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "count": np.array([len(values)], dtype=np.int64),
    }


def to_lerobot_video_stats(raw_stats: dict[str, np.ndarray], frame_count: int) -> dict[str, np.ndarray]:
    """Match LeRobot streaming-encoder image-stat shapes and frame-count convention."""
    stats: dict[str, np.ndarray] = {"count": np.array([frame_count], dtype=np.int64)}
    for key in ("min", "max", "mean", "std"):
        values = np.asarray(raw_stats[key], dtype=np.float64)
        stats[key] = np.squeeze(values.reshape(1, -1, 1, 1) / 255.0, axis=0)
    return stats


def camera_feature() -> dict[str, Any]:
    return {
        "dtype": "video",
        "shape": [512, 512, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.fps": float(FPS),
            "video.height": 512,
            "video.width": 512,
            "video.channels": 3,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }


def dataset_features() -> dict[str, Any]:
    features: dict[str, Any] = {
        "action": {
            "dtype": "float32",
            "shape": [8],
            "names": [f"action_{index}" for index in range(8)],
            "fps": float(FPS),
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [9],
            "names": [f"joint_{index}" for index in range(9)],
            "fps": float(FPS),
        },
    }
    for name, dtype in (("timestamp", "float32"), ("frame_index", "int64"), ("episode_index", "int64"), ("index", "int64"), ("task_index", "int64")):
        features[name] = {"dtype": dtype, "shape": [1], "names": None, "fps": float(FPS)}
    features["task"] = {"dtype": "string", "shape": [1], "names": None, "fps": float(FPS)}
    for camera_name in CAMERA_NAMES:
        features[f"observation.images.{camera_name}"] = camera_feature()
    return features


def as_uint8_batch(image: Any) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def video_path(root: Path, camera_name: str, episode_index: int) -> Path:
    return root / "videos" / f"observation.images.{camera_name}" / "chunk-000" / f"file-{episode_index:03d}.mp4"


def move_encoded_videos(
    root: Path,
    episode_index: int,
    encoder: StreamingVideoEncoder,
    frame_count: int,
) -> dict[str, dict[str, np.ndarray]]:
    result = encoder.finish_episode()
    episode_stats: dict[str, dict[str, np.ndarray]] = {}
    for camera_name in CAMERA_NAMES:
        key = f"observation.images.{camera_name}"
        temporary_path, raw_stats = result[key]
        if raw_stats is None:
            raise RuntimeError(f"No video statistics returned for episode {episode_index}, {camera_name}")
        destination = video_path(root, camera_name, episode_index)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.replace(destination)
        temporary_directory = temporary_path.parent
        if temporary_directory.exists() and not any(temporary_directory.iterdir()):
            temporary_directory.rmdir()
        episode_stats[key] = to_lerobot_video_stats(raw_stats, frame_count)
    return episode_stats


def write_frame_parquet(root: Path, episodes: list[dict[str, Any]]) -> int:
    actions = np.concatenate([episode["actions"] for episode in episodes])
    states = np.concatenate([episode["qpos"] for episode in episodes])
    lengths = [len(episode["actions"]) for episode in episodes]
    frame_indices = np.concatenate([np.arange(length, dtype=np.int64) for length in lengths])
    timestamps = np.concatenate([np.arange(length, dtype=np.float32) / FPS for length in lengths])
    episode_indices = np.concatenate(
        [np.full(length, index, dtype=np.int64) for index, length in enumerate(lengths)]
    )
    total_frames = len(actions)
    table = pa.Table.from_pydict(
        {
            "action": pa.array(actions.tolist(), type=pa.list_(pa.float32())),
            "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32())),
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(frame_indices, type=pa.int64()),
            "episode_index": pa.array(episode_indices, type=pa.int64()),
            "index": pa.array(np.arange(total_frames, dtype=np.int64), type=pa.int64()),
            "task_index": pa.array(np.zeros(total_frames, dtype=np.int64), type=pa.int64()),
            "task": pa.array([TASK] * total_frames, type=pa.string()),
        }
    )
    output_path = root / "data/chunk-000/file-000.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="snappy")
    return total_frames


def stats_to_json(stats: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, Any]]:
    return {
        feature: {key: np.asarray(values[key]).tolist() for key in STATS_KEYS}
        for feature, values in stats.items()
    }


def write_episode_metadata(root: Path, episodes: list[dict[str, Any]]) -> None:
    columns: dict[str, list[Any]] = {
        "episode_index": [],
        "data/chunk_index": [],
        "data/file_index": [],
        "dataset_from_index": [],
        "dataset_to_index": [],
        "tasks": [],
        "length": [],
        "meta/episodes/chunk_index": [],
        "meta/episodes/file_index": [],
    }
    for camera_name in CAMERA_NAMES:
        key = f"observation.images.{camera_name}"
        for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
            columns[f"videos/{key}/{suffix}"] = []
    feature_names = ("action", "observation.state", *(f"observation.images.{name}" for name in CAMERA_NAMES), "timestamp", "frame_index", "episode_index", "index", "task_index")
    for feature in feature_names:
        for stat_key in STATS_KEYS:
            columns[f"stats/{feature}/{stat_key}"] = []

    first_index = 0
    for episode_index, episode in enumerate(episodes):
        length = len(episode["actions"])
        columns["episode_index"].append(episode_index)
        columns["data/chunk_index"].append(0)
        columns["data/file_index"].append(0)
        columns["dataset_from_index"].append(first_index)
        columns["dataset_to_index"].append(first_index + length)
        columns["tasks"].append([TASK])
        columns["length"].append(length)
        columns["meta/episodes/chunk_index"].append(0)
        columns["meta/episodes/file_index"].append(0)
        for camera_name in CAMERA_NAMES:
            key = f"observation.images.{camera_name}"
            columns[f"videos/{key}/chunk_index"].append(0)
            columns[f"videos/{key}/file_index"].append(episode_index)
            columns[f"videos/{key}/from_timestamp"].append(0.0)
            columns[f"videos/{key}/to_timestamp"].append((length - 1) / FPS)
        for feature in feature_names:
            for stat_key in STATS_KEYS:
                columns[f"stats/{feature}/{stat_key}"].append(
                    np.asarray(episode["stats"][feature][stat_key]).tolist()
                )
        first_index += length

    output_path = root / "meta/episodes/chunk-000/file-000.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict(columns), output_path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_metadata(root: Path, episodes: list[dict[str, Any]], config_bytes: bytes, args: argparse.Namespace) -> None:
    total_frames = sum(len(episode["actions"]) for episode in episodes)
    global_stats = aggregate_stats([episode["stats"] for episode in episodes])
    data_path = root / "data/chunk-000/file-000.parquet"
    info = {
        "codebase_version": "v3.0",
        "robot_type": "panda",
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(episodes) * len(CAMERA_NAMES),
        "total_chunks": 1,
        "chunks_size": CHUNKS_SIZE,
        "fps": FPS,
        "data_files_size_in_mb": int(data_path.stat().st_size / (1024 * 1024)),
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": dataset_features(),
    }
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    (meta_dir / "stats.json").write_text(json.dumps(stats_to_json(global_stats), indent=2) + "\n")
    legacy_tasks = LEGACY_DATASET_ROOT / "meta/tasks.parquet"
    if not legacy_tasks.is_file():
        raise FileNotFoundError(f"Cannot copy the known task table: {legacy_tasks}")
    shutil.copy2(legacy_tasks, meta_dir / "tasks.parquet")
    root.joinpath(".gitattributes").write_text("*.mp4 filter=lfs diff=lfs merge=lfs -text\n")
    root.joinpath("camera_config.json").write_bytes(config_bytes)
    root.joinpath("render_manifest.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "source_trajectory": str(args.trajectory),
                "source_trajectory_sha256": sha256(args.trajectory),
                "camera_config": str(args.camera_config),
                "camera_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "camera_uids": list(CAMERA_NAMES),
                "image_size": [512, 512],
                "fps": FPS,
                "codec": args.vcodec,
                "source_episodes": [episode["source_episode"] for episode in episodes],
                "episode_seeds": [episode["seed"] for episode in episodes],
                "total_frames": total_frames,
            },
            indent=2,
        )
        + "\n"
    )
    root.joinpath("README.md").write_text(
        "---\npretty_name: SmolVLA-RLT PegInsertionSide-v1 Three Cameras\n"
        "tags:\n  - robotics\n  - lerobot\n  - maniskill\n  - imitation-learning\n---\n\n"
        "# SmolVLA-RLT: PegInsertionSide-v1 (three cameras)\n\n"
        "Successful motion-planning demonstrations replayed from saved ManiSkill states. "
        "Every 20 FPS observation provides 512×512 RGB from `environment_camera`, "
        "`hand_camera`, and `insertion_camera`. The insertion camera is mounted in the "
        "peg frame with its reference at the red insertion end. Actions are 8D absolute "
        "`pd_joint_pos` commands; `observation.state` is 9D Panda qpos. Exact source and "
        "camera-config hashes are in `render_manifest.json`.\n"
    )


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Output root must not exist: {args.output_root}")
    if args.batch_size < 1 or args.encoder_queue_maxsize < 1:
        raise ValueError("--batch-size and --encoder-queue-maxsize must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for batched rendering")
    config_bytes = args.camera_config.read_bytes()
    specs = json.loads(config_bytes)
    if set(specs) != {"environment_camera", "hand_camera", "insertion_camera"}:
        raise ValueError("Camera config must define exactly the three project cameras")
    seeds = read_episode_seeds(args.trajectory)
    source_episode_ids = select_episodes(args, seeds)
    args.output_root.mkdir(parents=True)
    episodes: list[dict[str, Any]] = []

    env = gym.make(
        "PegInsertionSideThreeCamera-v1",
        num_envs=args.batch_size,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        sim_backend="physx_cuda",
        camera_specs=specs,
    )
    try:
        with h5py.File(args.trajectory, "r") as h5_file:
            for group_start in range(0, len(source_episode_ids), args.batch_size):
                actual_ids = source_episode_ids[group_start : group_start + args.batch_size]
                render_ids = actual_ids + [actual_ids[-1]] * (args.batch_size - len(actual_ids))
                loaded = [load_episode(h5_file, source_id) for source_id in render_ids]
                env.reset(
                    seed=torch.tensor([seeds[source_id] for source_id in render_ids], device="cuda"),
                    options={"reconfigure": True},
                )
                encoders = []
                for _ in actual_ids:
                    encoder = StreamingVideoEncoder(
                        fps=FPS,
                        vcodec=args.vcodec,
                        pix_fmt="yuv420p",
                        g=2,
                        crf=30,
                        queue_maxsize=args.encoder_queue_maxsize,
                    )
                    encoder.start_episode(
                        [f"observation.images.{camera_name}" for camera_name in CAMERA_NAMES],
                        args.output_root,
                    )
                    encoders.append(encoder)
                max_frames = max(len(loaded[index][2]) for index in range(len(actual_ids)))
                for frame_index in range(max_frames):
                    state_batch = trajectory_utils.list_of_dicts_to_dict(
                        [states[min(frame_index, len(actions) - 1)] for states, _, actions in loaded]
                    )
                    env.unwrapped.set_state_dict(state_batch)
                    rendered = env.unwrapped.get_sensor_images()
                    cpu_images = {
                        camera_name: as_uint8_batch(rendered[camera_name]["rgb"])
                        for camera_name in CAMERA_NAMES
                    }
                    for env_index, encoder in enumerate(encoders):
                        if frame_index >= len(loaded[env_index][2]):
                            continue
                        for camera_name in CAMERA_NAMES:
                            encoder.feed_frame(
                                f"observation.images.{camera_name}", cpu_images[camera_name][env_index]
                            )
                for env_index, source_id in enumerate(actual_ids):
                    states, qpos, actions = loaded[env_index]
                    episode_index = len(episodes)
                    video_stats = move_encoded_videos(
                        args.output_root, episode_index, encoders[env_index], len(actions)
                    )
                    timestamps = np.arange(len(actions), dtype=np.float32) / FPS
                    episode_stats = {
                        "action": array_stats(actions),
                        "observation.state": array_stats(qpos),
                        "timestamp": array_stats(timestamps),
                        "frame_index": array_stats(np.arange(len(actions), dtype=np.int64)),
                        "episode_index": array_stats(np.full(len(actions), episode_index, dtype=np.int64)),
                        "index": array_stats(np.arange(len(actions), dtype=np.int64)),
                        "task_index": array_stats(np.zeros(len(actions), dtype=np.int64)),
                        **video_stats,
                    }
                    episodes.append(
                        {
                            "source_episode": source_id,
                            "seed": seeds[source_id],
                            "qpos": qpos,
                            "actions": actions,
                            "stats": episode_stats,
                        }
                    )
                if len(episodes) % 20 == 0 or len(episodes) == len(source_episode_ids):
                    print(f"Rendered {len(episodes)}/{len(source_episode_ids)} episodes", flush=True)
    finally:
        env.close()

    total_frames = write_frame_parquet(args.output_root, episodes)
    write_episode_metadata(args.output_root, episodes)
    write_metadata(args.output_root, episodes, config_bytes, args)
    print(
        f"Completed {len(episodes)} episodes / {total_frames} frames at {args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
