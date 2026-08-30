#!/usr/bin/env python3
"""Render side-by-side three-camera MP4 videos from recorded PegInsertion states.

The script restores each episode's recorded seed before applying its saved
states.  This recreates the randomized peg and box geometry exactly, while
using the camera settings from the supplied JSON file.  It never changes the
trajectory, dataset, or camera configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

import mani_skill.envs  # Registers stock environments used by the parent task.
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
DEFAULT_CONFIG = REPO_ROOT / "configs/vla/peg_insertion_three_cameras.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/camera_videos"
CAMERA_ORDER = ("environment_camera", "hand_camera", "insertion_camera")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        help="Explicit episode IDs. Omits random selection when provided.",
    )
    parser.add_argument("--count", type=int, default=5, help="Number of random episodes.")
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=20260830,
        help="Seed used only to select random episodes.",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing MP4 with the same episode ID.",
    )
    return parser.parse_args()


def to_uint8_rgb(image: Any) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(f"Expected one environment, got image shape {image.shape}")
        image = image[0]
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image


def make_montage(images: dict[str, np.ndarray], episode: int, frame: int) -> np.ndarray:
    frames = [Image.fromarray(images[name]) for name in CAMERA_ORDER]
    width = sum(frame_image.width for frame_image in frames)
    height = max(frame_image.height for frame_image in frames) + 36
    montage = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(montage)
    x = 0
    for name, frame_image in zip(CAMERA_ORDER, frames, strict=True):
        montage.paste(frame_image, (x, 36))
        draw.text((x + 8, 10), name, fill="white")
        x += frame_image.width
    draw.text((width - 190, 10), f"episode {episode}  frame {frame}", fill="white")
    return np.asarray(montage)


def read_metadata(trajectory_path: Path) -> dict[int, int]:
    with trajectory_path.with_suffix(".json").open() as metadata_file:
        episodes = json.load(metadata_file)["episodes"]
    return {int(item["episode_id"]): int(item["episode_seed"]) for item in episodes}


def select_episodes(args: argparse.Namespace, episode_seeds: dict[int, int]) -> list[int]:
    if args.episodes is not None:
        episode_ids = list(dict.fromkeys(args.episodes))
        missing = sorted(set(episode_ids) - set(episode_seeds))
        if missing:
            raise ValueError(f"Episode IDs absent from trajectory metadata: {missing}")
        return episode_ids
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.count > len(episode_seeds):
        raise ValueError(f"--count exceeds available episodes ({len(episode_seeds)})")
    return random.Random(args.sample_seed).sample(sorted(episode_seeds), args.count)


def states_for_episode(h5_file: h5py.File, episode: int) -> list[dict[str, Any]]:
    trajectory_key = f"traj_{episode}"
    if trajectory_key not in h5_file:
        raise KeyError(f"{trajectory_key} is not present in {h5_file.filename}")
    return trajectory_utils.dict_to_list_of_dicts(h5_file[trajectory_key]["env_states"])


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    config_bytes = args.config.read_bytes()
    camera_specs = json.loads(config_bytes)
    episode_seeds = read_metadata(args.trajectory)
    episode_ids = select_episodes(args, episode_seeds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        episode: args.output_dir / f"episode_{episode:04d}_three_cameras.mp4"
        for episode in episode_ids
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not args.overwrite:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Video output already exists: {formatted}. Pass --overwrite to replace it.")

    summary: list[dict[str, Any]] = []
    env = gym.make(
        "PegInsertionSideThreeCamera-v1",
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        sim_backend="physx_cpu",
        camera_specs=camera_specs,
    )
    try:
        with h5py.File(args.trajectory, "r") as h5_file:
            for episode in episode_ids:
                states = states_for_episode(h5_file, episode)
                output_path = targets[episode]
                temporary_path = output_path.with_suffix(".tmp.mp4")
                if temporary_path.exists():
                    temporary_path.unlink()
                print(f"Rendering episode {episode} ({len(states)} frames) -> {output_path}", flush=True)
                # A reset reconfigures the single-environment task, restoring
                # the seed-dependent peg length, box dimensions, and hole.
                env.reset(seed=episode_seeds[episode])
                with imageio.get_writer(
                    temporary_path,
                    fps=args.fps,
                    codec="libx264",
                    pixelformat="yuv420p",
                    macro_block_size=2,
                ) as writer:
                    for frame_index, state in enumerate(states):
                        env.unwrapped.set_state_dict(state)
                        sensor_images = env.unwrapped.get_sensor_images()
                        rgb_images = {
                            name: to_uint8_rgb(sensor_images[name]["rgb"])
                            for name in CAMERA_ORDER
                        }
                        writer.append_data(make_montage(rgb_images, episode, frame_index))
                temporary_path.replace(output_path)
                summary.append(
                    {
                        "episode": episode,
                        "episode_seed": episode_seeds[episode],
                        "frames": len(states),
                        "duration_seconds": len(states) / args.fps,
                        "video": output_path.name,
                    }
                )
    finally:
        env.close()

    manifest_path = args.output_dir / "render_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "trajectory": str(args.trajectory),
                "camera_config": str(args.config),
                "camera_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "fps": args.fps,
                "sample_seed": None if args.episodes is not None else args.sample_seed,
                "episodes": summary,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved manifest {manifest_path}")


if __name__ == "__main__":
    main()
