#!/usr/bin/env python3
"""Render a three-camera still from a recorded PegInsertionSide trajectory.

This script uses stored environment states rather than replaying actions, so a
camera adjustment can be checked on the exact demonstration pose immediately.
It does not modify the source trajectory or the training dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
from PIL import Image, ImageDraw

import mani_skill.envs  # Registers stock environments used by the parent task.
from mani_skill.trajectory import utils as trajectory_utils

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Importing the module registers PegInsertionSideThreeCamera-v1.
import smolvla_rltoken.envs.maniskill_env  # noqa: F401


DEFAULT_TRAJECTORY = (
    REPO_ROOT
    / "data/maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.h5"
)
DEFAULT_CONFIG = REPO_ROOT / "configs/vla/peg_insertion_three_cameras.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/camera_previews"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--episode", type=int, default=0, help="Demonstration episode index.")
    parser.add_argument(
        "--frame",
        type=int,
        default=-1,
        help="State frame to render; -1 selects the final successful state.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_state(trajectory_path: Path, episode: int, frame: int) -> tuple[dict[str, Any], int, int]:
    with h5py.File(trajectory_path, "r") as h5_file:
        trajectory_key = f"traj_{episode}"
        if trajectory_key not in h5_file:
            raise KeyError(f"{trajectory_key} is not present in {trajectory_path}")
        states = trajectory_utils.dict_to_list_of_dicts(h5_file[trajectory_key]["env_states"])
    if not -len(states) <= frame < len(states):
        raise IndexError(f"frame {frame} is outside [{-len(states)}, {len(states) - 1}]")
    resolved_frame = frame % len(states)
    with trajectory_path.with_suffix(".json").open() as metadata_file:
        metadata = json.load(metadata_file)
    episode_seed = next(
        item["episode_seed"] for item in metadata["episodes"] if item["episode_id"] == episode
    )
    return states[resolved_frame], resolved_frame, int(episode_seed)


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


def make_montage(images: dict[str, np.ndarray]) -> Image.Image:
    ordered_names = ["environment_camera", "hand_camera", "insertion_camera"]
    frames = [Image.fromarray(images[name]) for name in ordered_names]
    width = sum(frame.width for frame in frames)
    height = max(frame.height for frame in frames) + 36
    montage = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(montage)
    x = 0
    for name, frame in zip(ordered_names, frames, strict=True):
        montage.paste(frame, (x, 36))
        draw.text((x + 8, 10), name, fill="white")
        x += frame.width
    return montage


def main() -> None:
    args = parse_args()
    with args.config.open() as config_file:
        camera_specs = json.load(config_file)
    state, resolved_frame, episode_seed = read_state(args.trajectory, args.episode, args.frame)

    env = gym.make(
        "PegInsertionSideThreeCamera-v1",
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        sim_backend="physx_cpu",
        camera_specs=camera_specs,
    )
    try:
        # Restore the recorded seed so the randomized box/hole geometry matches
        # the demonstration before applying its saved dynamic state.
        env.reset(seed=episode_seed)
        env.unwrapped.set_state_dict(state)
        sensor_images = env.unwrapped.get_sensor_images()
        rgb_images = {
            name: to_uint8_rgb(sensor_images[name]["rgb"]) for name in sensor_images
        }
    finally:
        env.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"episode_{args.episode:04d}_frame_{resolved_frame:04d}.png"
    make_montage(rgb_images).save(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
