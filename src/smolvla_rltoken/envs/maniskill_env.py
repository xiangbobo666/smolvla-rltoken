"""ManiSkill environments used by this project.

The stock ``PegInsertionSide-v1`` task exposes a 128px scene camera and a
128px wrist camera.  ``PegInsertionSideThreeCamera-v1`` keeps the task and
robot unchanged, but adds a close-up camera mounted on the peg.  Its local
frame is anchored at the red insertion end, so it follows that end rather
than the randomized hole or a fixed world point.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env


# Environment-camera coordinates are in the task's world frame (metres).
# The insertion camera is mounted on the peg.  Its coordinates are offsets
# from the red insertion end, expressed in the peg frame.
DEFAULT_CAMERA_SPECS: dict[str, dict[str, Any]] = {
    "environment_camera": {
        "position": [0.45, -0.55, 0.45],
        "target": [0.0, 0.12, 0.10],
        "fov_deg": 70.0,
        "width": 512,
        "height": 512,
    },
    "insertion_camera": {
        # Start exactly at the red insertion end.  The non-zero target only
        # supplies a valid initial viewing direction (towards the hole).
        "relative_position": [0.0, 0.0, 0.0],
        "relative_target": [0.12, 0.0, 0.0],
        "fov_deg": 60.0,
        "width": 512,
        "height": 512,
    },
    # The wrist camera is mounted on Panda's ``camera_link``.  Only its image
    # settings and *local* optical offset belong here; its global pose changes
    # with the robot arm.
    "hand_camera": {
        "width": 512,
        "height": 512,
        "fov_deg": 90.0,
        "local_position": [0.0, 0.0, 0.0],
        "local_rpy_deg": [0.0, 0.0, 0.0],
    },
}


def _merge_camera_specs(overrides: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Merge user overrides without silently accepting misspelled camera names."""
    specs = deepcopy(DEFAULT_CAMERA_SPECS)
    if overrides is None:
        return specs

    unknown = set(overrides) - set(specs)
    if unknown:
        raise ValueError(f"Unknown camera spec(s): {sorted(unknown)}")
    for uid, values in overrides.items():
        specs[uid].update(values)
    return specs


def _static_camera_config(uid: str, spec: dict[str, Any]) -> CameraConfig:
    required = {"position", "target", "fov_deg", "width", "height"}
    missing = required - set(spec)
    if missing:
        raise ValueError(f"Camera '{uid}' is missing: {sorted(missing)}")

    pose = sapien_utils.look_at(spec["position"], spec["target"])
    return CameraConfig(
        uid=uid,
        pose=pose,
        width=int(spec["width"]),
        height=int(spec["height"]),
        fov=np.deg2rad(float(spec["fov_deg"])),
        near=0.01,
        far=100,
        shader_pack="minimal",
    )


def peg_end_camera_pose(spec: dict[str, Any], peg_end_in_peg: np.ndarray | Pose) -> Pose:
    """Return the camera pose in the peg frame from its red-end-relative spec."""
    required = {"relative_position", "relative_target", "fov_deg", "width", "height"}
    missing = required - set(spec)
    if missing:
        raise ValueError(f"insertion_camera is missing: {sorted(missing)}")
    relative_pose = sapien_utils.look_at(spec["relative_position"], spec["relative_target"])
    # ``peg_end_in_peg`` only translates along the peg x-axis, but composing
    # poses (instead of merely adding positions) keeps the convention correct
    # if its local orientation ever changes.
    if isinstance(peg_end_in_peg, Pose):
        peg_end_pose = peg_end_in_peg
    else:
        peg_end_pose = Pose.create(sapien.Pose(np.asarray(peg_end_in_peg, dtype=np.float64)))
    return peg_end_pose * relative_pose


def _insertion_camera_config(spec: dict[str, Any], peg, peg_end_in_peg: np.ndarray) -> CameraConfig:
    pose = peg_end_camera_pose(spec, peg_end_in_peg)
    return CameraConfig(
        uid="insertion_camera",
        pose=pose,
        width=int(spec["width"]),
        height=int(spec["height"]),
        fov=np.deg2rad(float(spec["fov_deg"])),
        near=0.01,
        far=100,
        mount=peg,
        shader_pack="minimal",
    )


def _quaternion_from_rpy_deg(rpy_deg: list[float]) -> list[float]:
    """Return a scalar-first quaternion for roll, pitch, yaw rotations."""
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, dtype=np.float64))
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return [
        float(cr * cp * cy + sr * sp * sy),
        float(sr * cp * cy - cr * sp * sy),
        float(cr * sp * cy + sr * cp * sy),
        float(cr * cp * sy - sr * sp * cy),
    ]


def hand_camera_pose(spec: dict[str, Any]) -> sapien.Pose:
    """Build the wrist camera's pose in Panda's ``camera_link`` frame."""
    position = spec.get("local_position", [0.0, 0.0, 0.0])
    rpy_deg = spec.get("local_rpy_deg", [0.0, 0.0, 0.0])
    if len(position) != 3 or len(rpy_deg) != 3:
        raise ValueError("hand_camera local_position and local_rpy_deg must each have three values")
    return sapien.Pose(position, _quaternion_from_rpy_deg(rpy_deg))


def hand_camera_sensor_overrides(spec: dict[str, Any]) -> dict[str, Any]:
    """Translate project camera fields to standard ManiSkill sensor fields."""
    pose = hand_camera_pose(spec)
    return {
        "width": int(spec["width"]),
        "height": int(spec["height"]),
        "fov": float(np.deg2rad(spec.get("fov_deg", 90.0))),
        # BaseEnv accepts a seven-value pose list and converts it to sapien.Pose.
        "pose": [*pose.p.tolist(), *pose.q.tolist()],
    }


@register_env("PegInsertionSideThreeCamera-v1", max_episode_steps=200)
class PegInsertionSideThreeCameraEnv(PegInsertionSideEnv):
    """Peg insertion with environment, wrist, and red-peg-end cameras.

    Pass ``camera_specs`` to ``gym.make`` to override only the values being
    tuned, e.g. ``{"insertion_camera": {"relative_position": [0.0, -0.3, 0.3]}}``.
    """

    def __init__(self, *args, camera_specs: dict[str, dict[str, Any]] | None = None, **kwargs):
        self.camera_specs = _merge_camera_specs(camera_specs)

        # The wrist camera is defined by PandaWristCam, so override its image
        # dimensions through BaseEnv's supported sensor-config hook.
        sensor_configs = deepcopy(kwargs.pop("sensor_configs", {}))
        hand_overrides = hand_camera_sensor_overrides(self.camera_specs["hand_camera"])
        sensor_configs["hand_camera"] = {
            **hand_overrides,
            **sensor_configs.get("hand_camera", {}),
        }
        super().__init__(*args, sensor_configs=sensor_configs, **kwargs)

    @property
    def _default_sensor_configs(self):
        # `_setup_sensors` is invoked after `_load_scene`, so the randomized
        # per-episode peg length (and therefore its red-end offset) exists.
        peg_end = self.peg_head_offsets
        return [
            _static_camera_config("environment_camera", self.camera_specs["environment_camera"]),
            _insertion_camera_config(
                self.camera_specs["insertion_camera"], self.peg, peg_end
            ),
        ]
