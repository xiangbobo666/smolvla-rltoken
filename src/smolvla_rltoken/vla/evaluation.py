"""Automatic SFT evaluation on the three-camera ManiSkill peg task.

The small, environment-agnostic rollout functions in this module intentionally
do not import ManiSkill, LeRobot, or torch at module import time.  That keeps
the success/timeout accounting unit-testable on CPU while the heavyweight
simulation and policy objects are loaded only for an actual evaluation run.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from smolvla_rltoken.paths import SFT_IMAGE_RENAME_MAP

CAMERA_NAMES = (
    "environment_camera",
    "hand_camera",
    "insertion_camera",
)
TASK_PROMPT = "Insert the peg into the hole from the side."
EXPECTED_STATE_DIM = 9
EXPECTED_ACTION_DIM = 8
DEFAULT_CHUNK_EXECUTION_RATIO = 0.70
DEFAULT_RECORD_SUCCESSES = 5
DEFAULT_RECORD_FAILURES = 5


class PolicyRunner(Protocol):
    """Minimal interface needed by :func:`run_episode`."""

    def reset(self) -> None: ...

    def __call__(self, observation: Mapping[str, Any]) -> np.ndarray: ...


@dataclass(frozen=True)
class EpisodeResult:
    episode_index: int
    seed: int
    success: bool
    steps: int
    termination: str
    reward_sum: float
    video: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OutcomeVideoBudget:
    """Keep videos for the first N successes and first N failures.

    Parallel rollouts do not know the outcome until the episode ends, so the
    evaluator records every unfinished quota slot in a batch and then asks this
    budget which files to keep.
    """

    max_successes: int
    max_failures: int
    kept_successes: int = 0
    kept_failures: int = 0

    def still_recording(self) -> bool:
        return self.kept_successes < self.max_successes or self.kept_failures < self.max_failures

    def decide(self, result: EpisodeResult) -> EpisodeResult:
        """Keep or drop ``result.video`` and rewrite kept paths by outcome."""
        if result.video is None:
            return result
        if result.success:
            if self.kept_successes >= self.max_successes:
                return replace(result, video=None)
            self.kept_successes += 1
            folder = "success"
        else:
            if self.kept_failures >= self.max_failures:
                return replace(result, video=None)
            self.kept_failures += 1
            folder = "failure"
        return replace(result, video=f"videos/{folder}/{Path(result.video).name}")


def delete_video_artifacts(path: Path) -> None:
    path.unlink(missing_ok=True)
    path.with_suffix(".partial.mp4").unlink(missing_ok=True)


def reconcile_recorded_video(result: EpisodeResult, decided: EpisodeResult, output_dir: Path) -> None:
    """Move a kept recording into success/failure, or delete a discarded one."""
    if result.video is None:
        return
    source = output_dir / result.video
    if decided.video is None:
        delete_video_artifacts(source)
        return
    destination = output_dir / decided.video
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        delete_video_artifacts(source)
        raise FileNotFoundError(f"Recorded video is missing: {source}")
    source.replace(destination)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def scalar_bool(value: Any, *, name: str) -> bool:
    array = _as_numpy(value)
    if array.size != 1:
        raise ValueError(f"{name} must contain one value for a single environment; got {array.shape}")
    return bool(array.reshape(-1)[0])


def scalar_float(value: Any, *, name: str) -> float:
    array = _as_numpy(value)
    if array.size != 1:
        raise ValueError(f"{name} must contain one value for a single environment; got {array.shape}")
    return float(array.reshape(-1)[0])


def vector_bool(value: Any, *, size: int, name: str) -> np.ndarray:
    array = _as_numpy(value).astype(bool, copy=False).reshape(-1)
    if array.size != size:
        raise ValueError(f"{name} must contain {size} values; got {array.shape}")
    return array


def vector_float(value: Any, *, size: int, name: str) -> np.ndarray:
    array = _as_numpy(value).astype(np.float64, copy=False).reshape(-1)
    if array.size != size:
        raise ValueError(f"{name} must contain {size} values; got {array.shape}")
    return array


def to_uint8_rgb(image: Any) -> np.ndarray:
    """Convert a single ManiSkill RGB tensor/array to contiguous HWC uint8."""
    array = _as_numpy(image)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"Expected one environment, got image shape {array.shape}")
        array = array[0]
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"Expected HWC RGB(A), got image shape {array.shape}")
    array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.asarray(array, dtype=np.float32)
        if array.size and float(np.nanmax(array)) <= 1.0 + 1e-6:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def to_uint8_rgb_batch(image: Any) -> np.ndarray:
    """Convert batched ManiSkill RGB to contiguous NHWC uint8."""
    array = _as_numpy(image)
    if array.ndim == 3:
        array = array[None]
    if array.ndim != 4 or array.shape[-1] not in (3, 4):
        raise ValueError(f"Expected NHWC RGB(A), got image shape {array.shape}")
    array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.asarray(array, dtype=np.float32)
        if array.size and float(np.nanmax(array)) <= 1.0 + 1e-6:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def extract_policy_observation_batch(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Extract batched state and RGB features without dropping the env axis."""
    try:
        qpos = _as_numpy(observation["agent"]["qpos"])
    except (KeyError, TypeError) as exc:
        raise KeyError("ManiSkill observation is missing agent.qpos") from exc
    if qpos.ndim == 1:
        qpos = qpos[None]
    if qpos.ndim != 2 or qpos.shape[1] != EXPECTED_STATE_DIM:
        raise ValueError(f"Expected batched {EXPECTED_STATE_DIM}D Panda qpos, got {qpos.shape}")

    try:
        sensor_data = observation["sensor_data"]
    except (KeyError, TypeError) as exc:
        raise KeyError("ManiSkill observation is missing sensor_data") from exc

    result: dict[str, np.ndarray] = {
        "observation.state": np.ascontiguousarray(qpos, dtype=np.float32)
    }
    for camera_name in CAMERA_NAMES:
        try:
            rgb = sensor_data[camera_name]["rgb"]
        except (KeyError, TypeError) as exc:
            raise KeyError(f"ManiSkill observation is missing {camera_name}.rgb") from exc
        images = to_uint8_rgb_batch(rgb)
        if images.shape[0] != qpos.shape[0]:
            raise ValueError(
                f"{camera_name} batch {images.shape[0]} does not match state batch {qpos.shape[0]}"
            )
        result[f"observation.images.{camera_name}"] = images
    return result


def extract_policy_observation(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Extract the exact 9D state and three RGB inputs used for SFT.

    ManiSkill's ``rgb`` observation is nested under ``agent`` and
    ``sensor_data``.  The returned keys remain descriptive; the checkpoint's
    saved preprocessor performs the training-time camera rename to camera1/2/3.
    """
    batch = extract_policy_observation_batch(observation)
    if batch["observation.state"].shape[0] != 1:
        raise ValueError(
            f"Expected one environment, got state batch {batch['observation.state'].shape[0]}"
        )
    return {key: np.ascontiguousarray(value[0]) for key, value in batch.items()}


def prepare_policy_observation(
    observation: Mapping[str, Any],
    *,
    task: str = TASK_PROMPT,
) -> dict[str, Any]:
    """Convert a ManiSkill observation to unbatched LeRobot tensors.

    The checkpoint preprocessor adds the batch dimension, tokenizes ``task``,
    moves tensors to the configured device, and applies saved normalization.
    """
    import torch

    raw = extract_policy_observation(observation)
    prepared: dict[str, Any] = {"task": task}
    for key, value in raw.items():
        tensor = torch.from_numpy(value)
        if key.startswith("observation.images."):
            tensor = tensor.permute(2, 0, 1).contiguous().to(torch.float32) / 255.0
        else:
            tensor = tensor.to(torch.float32)
        prepared[key] = tensor
    return prepared


def prepare_policy_observation_batch(
    observation: Mapping[str, Any],
    *,
    task: str = TASK_PROMPT,
) -> dict[str, Any]:
    """Convert batched ManiSkill observations to batched LeRobot tensors."""
    import torch

    raw = extract_policy_observation_batch(observation)
    batch_size = raw["observation.state"].shape[0]
    prepared: dict[str, Any] = {"task": [task] * batch_size}
    for key, value in raw.items():
        tensor = torch.from_numpy(value)
        if key.startswith("observation.images."):
            tensor = tensor.permute(0, 3, 1, 2).contiguous().to(torch.float32) / 255.0
        else:
            tensor = tensor.to(torch.float32)
        prepared[key] = tensor
    return prepared


def action_steps_for_ratio(chunk_size: int, ratio: float) -> int:
    """Convert a chunk execution ratio to a non-zero integer action count."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("chunk execution ratio must be in (0, 1]")
    return max(1, min(chunk_size, int(round(chunk_size * ratio))))


def _termination_reason(success: bool, terminated: bool, truncated: bool) -> str | None:
    if success:
        return "success"
    if terminated:
        return "terminated_failure"
    if truncated:
        return "time_limit"
    return None


def run_episode(
    env: Any,
    policy_runner: PolicyRunner,
    *,
    episode_index: int,
    seed: int,
    max_steps: int,
    frame_callback: Callable[[Mapping[str, Any], int], None] | None = None,
    video: str | None = None,
) -> EpisodeResult:
    """Run one episode and use ManiSkill's ``info['success']`` as truth."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    policy_runner.reset()
    observation, info = env.reset(seed=seed)
    if frame_callback is not None:
        frame_callback(observation, 0)

    initial_success = scalar_bool(info.get("success", False), name="info.success")
    if initial_success:
        return EpisodeResult(
            episode_index=episode_index,
            seed=seed,
            success=True,
            steps=0,
            termination="success",
            reward_sum=0.0,
            video=video,
        )

    reward_sum = 0.0
    for step in range(1, max_steps + 1):
        action = np.asarray(policy_runner(observation), dtype=np.float32)
        if action.ndim == 2 and action.shape[0] == 1:
            action = action[0]
        if action.shape != (EXPECTED_ACTION_DIM,):
            raise ValueError(f"Policy action must have shape ({EXPECTED_ACTION_DIM},), got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("Policy action contains NaN or infinity")

        observation, reward, terminated, truncated, info = env.step(action)
        reward_sum += scalar_float(reward, name="reward")
        if frame_callback is not None:
            frame_callback(observation, step)

        success = scalar_bool(info.get("success", False), name="info.success")
        terminated_flag = scalar_bool(terminated, name="terminated")
        truncated_flag = scalar_bool(truncated, name="truncated")
        reason = _termination_reason(success, terminated_flag, truncated_flag)
        if reason is not None:
            return EpisodeResult(
                episode_index=episode_index,
                seed=seed,
                success=success,
                steps=step,
                termination=reason,
                reward_sum=reward_sum,
                video=video,
            )

    return EpisodeResult(
        episode_index=episode_index,
        seed=seed,
        success=False,
        steps=max_steps,
        termination="max_steps",
        reward_sum=reward_sum,
        video=video,
    )


def run_episode_batch(
    env: Any,
    policy_runner: PolicyRunner,
    *,
    episode_indices: list[int],
    seeds: list[int],
    max_steps: int,
    frame_callbacks: list[Callable[[Mapping[str, Any], int, int], None] | None] | None = None,
    videos: list[str | None] | None = None,
) -> list[EpisodeResult]:
    """Run one synchronized ManiSkill batch and preserve each env's first terminal result.

    Finished environments remain in the simulator until the slowest member of
    the batch finishes, but their first success/termination is frozen and no
    later reward is accumulated.  This avoids unsafe partial policy-queue
    resets while still producing independent per-seed metrics.
    """
    batch_size = len(seeds)
    if batch_size == 0 or len(episode_indices) != batch_size:
        raise ValueError("episode_indices and seeds must have the same non-zero length")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if frame_callbacks is None:
        frame_callbacks = [None] * batch_size
    if videos is None:
        videos = [None] * batch_size
    if len(frame_callbacks) != batch_size or len(videos) != batch_size:
        raise ValueError("frame_callbacks and videos must match the batch size")

    policy_runner.reset()
    # Multi-env PegInsertion normally disables periodic reconfiguration.  An
    # explicit full reconfigure keeps seed-dependent peg/hole geometry aligned
    # with how the three-camera training dataset was rendered.
    observation, info = env.reset(seed=seeds, options={"reconfigure": True})
    for env_index, callback in enumerate(frame_callbacks):
        if callback is not None:
            callback(observation, 0, env_index)

    done = np.zeros(batch_size, dtype=bool)
    reward_sums = np.zeros(batch_size, dtype=np.float64)
    results: list[EpisodeResult | None] = [None] * batch_size
    initial_success = vector_bool(info.get("success", np.zeros(batch_size)), size=batch_size, name="info.success")
    for env_index in np.flatnonzero(initial_success):
        done[env_index] = True
        results[env_index] = EpisodeResult(
            episode_index=episode_indices[env_index],
            seed=seeds[env_index],
            success=True,
            steps=0,
            termination="success",
            reward_sum=0.0,
            video=videos[env_index],
        )

    for step in range(1, max_steps + 1):
        if done.all():
            break
        action = np.asarray(policy_runner(observation), dtype=np.float32)
        if action.ndim == 1 and batch_size == 1:
            action = action[None]
        if action.shape != (batch_size, EXPECTED_ACTION_DIM):
            raise ValueError(
                f"Batched policy action must have shape ({batch_size}, {EXPECTED_ACTION_DIM}), "
                f"got {action.shape}"
            )
        if not np.isfinite(action).all():
            raise ValueError("Policy action contains NaN or infinity")

        observation, reward, terminated, truncated, info = env.step(action)
        rewards = vector_float(reward, size=batch_size, name="reward")
        reward_sums[~done] += rewards[~done]
        for env_index, callback in enumerate(frame_callbacks):
            if callback is not None and not done[env_index]:
                callback(observation, step, env_index)

        successes = vector_bool(
            info.get("success", np.zeros(batch_size)), size=batch_size, name="info.success"
        )
        terminated_flags = vector_bool(terminated, size=batch_size, name="terminated")
        truncated_flags = vector_bool(truncated, size=batch_size, name="truncated")
        for env_index in np.flatnonzero(~done):
            reason = _termination_reason(
                bool(successes[env_index]),
                bool(terminated_flags[env_index]),
                bool(truncated_flags[env_index]),
            )
            if reason is None:
                continue
            done[env_index] = True
            results[env_index] = EpisodeResult(
                episode_index=episode_indices[env_index],
                seed=seeds[env_index],
                success=bool(successes[env_index]),
                steps=step,
                termination=reason,
                reward_sum=float(reward_sums[env_index]),
                video=videos[env_index],
            )

    for env_index in np.flatnonzero(~done):
        results[env_index] = EpisodeResult(
            episode_index=episode_indices[env_index],
            seed=seeds[env_index],
            success=False,
            steps=max_steps,
            termination="max_steps",
            reward_sum=float(reward_sums[env_index]),
            video=videos[env_index],
        )
    if any(result is None for result in results):
        raise RuntimeError("Internal error: missing result for a batched environment")
    return [result for result in results if result is not None]


def _wilson_interval(successes: int, episodes: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if episodes <= 0:
        return (0.0, 0.0)
    proportion = successes / episodes
    denominator = 1.0 + z * z / episodes
    center = (proportion + z * z / (2.0 * episodes)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / episodes + z * z / (4.0 * episodes * episodes)
        )
        / denominator
    )
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def summarize_results(results: list[EpisodeResult]) -> dict[str, Any]:
    if not results:
        raise ValueError("Cannot summarize zero episodes")
    successes = sum(result.success for result in results)
    success_steps = [result.steps for result in results if result.success]
    lower, upper = _wilson_interval(successes, len(results))
    termination_counts: dict[str, int] = {}
    for result in results:
        termination_counts[result.termination] = termination_counts.get(result.termination, 0) + 1
    return {
        "episodes": len(results),
        "successes": successes,
        "failures": len(results) - successes,
        "success_rate": successes / len(results),
        "success_rate_percent": 100.0 * successes / len(results),
        "success_rate_wilson_95": [lower, upper],
        "mean_episode_steps": mean(result.steps for result in results),
        "median_episode_steps": median(result.steps for result in results),
        "mean_success_steps": mean(success_steps) if success_steps else None,
        "termination_counts": termination_counts,
    }


def check_eval_inputs(checkpoint: str | Path, camera_config: str | Path) -> dict[str, Any]:
    """CPU-only evaluation preflight; reads metadata but imports no model/simulator."""
    checkpoint_path = Path(checkpoint)
    camera_config_path = Path(camera_config)
    errors: list[str] = []
    info: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "camera_config": str(camera_config_path),
    }

    required_checkpoint_files = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    )
    for filename in required_checkpoint_files:
        if not (checkpoint_path / filename).is_file():
            errors.append(f"missing checkpoint file: {checkpoint_path / filename}")

    policy_config: dict[str, Any] = {}
    if (checkpoint_path / "config.json").is_file():
        try:
            policy_config = json.loads((checkpoint_path / "config.json").read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read policy config: {exc}")
        else:
            info["policy_type"] = policy_config.get("type")
            info["chunk_size"] = policy_config.get("chunk_size")
            info["n_action_steps"] = policy_config.get("n_action_steps")
            if policy_config.get("type") != "smolvla":
                errors.append(f"policy type must be 'smolvla', got {policy_config.get('type')!r}")
            action_shape = ((policy_config.get("output_features") or {}).get("action") or {}).get("shape")
            if action_shape != [EXPECTED_ACTION_DIM]:
                errors.append(f"checkpoint action shape must be [{EXPECTED_ACTION_DIM}], got {action_shape}")

    if (checkpoint_path / "policy_preprocessor.json").is_file():
        try:
            preprocessor = json.loads((checkpoint_path / "policy_preprocessor.json").read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read policy preprocessor: {exc}")
        else:
            rename_steps = [
                step for step in preprocessor.get("steps", [])
                if step.get("registry_name") == "rename_observations_processor"
            ]
            rename_map = rename_steps[0].get("config", {}).get("rename_map") if rename_steps else None
            info["rename_map"] = rename_map
            if rename_map != SFT_IMAGE_RENAME_MAP:
                errors.append(f"checkpoint camera rename_map does not match training: {rename_map}")

    camera_specs: dict[str, Any] = {}
    if not camera_config_path.is_file():
        errors.append(f"missing camera config: {camera_config_path}")
    else:
        try:
            camera_specs = json.loads(camera_config_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read camera config: {exc}")
        else:
            if set(camera_specs) != set(CAMERA_NAMES):
                errors.append(
                    f"camera config must define {list(CAMERA_NAMES)}, got {sorted(camera_specs)}"
                )
            for camera_name in CAMERA_NAMES:
                spec = camera_specs.get(camera_name, {})
                if spec.get("width") != 512 or spec.get("height") != 512:
                    errors.append(f"{camera_name} must be 512x512")

    info["ok"] = not errors
    info["errors"] = errors
    if checkpoint_path.exists():
        info["resolved_checkpoint"] = str(checkpoint_path.resolve())
    return info


class SFTPolicyRunner:
    """Load SmolVLA and its saved processors for one-environment inference."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cuda",
        task: str = TASK_PROMPT,
        chunk_execution_ratio: float = DEFAULT_CHUNK_EXECUTION_RATIO,
    ):
        import torch
        from lerobot.policies.factory import make_pre_post_processors

        from smolvla_rltoken.vla.load import load_smolvla_policy

        self._torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA evaluation requested but torch.cuda.is_available() is false")
        self.task = task
        self.policy = load_smolvla_policy(str(checkpoint), device=str(self.device))
        self.chunk_size = int(self.policy.config.chunk_size)
        self.action_steps = action_steps_for_ratio(self.chunk_size, chunk_execution_ratio)
        self.chunk_execution_ratio = self.action_steps / self.chunk_size
        self.policy.config.n_action_steps = self.action_steps
        # from_pretrained() initializes a queue using the saved value (50).
        # Rebuild it after applying the evaluation-time closed-loop horizon.
        self.policy.reset()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

    def reset(self) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def __call__(self, observation: Mapping[str, Any]) -> np.ndarray:
        torch = self._torch
        batch = prepare_policy_observation_batch(observation, task=self.task)
        with torch.inference_mode():
            batch = self.preprocessor(batch)
            action = self.policy.select_action(batch)
            action = self.postprocessor(action)
        action_array = action.detach().cpu().numpy()
        if action_array.ndim == 2 and action_array.shape[0] == 1:
            action_array = action_array[0]
        return np.asarray(action_array, dtype=np.float32)


def make_maniskill_env(
    camera_specs: Mapping[str, Any],
    *,
    sim_backend: str = "physx_cuda",
    max_episode_steps: int = 200,
    num_envs: int = 1,
) -> Any:
    """Create the project environment lazily for a real evaluation run."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    import smolvla_rltoken.envs.maniskill_env  # noqa: F401

    return gym.make(
        "PegInsertionSideThreeCamera-v1",
        num_envs=num_envs,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        reward_mode="dense",
        sim_backend=sim_backend,
        camera_specs=dict(camera_specs),
        max_episode_steps=max_episode_steps,
    )
