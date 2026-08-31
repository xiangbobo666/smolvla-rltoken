"""Stage 0 SmolVLA SFT: config, local checks, and ``lerobot-train`` argv.

This is not a custom trainer. The official LeRobot loop does the gradient
updates; this module only prepares a PegInsertion-specific launch (cameras,
paths, freeze flags) and can validate the dataset on CPU without loading the VLA.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from smolvla_rltoken.paths import (
    DATASET_REPO_ID,
    DATASET_ROOT,
    HF_HOME,
    SFT_IMAGE_RENAME_MAP,
    SFT_OUTPUT_DIR,
    SFT_SMOKE_OUTPUT_DIR,
    SMOLVLA_BASE,
    WANDB_PROJECT,
)

EXPECTED_EPISODES = 1000
EXPECTED_FRAMES = 149_055
EXPECTED_IMAGE_HW = (512, 512)
EXPECTED_ACTION_DIM = 8
EXPECTED_STATE_DIM = 9
DATASET_CAMERAS = (
    "environment_camera",
    "hand_camera",
    "insertion_camera",
)
POLICY_CAMERAS = ("camera1", "camera2", "camera3")
CHECKPOINT_TRAIN_CONFIG = Path("checkpoints") / "last" / "pretrained_model" / "train_config.json"
SFT_SMOKE_JOB_NAME = "smolvla_sft_peg_insertion_smoke"
SFT_SMOKE_STEPS = 2
SFT_SMOKE_BATCH_SIZE = 2
SFT_SMOKE_LOG_FREQ = 1
SFT_SMOKE_SAVE_FREQ = 2
SFT_SMOKE_NUM_WORKERS = 2


@dataclass
class SFTConfig:
    """Launch settings for PegInsertion SmolVLA SFT (consumed by ``train_sft.py``)."""

    policy_path: str = str(SMOLVLA_BASE)
    dataset_repo_id: str = DATASET_REPO_ID
    dataset_root: str = str(DATASET_ROOT)
    output_dir: str = str(SFT_OUTPUT_DIR)
    job_name: str = "smolvla_sft_peg_insertion"

    batch_size: int = 32
    steps: int = 20_000
    save_freq: int = 5_000
    log_freq: int = 200
    num_workers: int = 16
    eval_freq: int = 0
    seed: int = 1000

    device: str = "cuda"
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True
    push_to_hub: bool = False
    wandb_enable: bool = True
    wandb_project: str = WANDB_PROJECT
    wandb_disable_artifact: bool = False
    video_backend: str = "torchcodec"

    rename_map: dict[str, str] = field(default_factory=lambda: dict(SFT_IMAGE_RENAME_MAP))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SFTConfig:
        known = {f.name for f in fields(cls)}
        payload = {k: v for k, v in data.items() if k in known}
        if "rename_map" in payload and payload["rename_map"] is not None:
            payload["rename_map"] = {str(k): str(v) for k, v in payload["rename_map"].items()}
        return cls(**payload)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SFTConfig:
        import yaml

        with Path(path).open() as f:
            payload = yaml.safe_load(f) or {}
        return cls.from_dict(payload)


@dataclass
class SFTCheckResult:
    ok: bool
    errors: list[str]
    warnings: list[str]
    info: dict[str, Any]

    def raise_if_failed(self) -> None:
        if self.ok:
            return
        details = "\n".join(f"  - {item}" for item in self.errors)
        raise FileNotFoundError(f"SFT preflight failed:\n{details}")


def load_dataset_info(dataset_root: str | Path) -> dict[str, Any]:
    path = Path(dataset_root) / "meta" / "info.json"
    with path.open() as f:
        return json.load(f)


def load_policy_config(policy_path: str | Path) -> dict[str, Any]:
    path = Path(policy_path) / "config.json"
    with path.open() as f:
        return json.load(f)


def _bool_cli(value: bool) -> str:
    return "true" if value else "false"


def resume_config_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / CHECKPOINT_TRAIN_CONFIG


def apply_smoke_overrides(cfg: SFTConfig) -> SFTConfig:
    """Short GPU loop that does not occupy the formal SFT output directory."""
    cfg.steps = SFT_SMOKE_STEPS
    cfg.log_freq = SFT_SMOKE_LOG_FREQ
    cfg.batch_size = SFT_SMOKE_BATCH_SIZE
    cfg.save_freq = SFT_SMOKE_SAVE_FREQ
    cfg.num_workers = SFT_SMOKE_NUM_WORKERS
    cfg.output_dir = str(SFT_SMOKE_OUTPUT_DIR)
    cfg.job_name = SFT_SMOKE_JOB_NAME
    cfg.wandb_disable_artifact = True
    return cfg


def is_throwaway_smoke_output(output_dir: str | Path) -> bool:
    return Path(output_dir).resolve() == SFT_SMOKE_OUTPUT_DIR.resolve()


def build_lerobot_train_argv(
    cfg: SFTConfig,
    extra: list[str] | None = None,
    *,
    resume: bool = False,
) -> list[str]:
    """Build ``lerobot-train`` argv. ``extra`` is appended so it overrides yaml values."""
    rename = json.dumps(cfg.rename_map, separators=(",", ":"))
    argv = [
        "lerobot-train",
        f"--policy.path={cfg.policy_path}",
        f"--policy.push_to_hub={_bool_cli(cfg.push_to_hub)}",
        f"--policy.freeze_vision_encoder={_bool_cli(cfg.freeze_vision_encoder)}",
        f"--policy.train_expert_only={_bool_cli(cfg.train_expert_only)}",
        f"--policy.train_state_proj={_bool_cli(cfg.train_state_proj)}",
        f"--policy.device={cfg.device}",
        f"--dataset.repo_id={cfg.dataset_repo_id}",
        f"--dataset.root={cfg.dataset_root}",
        f"--dataset.video_backend={cfg.video_backend}",
        f"--output_dir={cfg.output_dir}",
        f"--batch_size={cfg.batch_size}",
        f"--steps={cfg.steps}",
        f"--save_freq={cfg.save_freq}",
        f"--log_freq={cfg.log_freq}",
        f"--num_workers={cfg.num_workers}",
        f"--eval_freq={cfg.eval_freq}",
        f"--seed={cfg.seed}",
        f"--job_name={cfg.job_name}",
        f"--wandb.enable={_bool_cli(cfg.wandb_enable)}",
        f"--wandb.project={cfg.wandb_project}",
        f"--wandb.disable_artifact={_bool_cli(cfg.wandb_disable_artifact)}",
        f"--rename_map={rename}",
    ]
    if resume:
        argv.extend(
            [
                "--resume=true",
                f"--config_path={resume_config_path(cfg.output_dir)}",
            ]
        )
    if extra:
        argv.extend(extra)
    return argv


def _image_feature_keys(info: dict[str, Any]) -> list[str]:
    keys = []
    for name, spec in (info.get("features") or {}).items():
        if not name.startswith("observation.images."):
            continue
        if spec.get("dtype") in {"video", "image"}:
            keys.append(name)
    return sorted(keys)


def _feature_shape(info: dict[str, Any], key: str) -> list[int] | None:
    spec = (info.get("features") or {}).get(key)
    if not spec:
        return None
    return list(spec.get("shape") or [])


def _count_mp4(camera_dir: Path) -> int:
    if not camera_dir.is_dir():
        return 0
    return sum(1 for _ in camera_dir.rglob("*.mp4"))


def _tokenizer_cached() -> bool:
    hub = HF_HOME / "hub"
    if not hub.is_dir():
        return False
    return any(hub.glob("models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct*"))


def torchcodec_loads() -> bool:
    """True when the native torchcodec decoder can be imported (not just the wheel)."""
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401
    except Exception:
        return False
    return True


def check_sft(cfg: SFTConfig, *, resume: bool = False) -> SFTCheckResult:
    """CPU-only preflight: JSON + paths. Does not load SmolVLA or decode videos."""
    errors: list[str] = []
    warnings: list[str] = []
    info: dict[str, Any] = {"dataset_root": cfg.dataset_root, "policy_path": cfg.policy_path}

    if cfg.rename_map != SFT_IMAGE_RENAME_MAP:
        errors.append(
            "rename_map must map environment/hand/insertion cameras to "
            f"camera1/2/3; got {cfg.rename_map}"
        )

    expected_src = {f"observation.images.{name}" for name in DATASET_CAMERAS}
    expected_dst = {f"observation.images.{name}" for name in POLICY_CAMERAS}
    if set(cfg.rename_map) != expected_src:
        errors.append(f"rename_map keys {sorted(cfg.rename_map)} != {sorted(expected_src)}")
    if set(cfg.rename_map.values()) != expected_dst:
        errors.append(f"rename_map values {sorted(cfg.rename_map.values())} != {sorted(expected_dst)}")

    if not cfg.freeze_vision_encoder or not cfg.train_expert_only:
        warnings.append(
            "SmolVLA SFT default is freeze_vision_encoder=true and train_expert_only=true"
        )

    info["video_backend"] = cfg.video_backend
    if cfg.video_backend == "torchcodec" and not torchcodec_loads():
        errors.append(
            "video_backend=torchcodec but the native decoder failed to load. "
            "Use torchcodec==0.7.* with torch 2.8, or set video_backend: pyav"
        )
    elif cfg.video_backend == "pyav":
        warnings.append(
            "video_backend=pyav is LeRobot's fallback decoder; prefer torchcodec when it loads"
        )

    root = Path(cfg.dataset_root)
    if not root.is_dir():
        errors.append(f"dataset root missing: {root}")
        return SFTCheckResult(ok=False, errors=errors, warnings=warnings, info=info)

    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        errors.append(f"dataset meta missing: {info_path}")
        return SFTCheckResult(ok=False, errors=errors, warnings=warnings, info=info)

    try:
        ds_info = load_dataset_info(root)
    except json.JSONDecodeError as exc:
        errors.append(f"cannot parse dataset info.json: {exc}")
        return SFTCheckResult(ok=False, errors=errors, warnings=warnings, info=info)

    episodes = ds_info.get("total_episodes")
    frames = ds_info.get("total_frames")
    info["episodes"] = episodes
    info["frames"] = frames
    if episodes != EXPECTED_EPISODES:
        errors.append(f"expected {EXPECTED_EPISODES} episodes, dataset has {episodes}")
    if frames != EXPECTED_FRAMES:
        errors.append(f"expected {EXPECTED_FRAMES} frames, dataset has {frames}")

    image_keys = _image_feature_keys(ds_info)
    info["image_keys"] = image_keys
    missing_cams = [f"observation.images.{c}" for c in DATASET_CAMERAS if f"observation.images.{c}" not in image_keys]
    extra_cams = [k for k in image_keys if k not in expected_src]
    if missing_cams:
        errors.append(f"dataset missing cameras: {missing_cams}")
    if extra_cams:
        errors.append(f"dataset has unexpected cameras: {extra_cams}")

    for cam in DATASET_CAMERAS:
        key = f"observation.images.{cam}"
        shape = _feature_shape(ds_info, key)
        if shape is None:
            continue
        h, w = shape[0], shape[1]
        info[f"{cam}_hw"] = (h, w)
        if (h, w) != EXPECTED_IMAGE_HW:
            errors.append(f"{key} shape {shape} is not {EXPECTED_IMAGE_HW[0]}x{EXPECTED_IMAGE_HW[1]} RGB")

    action_shape = _feature_shape(ds_info, "action")
    state_shape = _feature_shape(ds_info, "observation.state")
    info["action_dim"] = action_shape[-1] if action_shape else None
    info["state_dim"] = state_shape[-1] if state_shape else None
    if not action_shape or action_shape[-1] != EXPECTED_ACTION_DIM:
        errors.append(f"action dim {action_shape} != {EXPECTED_ACTION_DIM} (pd_joint_pos)")
    if not state_shape or state_shape[-1] != EXPECTED_STATE_DIM:
        errors.append(f"observation.state dim {state_shape} != {EXPECTED_STATE_DIM}")

    parquet = root / "data" / "chunk-000" / "file-000.parquet"
    if not parquet.is_file():
        errors.append(f"missing parquet: {parquet}")

    video_counts: dict[str, int] = {}
    for cam in DATASET_CAMERAS:
        cam_dir = root / "videos" / f"observation.images.{cam}"
        n = _count_mp4(cam_dir)
        video_counts[cam] = n
        if n != EXPECTED_EPISODES:
            errors.append(f"{cam}: expected {EXPECTED_EPISODES} mp4 files, found {n} in {cam_dir}")
    info["video_counts"] = video_counts

    policy_dir = Path(cfg.policy_path)
    weights = policy_dir / "model.safetensors"
    policy_cfg_path = policy_dir / "config.json"
    if not policy_dir.is_dir():
        errors.append(f"SmolVLA checkpoint missing: {policy_dir}")
    else:
        if not weights.is_file():
            errors.append(f"missing weights: {weights}")
        if not policy_cfg_path.is_file():
            errors.append(f"missing policy config: {policy_cfg_path}")
        else:
            policy_cfg = load_policy_config(policy_dir)
            if policy_cfg.get("type") != "smolvla":
                errors.append(f"policy type is {policy_cfg.get('type')!r}, expected 'smolvla'")
            input_features = policy_cfg.get("input_features") or {}
            for cam in POLICY_CAMERAS:
                key = f"observation.images.{cam}"
                if key not in input_features:
                    errors.append(f"smolvla_base is missing pretrained key {key}")
            resize = tuple(policy_cfg.get("resize_imgs_with_padding") or ())
            if resize != EXPECTED_IMAGE_HW:
                warnings.append(
                    f"policy resize_imgs_with_padding={resize}, dataset is {EXPECTED_IMAGE_HW}"
                )
            info["policy_type"] = policy_cfg.get("type")
            info["policy_resize"] = resize

    out = Path(cfg.output_dir)
    info["output_dir"] = str(out)
    if resume:
        ckpt = resume_config_path(out)
        if not ckpt.is_file():
            errors.append(f"resume requested but missing {ckpt}")
    elif out.exists():
        errors.append(
            f"output_dir already exists: {out}. LeRobot refuses to overwrite; "
            "pass --resume or choose a new --output-dir"
        )

    if not _tokenizer_cached():
        warnings.append(
            "SmolVLM2 tokenizer is not in HF_HOME yet; the first GPU run will download "
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct via the project HTTP proxy"
        )

    return SFTCheckResult(ok=not errors, errors=errors, warnings=warnings, info=info)
