"""Stage 1 RL Token training: config, CPU preflight, and reconstruction loop.

VLA stays frozen. Each step runs an online prefix forward (no disk embedding
cache). YAML is the source of truth; CLI flags override only when passed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from smolvla_rltoken.paths import (
    DATASET_REPO_ID,
    DATASET_ROOT,
    RL_TOKEN_OUTPUT_DIR,
    RL_TOKEN_SMOKE_OUTPUT_DIR,
    SFT_IMAGE_RENAME_MAP,
    SFT_LAST_PRETRAINED,
    SFT_OUTPUT_DIR,
    SMOLVLA_BASE,
    WANDB_PROJECT,
)
from smolvla_rltoken.rlt.config import RLTokenConfig
from smolvla_rltoken.vla.sft import (
    POLICY_CAMERAS,
    SFTConfig,
    check_sft,
    torchcodec_loads,
)

RL_TOKEN_SMOKE_JOB_NAME = "smolvla_rltoken_stage1_smoke"
RL_TOKEN_SMOKE_STEPS = 2
RL_TOKEN_SMOKE_BATCH_SIZE = 2
RL_TOKEN_SMOKE_LOG_FREQ = 1
RL_TOKEN_SMOKE_SAVE_FREQ = 2
RL_TOKEN_SMOKE_NUM_WORKERS = 2
CHECKPOINT_WEIGHTS = "model.safetensors"
CHECKPOINT_CONFIG = "config.json"


@dataclass
class RLTokenTrainConfig:
    """Launch + architecture settings for Stage 1 (consumed by ``train_rltoken.py``)."""

    checkpoint: str = str(SFT_LAST_PRETRAINED)
    dataset_repo_id: str = DATASET_REPO_ID
    dataset_root: str = str(DATASET_ROOT)
    output_dir: str = str(RL_TOKEN_OUTPUT_DIR)
    job_name: str = "smolvla_rltoken_stage1"

    vla_width: int = 960
    d_model: int = 512
    n_heads: int = 8
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    use_image_tokens_only: bool = True
    max_recon_tokens: int = 256
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    steps: int = 5000
    batch_size: int = 16

    num_workers: int = 16
    log_freq: int = 20
    save_freq: int = 500
    seed: int = 1000
    device: str = "cuda"
    dtype: str | None = None
    video_backend: str = "torchcodec"

    wandb_enable: bool = True
    wandb_project: str = WANDB_PROJECT
    wandb_disable_artifact: bool = False

    rename_map: dict[str, str] = field(default_factory=lambda: dict(SFT_IMAGE_RENAME_MAP))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_module_config(self) -> RLTokenConfig:
        return RLTokenConfig(
            vla_width=self.vla_width,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_encoder_layers=self.n_encoder_layers,
            n_decoder_layers=self.n_decoder_layers,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            use_image_tokens_only=self.use_image_tokens_only,
            max_recon_tokens=self.max_recon_tokens,
            lr=self.lr,
            weight_decay=self.weight_decay,
            grad_clip_norm=self.grad_clip_norm,
            steps=self.steps,
            batch_size=self.batch_size,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RLTokenTrainConfig:
        known = {f.name for f in fields(cls)}
        payload = {k: v for k, v in data.items() if k in known}
        if "rename_map" in payload and payload["rename_map"] is not None:
            payload["rename_map"] = {str(k): str(v) for k, v in payload["rename_map"].items()}
        if "dtype" in payload and payload["dtype"] in {"", "null", "None"}:
            payload["dtype"] = None
        return cls(**payload)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RLTokenTrainConfig:
        import yaml

        with Path(path).open() as f:
            payload = yaml.safe_load(f) or {}
        return cls.from_dict(payload)


@dataclass
class RLTokenCheckResult:
    ok: bool
    errors: list[str]
    warnings: list[str]
    info: dict[str, Any]

    def raise_if_failed(self) -> None:
        if self.ok:
            return
        details = "\n".join(f"  - {item}" for item in self.errors)
        raise FileNotFoundError(f"Stage 1 preflight failed:\n{details}")


def apply_smoke_overrides(cfg: RLTokenTrainConfig) -> RLTokenTrainConfig:
    """Short GPU loop that does not occupy the formal Stage 1 output directory."""
    cfg.steps = RL_TOKEN_SMOKE_STEPS
    cfg.log_freq = RL_TOKEN_SMOKE_LOG_FREQ
    cfg.batch_size = RL_TOKEN_SMOKE_BATCH_SIZE
    cfg.save_freq = RL_TOKEN_SMOKE_SAVE_FREQ
    cfg.num_workers = RL_TOKEN_SMOKE_NUM_WORKERS
    cfg.output_dir = str(RL_TOKEN_SMOKE_OUTPUT_DIR)
    cfg.job_name = RL_TOKEN_SMOKE_JOB_NAME
    cfg.wandb_disable_artifact = True
    return cfg


def is_throwaway_smoke_output(output_dir: str | Path) -> bool:
    return Path(output_dir).resolve() == RL_TOKEN_SMOKE_OUTPUT_DIR.resolve()


def apply_cli_overrides(cfg: RLTokenTrainConfig, args: Any) -> RLTokenTrainConfig:
    """Apply argparse overrides. ``None`` means keep the YAML value."""
    if getattr(args, "smoke", False):
        apply_smoke_overrides(cfg)
    if getattr(args, "checkpoint", None) is not None:
        cfg.checkpoint = args.checkpoint
    if getattr(args, "output_dir", None) is not None:
        cfg.output_dir = args.output_dir
    if getattr(args, "batch_size", None) is not None:
        cfg.batch_size = args.batch_size
    if getattr(args, "steps", None) is not None:
        cfg.steps = args.steps
    if getattr(args, "num_workers", None) is not None:
        cfg.num_workers = args.num_workers
    if getattr(args, "device", None) is not None:
        cfg.device = args.device
    if getattr(args, "dtype", None) is not None:
        cfg.dtype = args.dtype
    if getattr(args, "all_prefix_tokens", False):
        cfg.use_image_tokens_only = False
    return cfg


def _sft_preflight_cfg(cfg: RLTokenTrainConfig) -> SFTConfig:
    """Reuse SFT dataset/camera JSON checks without LeRobot overwrite rules."""
    dummy_out = str(Path(cfg.output_dir).resolve().parent / "_rltoken_preflight_unused")
    return SFTConfig(
        policy_path=cfg.checkpoint,
        dataset_repo_id=cfg.dataset_repo_id,
        dataset_root=cfg.dataset_root,
        output_dir=dummy_out,
        rename_map=dict(cfg.rename_map),
        video_backend=cfg.video_backend,
        wandb_project=cfg.wandb_project,
    )


def check_rl_token(cfg: RLTokenTrainConfig) -> RLTokenCheckResult:
    """CPU-only preflight: JSON + paths. Does not load SmolVLA or decode videos."""
    errors: list[str] = []
    warnings: list[str] = []
    ckpt = Path(cfg.checkpoint)
    info: dict[str, Any] = {
        "checkpoint": str(ckpt),
        "dataset_root": cfg.dataset_root,
        "output_dir": cfg.output_dir,
        "sft_last_pretrained": str(SFT_LAST_PRETRAINED),
        "smolvla_base": str(SMOLVLA_BASE),
    }

    sft_result = check_sft(_sft_preflight_cfg(cfg), resume=False)
    errors.extend(sft_result.errors)
    warnings.extend(sft_result.warnings)
    info.update(sft_result.info)
    info["checkpoint"] = str(ckpt)
    info["output_dir"] = cfg.output_dir

    if cfg.rename_map != SFT_IMAGE_RENAME_MAP:
        errors.append(
            "rename_map must match SFT (environment/hand/insertion → camera1/2/3); "
            f"got {cfg.rename_map}"
        )

    if ckpt.resolve() == SFT_OUTPUT_DIR.resolve():
        errors.append(
            f"checkpoint is the SFT output directory {ckpt}. Pass "
            f"{SFT_LAST_PRETRAINED} (LeRobot pretrained_model), not the run root."
        )

    last = SFT_LAST_PRETRAINED
    if ckpt.resolve() == last.resolve() and not (last / CHECKPOINT_WEIGHTS).is_file():
        errors.append(
            f"SFT last checkpoint is missing: {last}. Wait for SFT to save, or pass "
            f"--checkpoint {SMOLVLA_BASE} to check the loop against smolvla_base."
        )
    elif not ckpt.is_dir():
        errors.append(
            f"checkpoint directory missing: {ckpt}. Formal Stage 1 uses {last}; "
            f"loop smoke can use {SMOLVLA_BASE}."
        )
    else:
        if not (ckpt / CHECKPOINT_WEIGHTS).is_file():
            errors.append(f"missing weights: {ckpt / CHECKPOINT_WEIGHTS}")
        cfg_path = ckpt / CHECKPOINT_CONFIG
        if not cfg_path.is_file():
            errors.append(f"missing policy config: {cfg_path}")
        else:
            try:
                with cfg_path.open() as f:
                    policy_cfg = json.load(f)
            except json.JSONDecodeError as exc:
                errors.append(f"cannot parse checkpoint config.json: {exc}")
            else:
                input_features = policy_cfg.get("input_features") or {}
                for cam in POLICY_CAMERAS:
                    key = f"observation.images.{cam}"
                    if key not in input_features:
                        errors.append(f"checkpoint is missing pretrained key {key}")
                info["policy_type"] = policy_cfg.get("type")

    info["video_backend"] = cfg.video_backend
    if cfg.video_backend == "torchcodec" and not torchcodec_loads():
        if not any("torchcodec" in e for e in errors):
            errors.append(
                "video_backend=torchcodec but the native decoder failed to load. "
                "Use torchcodec==0.7.* with torch 2.8, or set video_backend: pyav"
            )

    info["use_image_tokens_only"] = cfg.use_image_tokens_only
    info["steps"] = cfg.steps
    info["batch_size"] = cfg.batch_size
    return RLTokenCheckResult(ok=not errors, errors=errors, warnings=warnings, info=info)


def _move_batch_tensors(batch: dict, device: str) -> dict:
    import torch

    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def _init_wandb(cfg: RLTokenTrainConfig):
    if not cfg.wandb_enable:
        return None
    import wandb

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=cfg.wandb_project,
        name=cfg.job_name,
        config=cfg.to_dict(),
        dir=str(cfg.output_dir),
    )


def train_rl_token(cfg: RLTokenTrainConfig) -> None:
    """Frozen-VLA reconstruction loop. Caller must have passed CPU preflight."""
    import time

    import torch

    from smolvla_rltoken.rlt.module import RLTokenModule
    from smolvla_rltoken.vla.dataset import build_dataset_and_processors
    from smolvla_rltoken.vla.extractor import SmolVLAPrefixExtractor
    from smolvla_rltoken.vla.load import load_smolvla_policy

    torch.manual_seed(cfg.seed)
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    module_cfg = cfg.to_module_config()
    print(f"[stage1] loading SmolVLA from {cfg.checkpoint} ...", flush=True)
    policy = load_smolvla_policy(cfg.checkpoint, device=cfg.device, dtype=cfg.dtype)
    dataset, preprocessor, _ = build_dataset_and_processors(
        policy,
        cfg.dataset_repo_id,
        cfg.dataset_root,
        rename_map=cfg.rename_map,
        video_backend=cfg.video_backend,
    )
    extractor = SmolVLAPrefixExtractor(policy)
    policy.requires_grad_(False)
    policy.eval()

    rl_token = RLTokenModule(module_cfg).to(cfg.device)
    n_params = sum(p.numel() for p in rl_token.parameters())
    print(
        f"[stage1] RL token module: {n_params / 1e6:.1f}M params, d_model={module_cfg.d_model}",
        flush=True,
    )

    params = list(rl_token.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.startswith("cuda"),
        drop_last=True,
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = _init_wandb(cfg)

    step, t0 = 0, time.time()
    data_iter = iter(loader)
    while step < cfg.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        batch = preprocessor(batch)
        batch = _move_batch_tensors(batch, cfg.device)

        feats = extractor.extract(batch)
        z, mask = extractor.select_tokens(feats, cfg.use_image_tokens_only)
        loss, _ = rl_token.reconstruction_loss(z, mask)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
        opt.step()
        step += 1

        if step % cfg.log_freq == 0 or step == 1:
            speed = step / (time.time() - t0)
            print(
                f"[stage1] step {step}/{cfg.steps} loss_ro={loss.item():.5f} "
                f"z_shape={tuple(z.shape)} grad_norm={float(grad_norm):.4f} "
                f"({speed:.2f} it/s)",
                flush=True,
            )
            if run is not None:
                run.log(
                    {
                        "loss_ro": loss.item(),
                        "grad_norm": float(grad_norm),
                        "it_s": speed,
                        "lr": cfg.lr,
                    },
                    step=step,
                )

        if step % cfg.save_freq == 0 or step == cfg.steps:
            ckpt = {
                "rl_token": rl_token.state_dict(),
                "config": module_cfg.to_dict(),
                "step": step,
            }
            torch.save(ckpt, out_dir / "rl_token.pt")
            print(f"[stage1] saved {out_dir / 'rl_token.pt'}", flush=True)

    if run is not None:
        run.finish()
    print(f"[stage1] done; saved to {out_dir / 'rl_token.pt'}", flush=True)
