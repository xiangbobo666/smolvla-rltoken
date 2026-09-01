"""Stage 1 RL Token training: config, CPU preflight, and reconstruction loop.

VLA stays frozen. Each step runs an online prefix forward (no disk embedding
cache). YAML is the source of truth; CLI flags override only when passed.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from smolvla_rltoken.paths import (
    DATASET_REPO_ID,
    DATASET_ROOT,
    RL_TOKEN_OUTPUT_DIR,
    RL_TOKEN_PRESSURE_OUTPUT_DIR,
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
RL_TOKEN_SMOKE_VAL_FREQ = 1
RL_TOKEN_SMOKE_MAX_VAL_BATCHES = 1
RL_TOKEN_PRESSURE_JOB_NAME = "smolvla_rltoken_stage1_pressure"
RL_TOKEN_PRESSURE_STEPS = 8
RL_TOKEN_PRESSURE_NUM_WORKERS = 4
RL_TOKEN_PRESSURE_BATCH_SIZES = (16, 32, 40, 48, 56, 64)
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
    batch_size: int = 32

    num_workers: int = 4
    log_freq: int = 20
    save_freq: int = 500
    val_ratio: float = 0.1
    val_freq: int = 100
    max_val_batches: int | None = 16
    val_num_workers: int = 0
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
        if "max_val_batches" in payload and payload["max_val_batches"] in {"", "null", "None"}:
            payload["max_val_batches"] = None
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
    cfg.val_freq = RL_TOKEN_SMOKE_VAL_FREQ
    cfg.max_val_batches = RL_TOKEN_SMOKE_MAX_VAL_BATCHES
    cfg.val_num_workers = 0
    cfg.output_dir = str(RL_TOKEN_SMOKE_OUTPUT_DIR)
    cfg.job_name = RL_TOKEN_SMOKE_JOB_NAME
    cfg.wandb_disable_artifact = True
    return cfg


def apply_pressure_overrides(cfg: RLTokenTrainConfig) -> RLTokenTrainConfig:
    """Batch-size GPU sweep that does not occupy the formal Stage 1 output directory."""
    cfg.steps = RL_TOKEN_PRESSURE_STEPS
    cfg.log_freq = 1
    cfg.save_freq = RL_TOKEN_PRESSURE_STEPS
    cfg.num_workers = RL_TOKEN_PRESSURE_NUM_WORKERS
    cfg.val_ratio = 0.0
    cfg.val_freq = 0
    cfg.max_val_batches = 0
    cfg.output_dir = str(RL_TOKEN_PRESSURE_OUTPUT_DIR)
    cfg.job_name = RL_TOKEN_PRESSURE_JOB_NAME
    cfg.wandb_enable = False
    cfg.wandb_disable_artifact = True
    return cfg


def is_throwaway_smoke_output(output_dir: str | Path) -> bool:
    return Path(output_dir).resolve() == RL_TOKEN_SMOKE_OUTPUT_DIR.resolve()


def is_throwaway_pressure_output(output_dir: str | Path) -> bool:
    return Path(output_dir).resolve() == RL_TOKEN_PRESSURE_OUTPUT_DIR.resolve()


def is_throwaway_output(output_dir: str | Path) -> bool:
    return is_throwaway_smoke_output(output_dir) or is_throwaway_pressure_output(output_dir)


def is_formal_output_root(output_dir: str | Path, *, root: str | Path | None = None) -> bool:
    """True when the path is the Stage 1 parent, not a per-run subdirectory."""
    parent = Path(root) if root is not None else RL_TOKEN_OUTPUT_DIR
    return Path(output_dir).resolve() == parent.resolve()


def new_run_directory(root: str | Path, *, when: datetime | None = None) -> Path:
    """Create ``run_YYYYMMDD_HHMMSS`` under ``root``. Does not overwrite existing dirs."""
    stamp = (when or datetime.now(UTC)).strftime("%Y%m%d_%H%M%S")
    parent = Path(root)
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / f"run_{stamp}"
    if path.exists():
        for index in range(2, 100):
            candidate = parent / f"run_{stamp}_{index}"
            if not candidate.exists():
                path = candidate
                break
        else:
            raise FileExistsError(f"could not allocate a free run directory under {parent}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def allocate_formal_run_dir(
    cfg: RLTokenTrainConfig,
    *,
    when: datetime | None = None,
    root: str | Path | None = None,
) -> RLTokenTrainConfig:
    """Write each formal train into ``<root>/run_<utc>/``, not the parent."""
    parent = Path(root) if root is not None else RL_TOKEN_OUTPUT_DIR
    if is_throwaway_output(cfg.output_dir) or not is_formal_output_root(cfg.output_dir, root=parent):
        return cfg
    run_dir = new_run_directory(cfg.output_dir, when=when)
    stamp = run_dir.name.removeprefix("run_")
    cfg.output_dir = str(run_dir)
    if stamp not in cfg.job_name:
        cfg.job_name = f"{cfg.job_name}_{stamp}"
    return cfg


def apply_cli_overrides(cfg: RLTokenTrainConfig, args: Any) -> RLTokenTrainConfig:
    """Apply argparse overrides. ``None`` means keep the YAML value."""
    if getattr(args, "pressure", False):
        apply_pressure_overrides(cfg)
    elif getattr(args, "smoke", False):
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
    if getattr(args, "val_freq", None) is not None:
        cfg.val_freq = args.val_freq
    if getattr(args, "val_ratio", None) is not None:
        cfg.val_ratio = args.val_ratio
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

    if cfg.val_ratio < 0.0 or cfg.val_ratio >= 1.0:
        errors.append(f"val_ratio must be in [0, 1); got {cfg.val_ratio}")
    if cfg.val_freq < 0:
        errors.append(f"val_freq must be >= 0; got {cfg.val_freq}")
    if cfg.max_val_batches is not None and cfg.max_val_batches < 0:
        errors.append(f"max_val_batches must be >= 0 (0 = full val); got {cfg.max_val_batches}")
    if cfg.val_num_workers < 0:
        errors.append(f"val_num_workers must be >= 0; got {cfg.val_num_workers}")

    n_episodes = info.get("episodes")
    if isinstance(n_episodes, int) and n_episodes > 0 and 0.0 <= cfg.val_ratio < 1.0:
        train_ids, val_ids = split_episode_indices(n_episodes, cfg.val_ratio, cfg.seed)
        info["train_episodes"] = len(train_ids)
        info["val_episodes"] = len(val_ids)
        if cfg.val_ratio > 0.0 and cfg.val_freq <= 0:
            warnings.append(
                f"val_ratio={cfg.val_ratio} holds out {len(val_ids)} episodes but val_freq=0, "
                "so val_loss_ro will not be computed"
            )
    info["val_ratio"] = cfg.val_ratio
    info["val_freq"] = cfg.val_freq
    info["max_val_batches"] = cfg.max_val_batches
    info["output_root"] = str(RL_TOKEN_OUTPUT_DIR)
    if is_formal_output_root(cfg.output_dir):
        info["output_dir_note"] = (
            "formal train creates a new outputs/rl_token/run_YYYYMMDD_HHMMSS directory; "
            "it does not overwrite files in the parent"
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


def masked_token_rms(z, mask) -> float:
    """RMS of selected VLA tokens, averaged over the feature dim then valid tokens."""
    weight = mask.float()
    per_token = z.detach().float().pow(2).mean(dim=-1)
    mean_sq = (per_token * weight).sum() / weight.sum().clamp(min=1.0)
    return float(mean_sq.sqrt().item())


def split_episode_indices(
    n_episodes: int, val_ratio: float, seed: int
) -> tuple[list[int], list[int]]:
    """Hold out a disjoint episode subset. Frame-level splits leak adjacent observations."""
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1; got {n_episodes}")
    if val_ratio < 0.0 or val_ratio >= 1.0:
        raise ValueError(f"val_ratio must be in [0, 1); got {val_ratio}")
    indices = list(range(n_episodes))
    if val_ratio == 0.0:
        return indices, []
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = min(n_episodes - 1, max(1, int(round(n_episodes * val_ratio))))
    val_ids = sorted(indices[:n_val])
    train_ids = sorted(indices[n_val:])
    return train_ids, val_ids


def make_stage1_loader(
    dataset,
    cfg: RLTokenTrainConfig,
    *,
    shuffle: bool = True,
    drop_last: bool | None = None,
    num_workers: int | None = None,
):
    import torch

    if drop_last is None:
        drop_last = shuffle
    workers = cfg.num_workers if num_workers is None else num_workers
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=cfg.device.startswith("cuda"),
        drop_last=drop_last,
    )


def reconstruction_step(extractor, rl_token, batch, cfg: RLTokenTrainConfig):
    """Frozen-VLA prefix extract + masked reconstruction loss on one batch."""
    feats = extractor.extract(batch)
    z, mask = extractor.select_tokens(feats, cfg.use_image_tokens_only)
    loss, _ = rl_token.reconstruction_loss(z, mask)
    return loss, z, mask


def evaluate_reconstruction(
    extractor,
    rl_token,
    loader,
    preprocessor,
    cfg: RLTokenTrainConfig,
    *,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Token-weighted mean of ``loss_ro`` on a held-out loader. No gradients."""
    import torch

    if max_batches is not None and max_batches <= 0:
        max_batches = None
    was_training = rl_token.training
    rl_token.eval()
    total_loss = 0.0
    total_valid = 0
    total_rms = 0.0
    n_batches = 0
    try:
        with torch.no_grad():
            for batch in loader:
                if max_batches is not None and n_batches >= max_batches:
                    break
                batch = preprocessor(batch)
                batch = _move_batch_tensors(batch, cfg.device)
                loss, z, mask = reconstruction_step(extractor, rl_token, batch, cfg)
                n_valid = int(mask.sum().item())
                if n_valid <= 0:
                    continue
                total_loss += float(loss.item()) * n_valid
                total_rms += masked_token_rms(z, mask) * n_valid
                total_valid += n_valid
                n_batches += 1
    finally:
        if was_training:
            rl_token.train()
    denom = max(total_valid, 1)
    if n_batches == 0:
        return {
            "val_loss_ro": float("inf"),
            "val_z_rms": 0.0,
            "val_n_valid": 0.0,
            "val_batches": 0.0,
        }
    return {
        "val_loss_ro": total_loss / denom,
        "val_z_rms": total_rms / denom,
        "val_n_valid": float(total_valid),
        "val_batches": float(n_batches),
    }


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
    from smolvla_rltoken.vla.sft import load_dataset_info

    torch.manual_seed(cfg.seed)
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    module_cfg = cfg.to_module_config()
    print(f"[stage1] loading SmolVLA from {cfg.checkpoint} ...", flush=True)
    policy = load_smolvla_policy(cfg.checkpoint, device=cfg.device, dtype=cfg.dtype)
    n_episodes = int(load_dataset_info(cfg.dataset_root)["total_episodes"])
    train_ids, val_ids = split_episode_indices(n_episodes, cfg.val_ratio, cfg.seed)
    train_episodes = None if not val_ids else train_ids
    dataset, preprocessor, _ = build_dataset_and_processors(
        policy,
        cfg.dataset_repo_id,
        cfg.dataset_root,
        rename_map=cfg.rename_map,
        video_backend=cfg.video_backend,
        episodes=train_episodes,
    )
    val_loader = None
    if val_ids:
        val_dataset, _, _ = build_dataset_and_processors(
            policy,
            cfg.dataset_repo_id,
            cfg.dataset_root,
            rename_map=cfg.rename_map,
            video_backend=cfg.video_backend,
            episodes=val_ids,
        )
        val_loader = make_stage1_loader(
            val_dataset,
            cfg,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.val_num_workers,
        )
        print(
            f"[stage1] split episodes train={len(train_ids)} val={len(val_ids)} "
            f"(ratio={cfg.val_ratio}, seed={cfg.seed})",
            flush=True,
        )
    else:
        print(f"[stage1] no val split (val_ratio={cfg.val_ratio})", flush=True)

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
    loader = make_stage1_loader(dataset, cfg)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = _init_wandb(cfg)

    step, t0 = 0, time.time()
    data_iter = iter(loader)
    best_val = float("inf")
    last_val_metrics: dict[str, float] | None = None
    while step < cfg.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        batch = preprocessor(batch)
        batch = _move_batch_tensors(batch, cfg.device)

        loss, z, mask = reconstruction_step(extractor, rl_token, batch, cfg)
        z_rms = masked_token_rms(z, mask)
        n_valid = int(mask.sum().item())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
        opt.step()
        step += 1

        if step % cfg.log_freq == 0 or step == 1:
            speed = step / (time.time() - t0)
            print(
                f"[stage1] step {step}/{cfg.steps} loss_ro={loss.item():.5f} "
                f"z_shape={tuple(z.shape)} z_rms={z_rms:.4f} n_valid={n_valid} "
                f"grad_norm={float(grad_norm):.4f} ({speed:.2f} it/s)",
                flush=True,
            )
            if run is not None:
                run.log(
                    {
                        "loss_ro": loss.item(),
                        "z_rms": z_rms,
                        "n_valid": n_valid,
                        "grad_norm": float(grad_norm),
                        "it_s": speed,
                        "lr": cfg.lr,
                    },
                    step=step,
                )

        should_val = (
            val_loader is not None
            and cfg.val_freq > 0
            and (step % cfg.val_freq == 0 or step == 1 or step == cfg.steps)
        )
        if should_val:
            cap = None if step == cfg.steps else cfg.max_val_batches
            last_val_metrics = evaluate_reconstruction(
                extractor,
                rl_token,
                val_loader,
                preprocessor,
                cfg,
                max_batches=cap,
            )
            print(
                f"[stage1] step {step}/{cfg.steps} "
                f"val_loss_ro={last_val_metrics['val_loss_ro']:.5f} "
                f"val_z_rms={last_val_metrics['val_z_rms']:.4f} "
                f"val_batches={int(last_val_metrics['val_batches'])} "
                f"val_n_valid={int(last_val_metrics['val_n_valid'])}",
                flush=True,
            )
            if run is not None:
                run.log(last_val_metrics, step=step)
            if last_val_metrics["val_loss_ro"] < best_val:
                best_val = last_val_metrics["val_loss_ro"]
                _save_stage1_checkpoint(
                    out_dir / "rl_token_best.pt",
                    rl_token,
                    module_cfg,
                    step,
                    train_ids=train_ids,
                    val_ids=val_ids,
                    val_loss_ro=best_val,
                )
                print(
                    f"[stage1] saved {out_dir / 'rl_token_best.pt'} val_loss_ro={best_val:.5f}",
                    flush=True,
                )

        if step % cfg.save_freq == 0 or step == cfg.steps:
            _save_stage1_checkpoint(
                out_dir / "rl_token.pt",
                rl_token,
                module_cfg,
                step,
                train_ids=train_ids,
                val_ids=val_ids,
                val_loss_ro=None if last_val_metrics is None else last_val_metrics["val_loss_ro"],
            )
            print(f"[stage1] saved {out_dir / 'rl_token.pt'}", flush=True)

    if run is not None:
        run.finish()
    print(f"[stage1] done; saved to {out_dir / 'rl_token.pt'}", flush=True)


def _save_stage1_checkpoint(
    path: Path,
    rl_token,
    module_cfg,
    step: int,
    *,
    train_ids: list[int],
    val_ids: list[int],
    val_loss_ro: float | None,
) -> None:
    import torch

    torch.save(
        {
            "rl_token": rl_token.state_dict(),
            "config": module_cfg.to_dict(),
            "step": step,
            "train_episodes": train_ids,
            "val_episodes": val_ids,
            "val_loss_ro": val_loss_ro,
        },
        path,
    )
