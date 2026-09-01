#!/usr/bin/env python
"""Stage 1: train RL Token encoder/decoder on PegInsertion demos (Eq. 2).

CPU-only preflight (no GPU, does not load the VLA):

  python scripts/train_rltoken.py --check

GPU smoke (2 steps, throwaway output dir; do not run while SFT holds the GPU):

  python scripts/train_rltoken.py --smoke --checkpoint /root/autodl-tmp/smolvla-rltoken/models/lerobot/smolvla_base

GPU batch-size pressure (throwaway dir, logs loss_ro every step; val off):

  python scripts/train_rltoken.py --pressure

Train when a CUDA GPU is free (conda env ``smolvla-rlt``), after SFT last exists.
Logs train ``loss_ro`` and held-out ``val_loss_ro`` (episode split).
Each formal run writes ``outputs/rl_token/run_YYYYMMDD_HHMMSS/``:

  python scripts/train_rltoken.py
  bash scripts/train_rltoken.sh
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:18082")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:18082")

from smolvla_rltoken.paths import RL_TOKEN_CONFIG_PATH
from smolvla_rltoken.rlt.pressure import pressure_rl_token, resolve_pressure_batch_sizes
from smolvla_rltoken.rlt.train import (
    RLTokenTrainConfig,
    allocate_formal_run_dir,
    apply_cli_overrides,
    check_rl_token,
    is_throwaway_output,
    train_rl_token,
)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _clear_throwaway_dir(output_dir: str) -> None:
    path = Path(output_dir)
    if not is_throwaway_output(path) or not path.exists():
        return
    print(f"[stage1] warning: removing throwaway dir {path}", flush=True)
    shutil.rmtree(path)


def _print_report(cfg: RLTokenTrainConfig) -> int:
    result = check_rl_token(cfg)
    print(f"[stage1] config ok={result.ok}")
    for key in (
        "episodes",
        "frames",
        "image_keys",
        "action_dim",
        "state_dim",
        "video_counts",
        "checkpoint",
        "output_dir",
        "output_root",
        "output_dir_note",
        "use_image_tokens_only",
        "steps",
        "batch_size",
        "train_episodes",
        "val_episodes",
        "val_ratio",
        "val_freq",
        "max_val_batches",
    ):
        if key in result.info:
            print(f"[stage1]   {key}={result.info[key]}")
    for warning in result.warnings:
        print(f"[stage1] warning: {warning}")
    for error in result.errors:
        print(f"[stage1] error: {error}")
    return 0 if result.ok else 1


def _launch(cfg: RLTokenTrainConfig, allow_cpu: bool, *, pressure: bool, batch_sizes) -> None:
    if is_throwaway_output(cfg.output_dir):
        _clear_throwaway_dir(cfg.output_dir)
    result = check_rl_token(cfg)
    result.raise_if_failed()
    if cfg.device.startswith("cuda") and not _cuda_available():
        if allow_cpu:
            print("[stage1] warning: no CUDA GPU; continuing because --allow-cpu was set")
        else:
            print(
                "[stage1] no CUDA GPU. Preflight passed; run this same command when a "
                "4090 is free, or pass --check / --allow-cpu.",
                file=sys.stderr,
            )
            sys.exit(2)
    if pressure:
        pressure_rl_token(cfg, batch_sizes=batch_sizes)
        return
    allocate_formal_run_dir(cfg)
    print(f"[stage1] run dir {cfg.output_dir}", flush=True)
    train_rl_token(cfg)


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 1 RL Token reconstruction training")
    p.add_argument("--check", action="store_true", help="Validate dataset/checkpoint; do not load the VLA")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="2-step GPU smoke into outputs/rl_token_smoke (W&B on, no artifact)",
    )
    p.add_argument(
        "--pressure",
        action="store_true",
        help="GPU batch-size sweep into outputs/rl_token_pressure; logs loss_ro every step",
    )
    p.add_argument("--config", default=str(RL_TOKEN_CONFIG_PATH), help="YAML launch config")
    p.add_argument("--allow-cpu", action="store_true", help="Allow training without CUDA (not useful here)")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument(
        "--batch-sizes",
        default=None,
        help="Comma-separated pressure sweep, e.g. 16,32,48,64",
    )
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None, choices=["float32", "bfloat16"])
    p.add_argument("--all-prefix-tokens", action="store_true", help="Reconstruct lang+state too")
    p.add_argument("--val-freq", type=int, default=None, help="Steps between val_loss_ro (0 disables)")
    p.add_argument("--val-ratio", type=float, default=None, help="Episode holdout fraction in [0, 1)")
    args = p.parse_args()
    if args.smoke and args.pressure:
        p.error("use only one of --smoke / --pressure")

    cfg = apply_cli_overrides(RLTokenTrainConfig.from_yaml(args.config), args)
    if args.check:
        sys.exit(_print_report(cfg))
    batch_sizes = resolve_pressure_batch_sizes(args) if args.pressure else None
    _launch(cfg, allow_cpu=args.allow_cpu, pressure=args.pressure, batch_sizes=batch_sizes)


if __name__ == "__main__":
    main()
