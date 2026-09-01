#!/usr/bin/env python
"""Stage 1: train RL Token encoder/decoder on PegInsertion demos (Eq. 2).

CPU-only preflight (no GPU, does not load the VLA):

  python scripts/train_rltoken.py --check

GPU smoke (2 steps, throwaway output dir; do not run while SFT holds the GPU):

  python scripts/train_rltoken.py --smoke --checkpoint /root/autodl-tmp/smolvla-rltoken/models/lerobot/smolvla_base

Train when a CUDA GPU is free (conda env ``smolvla-rlt``), after SFT last exists:

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
from smolvla_rltoken.rlt.train import (
    RLTokenTrainConfig,
    apply_cli_overrides,
    check_rl_token,
    is_throwaway_smoke_output,
    train_rl_token,
)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _clear_throwaway_smoke_dir(output_dir: str) -> None:
    path = Path(output_dir)
    if not is_throwaway_smoke_output(path) or not path.exists():
        return
    print(f"[stage1] warning: removing throwaway smoke dir {path}", flush=True)
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
        "use_image_tokens_only",
        "steps",
        "batch_size",
    ):
        if key in result.info:
            print(f"[stage1]   {key}={result.info[key]}")
    for warning in result.warnings:
        print(f"[stage1] warning: {warning}")
    for error in result.errors:
        print(f"[stage1] error: {error}")
    return 0 if result.ok else 1


def _launch(cfg: RLTokenTrainConfig, allow_cpu: bool) -> None:
    if is_throwaway_smoke_output(cfg.output_dir):
        _clear_throwaway_smoke_dir(cfg.output_dir)
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
    train_rl_token(cfg)


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 1 RL Token reconstruction training")
    p.add_argument("--check", action="store_true", help="Validate dataset/checkpoint; do not load the VLA")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="2-step GPU smoke into outputs/rl_token_smoke (W&B on, no artifact)",
    )
    p.add_argument("--config", default=str(RL_TOKEN_CONFIG_PATH), help="YAML launch config")
    p.add_argument("--allow-cpu", action="store_true", help="Allow training without CUDA (not useful here)")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None, choices=["float32", "bfloat16"])
    p.add_argument("--all-prefix-tokens", action="store_true", help="Reconstruct lang+state too")
    args = p.parse_args()

    cfg = apply_cli_overrides(RLTokenTrainConfig.from_yaml(args.config), args)
    if args.check:
        sys.exit(_print_report(cfg))
    _launch(cfg, allow_cpu=args.allow_cpu)


if __name__ == "__main__":
    main()
