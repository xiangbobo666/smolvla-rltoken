#!/usr/bin/env python
"""Stage 0: SmolVLA SFT on PegInsertion (LeRobot train loop, not a custom trainer).

CPU-only preflight (no GPU, does not load the VLA):

  python scripts/train_sft.py --check

GPU smoke (2 steps, throwaway output dir, W&B on, no model artifact):

  python scripts/train_sft.py --smoke

Train when a CUDA GPU is available (conda env ``smolvla-rlt``):

  python scripts/train_sft.py
  bash scripts/train_sft.sh

Extra ``lerobot-train`` flags can be appended, e.g. ``--batch_size=4``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:18082")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:18082")

from smolvla_rltoken.vla.sft import (
    SFTConfig,
    apply_smoke_overrides,
    build_lerobot_train_argv,
    check_sft,
    is_throwaway_smoke_output,
)
from smolvla_rltoken.paths import SFT_CONFIG_PATH


def _apply_overrides(cfg: SFTConfig, args: argparse.Namespace) -> SFTConfig:
    if args.smoke:
        apply_smoke_overrides(cfg)
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.steps is not None:
        cfg.steps = args.steps
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.device is not None:
        cfg.device = args.device
    return cfg


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
    print(f"[sft] warning: removing throwaway smoke dir {path}", flush=True)
    shutil.rmtree(path)


def _print_report(cfg: SFTConfig, extra: list[str], resume: bool) -> int:
    result = check_sft(cfg, resume=resume)
    argv = build_lerobot_train_argv(cfg, extra, resume=resume)
    print(f"[sft] config ok={result.ok}")
    for key in ("episodes", "frames", "image_keys", "action_dim", "state_dim", "video_counts"):
        if key in result.info:
            print(f"[sft]   {key}={result.info[key]}")
    for warning in result.warnings:
        print(f"[sft] warning: {warning}")
    for error in result.errors:
        print(f"[sft] error: {error}")
    print("[sft] command:")
    print("  " + shlex.join(argv))
    return 0 if result.ok else 1


def _launch(cfg: SFTConfig, extra: list[str], resume: bool, allow_cpu: bool) -> None:
    if is_throwaway_smoke_output(cfg.output_dir) and not resume:
        _clear_throwaway_smoke_dir(cfg.output_dir)
    result = check_sft(cfg, resume=resume)
    result.raise_if_failed()
    if cfg.device.startswith("cuda") and not _cuda_available():
        if allow_cpu:
            print("[sft] warning: no CUDA GPU; continuing because --allow-cpu was set")
        else:
            print(
                "[sft] no CUDA GPU. Preflight passed; run this same command on a 4090 "
                "instance, or pass --check / --allow-cpu.",
                file=sys.stderr,
            )
            sys.exit(2)

    Path(cfg.output_dir).parent.mkdir(parents=True, exist_ok=True)
    argv = build_lerobot_train_argv(cfg, extra, resume=resume)
    resolved = shutil.which(argv[0])
    if resolved is None:
        raise FileNotFoundError(
            "lerobot-train not found on PATH. Activate conda env smolvla-rlt first."
        )
    print(f"[sft] exec {shlex.join(argv)}", flush=True)
    os.execv(resolved, argv)


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 0 SmolVLA SFT (wraps lerobot-train)")
    p.add_argument("--check", action="store_true", help="Validate dataset/checkpoint and print the command")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="2-step GPU smoke into outputs/sft/peg_insertion_smoke (W&B on, no artifact)",
    )
    p.add_argument("--config", default=str(SFT_CONFIG_PATH), help="YAML launch config")
    p.add_argument("--allow-cpu", action="store_true", help="Allow lerobot-train without CUDA (not useful here)")
    p.add_argument("--resume", action="store_true", help="Resume from output_dir/checkpoints/last")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", default=None)
    args, extra = p.parse_known_args()

    cfg = _apply_overrides(SFTConfig.from_yaml(args.config), args)
    if args.check:
        sys.exit(_print_report(cfg, extra, resume=args.resume))
    _launch(cfg, extra, resume=args.resume, allow_cpu=args.allow_cpu)


if __name__ == "__main__":
    main()
