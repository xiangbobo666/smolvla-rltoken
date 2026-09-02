#!/usr/bin/env python
"""Stage 2: online chunk-level Actor-Critic (frozen VLA + encoder).

CPU-only preflight (no GPU, does not load the VLA or simulator):

  python scripts/train_online_rl.py --check

Mock smoke (CPU, tagged dir under outputs/online_rl_smoke, no VLA/ManiSkill):

  python scripts/train_online_rl.py --smoke

GPU smoke (real VLA+env, parallel ManiSkill, resource peaks):

  python scripts/train_online_rl.py --gpu-smoke
  python scripts/train_online_rl.py --gpu-smoke --num-envs 8

Formal train from the AutoDL web terminal (not a Cursor/Codex agent shell):

  bash scripts/train_online_rl.sh
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:18082")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:18082")

from smolvla_rltoken.paths import ONLINE_RL_CONFIG_PATH
from smolvla_rltoken.rl.config import OnlineRLConfig
from smolvla_rltoken.rl.train import (
    allocate_formal_run_dir,
    allocate_smoke_run_dir,
    apply_cli_overrides,
    check_online_rl,
    is_formal_output_root,
    is_smoke_output_root,
    train_online_rl,
)


def _print_report(cfg: OnlineRLConfig) -> int:
    result = check_online_rl(cfg)
    print(f"[stage2] config ok={result.ok}")
    for key in (
        "vla_checkpoint",
        "rl_token_checkpoint",
        "output_dir",
        "chunk_len",
        "vla_horizon",
        "stride",
        "use_residual_actor",
        "critic_residual_input",
        "critic_residual_scale",
        "actor_drift_ceiling",
        "human_intervention",
        "num_envs",
        "reconfigure_every_episodes",
        "reward",
        "bc_reduction",
        "bc_beta",
        "bc_pretrain_updates",
        "handover_bc_threshold",
        "ref_dropout",
        "explore_std",
        "success_sample_frac",
        "actor_success_sample_frac",
        "reference_probe_every_episodes",
        "reference_probe_env_steps",
        "probe_deterministic_actor",
        "mock_env",
        "mock_vla",
    ):
        if key in result.info:
            print(f"[stage2]   {key}={result.info[key]}")
    bounds = result.info.get("action_bounds")
    if bounds:
        print(
            f"[stage2]   action_norm_mode={bounds['mode']} "
            f"bound_margin={bounds['margin']}"
        )
        print(f"[stage2]   action_bound_lo={bounds['lo']}")
        print(f"[stage2]   action_bound_hi={bounds['hi']}")
    for warning in result.warnings:
        print(f"[stage2] warning: {warning}")
    for error in result.errors:
        print(f"[stage2] error: {error}")
    return 0 if result.ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 online RL (chunk Actor-Critic)")
    parser.add_argument("--check", action="store_true", help="Validate YAML/paths; do not load the VLA")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Short mock loop under outputs/online_rl_smoke/<tag>_<timestamp>/",
    )
    parser.add_argument(
        "--gpu-smoke",
        dest="gpu_smoke",
        action="store_true",
        help="Short real VLA+ManiSkill loop with parallel envs; records GPU/VRAM/RAM peaks",
    )
    parser.add_argument("--config", default=str(ONLINE_RL_CONFIG_PATH), help="YAML launch config")
    parser.add_argument("--vla-checkpoint", dest="vla_checkpoint", default=None)
    parser.add_argument("--rl-token-checkpoint", dest="rl_token_checkpoint", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--total-env-steps", dest="total_env_steps", type=int, default=None)
    parser.add_argument("--num-envs", dest="num_envs", type=int, default=None)
    args = parser.parse_args()
    if args.smoke and args.gpu_smoke:
        parser.error("use only one of --smoke / --gpu-smoke")

    cfg = apply_cli_overrides(OnlineRLConfig.from_yaml(args.config), args)
    if args.check:
        sys.exit(_print_report(cfg))
    result = check_online_rl(cfg)
    result.raise_if_failed()
    if is_smoke_output_root(cfg.output_dir):
        allocate_smoke_run_dir(cfg)
        print(f"[stage2] smoke run dir {cfg.output_dir}", flush=True)
    elif is_formal_output_root(cfg.output_dir):
        allocate_formal_run_dir(cfg)
        print(f"[stage2] run dir {cfg.output_dir}", flush=True)
    if args.gpu_smoke:
        from smolvla_rltoken.rl.gpu_smoke import gpu_smoke_online_rl

        gpu_smoke_online_rl(cfg)
        return
    train_online_rl(cfg)


if __name__ == "__main__":
    main()
