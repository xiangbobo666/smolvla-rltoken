#!/usr/bin/env python3
"""Record one PegInsertion episode for rl-experiment-site's 3D replay viewer.

Two sources (protocol in ``docs/site/viewer3d.md``):

  --demo      replay a ManiSkill demo ``trajectory.h5`` through ``env_states``
              (no policy, no GPU; validates the export path end to end)
  --checkpoint
              run the SFT SmolVLA policy with the same env, seed handling and
              chunk execution as ``scripts/eval.py`` (needs GPU)

Writes ``<key>.glb``, ``<key>.traj.json`` and ``<key>.json`` to
``outputs/viewer3d/``.  ``--check`` validates inputs on CPU without importing
the simulator or the model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smolvla_rltoken import viewer3d  # noqa: E402
from smolvla_rltoken.paths import HF_HOME, SFT_LAST_PRETRAINED  # noqa: E402

DEFAULT_DEMO = REPO_ROOT / "data/maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.h5"
DEFAULT_CAMERA_CONFIG = REPO_ROOT / "configs/vla/peg_insertion_three_cameras.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/viewer3d"
PROJECT_ID = "smolvla-rltoken"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--demo", type=Path, help=f"Demo trajectory.h5 (default when no source: {DEFAULT_DEMO})")
    source.add_argument("--checkpoint", type=Path, help=f"SFT pretrained_model dir, e.g. {SFT_LAST_PRETRAINED}")
    parser.add_argument("--episode-id", type=int, default=0, help="Demo episode_id (demo mode).")
    parser.add_argument("--seed", type=int, default=10_000, help="Env seed (policy mode; eval.py starts at 10000).")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--chunk-execution-ratio", type=float, default=None, help="Policy mode; eval.py default.")
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sim-backend",
        choices=("physx_cpu", "physx_cuda"),
        default=None,
        help="Default: physx_cpu for demos, physx_cuda for policies (same as eval.py).",
    )
    parser.add_argument("--key", help="Recording key; default derived from source and seed.")
    parser.add_argument("--label", help="Human-readable label for the site recording list.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing recording with the same key.")
    parser.add_argument("--check", action="store_true", help="CPU-only input validation.")
    args = parser.parse_args(argv)
    if args.checkpoint is None and args.demo is None:
        args.demo = DEFAULT_DEMO
    if args.sim_backend is None:
        args.sim_backend = "physx_cpu" if args.demo is not None else "physx_cuda"
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    return args


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def demo_episode(demo: Path, episode_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (env_info, episode) from the demo's sidecar JSON."""
    meta_path = demo.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    for episode in meta["episodes"]:
        if int(episode["episode_id"]) == episode_id:
            return meta["env_info"], episode
    raise KeyError(f"episode_id {episode_id} not in {meta_path} ({len(meta['episodes'])} episodes)")


def episode_seed(episode: dict[str, Any]) -> int:
    """motionplanning demos store reset_kwargs.seed; rl demos only episode_seed."""
    seed = episode.get("reset_kwargs", {}).get("seed")
    return int(episode["episode_seed"] if seed is None else seed)


def default_key(args: argparse.Namespace, seed: int) -> str:
    if args.demo is not None:
        flavour = args.demo.parent.name  # motionplanning / rl
        short = {"motionplanning": "mp"}.get(flavour, flavour)
        return f"demo-{short}-ep{args.episode_id:04d}-seed{seed}"
    return f"sft-seed{seed}"


def check_inputs(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []
    info: dict[str, Any] = {"output_dir": rel(args.output_dir), "sim_backend": args.sim_backend}
    seed = args.seed
    if args.demo is not None:
        info["source"] = "demo"
        info["demo"] = rel(args.demo)
        if not args.demo.is_file() or not args.demo.with_suffix(".json").is_file():
            errors.append(f"missing demo {args.demo} (+ .json); run: python -m mani_skill.utils.download_demo "
                          "PegInsertionSide-v1 -o data/maniskill/demos")
        else:
            try:
                env_info, episode = demo_episode(args.demo, args.episode_id)
            except KeyError as exc:
                errors.append(str(exc))
            else:
                seed = episode_seed(episode)
                info.update(env_id=env_info["env_id"], control_mode=episode["control_mode"],
                            demo_steps=episode["elapsed_steps"], demo_success=episode.get("success"))
    else:
        from smolvla_rltoken.vla.evaluation import check_eval_inputs

        info["source"] = "policy"
        check = check_eval_inputs(args.checkpoint, args.camera_config)
        info["checkpoint"] = check
        errors.extend(check["errors"])
    info["seed"] = seed
    key = args.key or default_key(args, seed)
    try:
        viewer3d.validate_key(key)
    except ValueError as exc:
        errors.append(str(exc))
    info["key"] = key
    if (args.output_dir / f"{key}.json").exists() and not args.overwrite:
        errors.append(f"recording {key} already exists in {rel(args.output_dir)} (use --overwrite)")
    info["ok"] = not errors
    info["errors"] = errors
    return info


def make_demo_env(env_info: dict[str, Any], control_mode: str, sim_backend: str):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    return gym.make(env_info["env_id"], obs_mode="none", control_mode=control_mode,
                    reward_mode="none", sim_backend=sim_backend)


def start_recorder(unwrapped) -> tuple[viewer3d.TrajectoryRecorder, bytes, dict[str, Any]]:
    """Export the episode's meshes right after reset (geometry is per reconfigure)."""
    parts, dynamic = viewer3d.collect_mesh_parts(unwrapped)
    static = viewer3d.read_poses(unwrapped, sorted({p.body for p in parts} - set(dynamic)))
    glb = viewer3d.build_glb(parts)
    extras = {
        "geometry": viewer3d.episode_geometry(unwrapped),
        "camera_presets": viewer3d.episode_camera_presets(unwrapped),
        "default_preset": "hole_side",
        "xray_names": list(viewer3d.XRAY_NAMES),
        "xray_label": viewer3d.XRAY_LABEL,
        "hud_fields": [dict(f) for f in viewer3d.HUD_FIELDS],
        "hud_progress": dict(viewer3d.HUD_PROGRESS),
        "mesh_parts": len(parts),
    }
    return viewer3d.TrajectoryRecorder(dynamic, static), glb, extras


def capture(recorder: viewer3d.TrajectoryRecorder, unwrapped, step: int, step_dt: float, source: str) -> dict:
    hud = viewer3d.read_peg_hud(unwrapped)
    row = {"step": step, "t": step * step_dt, "source": source, **hud}
    recorder.capture(viewer3d.read_poses(unwrapped, recorder.dynamic_bodies), row)
    return row


def record_demo(args: argparse.Namespace, key: str) -> tuple[dict, bytes, dict]:
    import h5py
    from mani_skill.utils import common

    env_info, episode = demo_episode(args.demo, args.episode_id)
    seed = episode_seed(episode)
    env = make_demo_env(env_info, episode["control_mode"], args.sim_backend)
    try:
        unwrapped = env.unwrapped
        env.reset(seed=seed, options={"reconfigure": True})
        with h5py.File(args.demo, "r") as handle:
            group = handle[f"traj_{args.episode_id}"]
            states = common.to_numpy({k: {kk: vv[:] for kk, vv in v.items()} for k, v in group["env_states"].items()})
        # Older demos (the rl set) name the merged actors per sub-scene: peg_0, box_with_hole_0.
        for legacy in [name for name in states["actors"] if name in ("peg_0", "box_with_hole_0")]:
            states["actors"][legacy[: -len("_0")]] = states["actors"].pop(legacy)
        num_states = len(states["actors"]["peg"])
        # Same seed must rebuild the same peg/box; compare against the demo's first state.
        reset_peg = viewer3d.read_poses(unwrapped, ["peg"])["peg"]
        drift = max(abs(a - b) for a, b in zip(reset_peg, states["actors"]["peg"][0][:7]))
        if drift > 1e-4:
            raise RuntimeError(f"reset(seed={seed}) does not reproduce the demo's initial peg pose (max |diff|={drift:.2e})")

        step_dt = 1.0 / float(unwrapped.control_freq)
        recorder: viewer3d.TrajectoryRecorder | None = None
        glb = b""
        extras: dict[str, Any] = {}
        rows = []
        for i in range(num_states):
            unwrapped.set_state_dict(common.to_tensor(common.index_dict_array(states, i, inplace=False)))
            if recorder is None:
                recorder, glb, extras = start_recorder(unwrapped)
            rows.append(capture(recorder, unwrapped, i, step_dt, "demo"))
    finally:
        env.close()

    final = rows[-1]
    demo_success = bool(episode.get("success"))
    if final["success"] != demo_success:
        print(f"warning: replayed final success={final['success']} but demo says {demo_success}", flush=True)
    metadata = {
        "project": PROJECT_ID,
        "key": key,
        "source": "demo",
        "provenance": f"ManiSkill {args.demo.parent.name} demo replayed via env_states (no policy)",
        "demo": rel(args.demo),
        "episode_id": args.episode_id,
        "seed": seed,
        "env_id": env_info["env_id"],
        "sim_backend": args.sim_backend,
        "step_dt": step_dt,
        "fps": round(1.0 / step_dt),
        **extras,
    }
    summary = summary_fields(metadata, rows, reason="success" if final["success"] else "demo_end")
    return recorder.payload(metadata), glb, summary


def record_policy(args: argparse.Namespace, key: str) -> tuple[dict, bytes, dict]:
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    from smolvla_rltoken.vla.evaluation import (
        DEFAULT_CHUNK_EXECUTION_RATIO,
        SFTPolicyRunner,
        make_maniskill_env,
        run_episode,
    )

    ratio = args.chunk_execution_ratio if args.chunk_execution_ratio is not None else DEFAULT_CHUNK_EXECUTION_RATIO
    camera_specs = json.loads(args.camera_config.read_text())
    print(f"Loading policy from {args.checkpoint} on {args.device}", flush=True)
    runner = SFTPolicyRunner(args.checkpoint, device=args.device, chunk_execution_ratio=ratio)
    env = make_maniskill_env(camera_specs, sim_backend=args.sim_backend, max_episode_steps=args.max_steps, num_envs=1)
    unwrapped = env.unwrapped
    step_dt = 1.0 / float(unwrapped.control_freq)
    state: dict[str, Any] = {"recorder": None, "rows": []}

    def on_frame(_observation, step: int) -> None:
        if step == 0:
            state["recorder"], state["glb"], state["extras"] = start_recorder(unwrapped)
        # SFT-only for now; Stage 2 recordings will mark Actor chunks as "actor".
        state["rows"].append(capture(state["recorder"], unwrapped, step, step_dt, "vla"))

    try:
        result = run_episode(env, runner, episode_index=0, seed=args.seed, max_steps=args.max_steps,
                             frame_callback=on_frame)
    finally:
        env.close()

    metadata = {
        "project": PROJECT_ID,
        "key": key,
        "source": "sft",
        "provenance": "SFT SmolVLA single-env rollout (same runner/seed path as scripts/eval.py)",
        "checkpoint": rel(args.checkpoint),
        "seed": args.seed,
        "env_id": "PegInsertionSideThreeCamera-v1",
        "sim_backend": args.sim_backend,
        "chunk_size": runner.chunk_size,
        "action_steps_per_inference": runner.action_steps,
        "max_steps": args.max_steps,
        "step_dt": step_dt,
        "fps": round(1.0 / step_dt),
        **state["extras"],
    }
    summary = summary_fields(metadata, state["rows"], reason=result.termination or "max_steps")
    return state["recorder"].payload(metadata), state["glb"], summary


def summary_fields(metadata: dict[str, Any], rows: list[dict[str, Any]], *, reason: str) -> dict[str, Any]:
    final = rows[-1]
    keep = ("project", "source", "provenance", "demo", "episode_id", "checkpoint", "seed", "env_id", "geometry")
    return {
        **{k: metadata[k] for k in keep if k in metadata},
        "steps": len(rows) - 1,
        "seconds": round((len(rows) - 1) * metadata["step_dt"], 3),
        "ok": bool(final["success"]),
        "reason": reason,
        "depth_mm": final["depth_mm"],
        "max_depth_mm": max(r["depth_mm"] for r in rows),
        "lateral_mm": final["lateral_mm"],
        "success_depth_mm": final["success_depth_mm"],
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    check = check_inputs(args)
    if args.check or not check["ok"]:
        print(json.dumps(check, indent=2, ensure_ascii=False))
        if not check["ok"]:
            raise SystemExit(1)
        return
    key = check["key"]
    started = time.time()
    traj, glb, summary = record_demo(args, key) if args.demo is not None else record_policy(args, key)
    summary["glb_bytes"] = len(glb)
    if args.label:
        summary["label"] = args.label
        traj["metadata"]["label"] = args.label
    paths = viewer3d.write_recording(args.output_dir, key, glb, traj, summary)
    print(
        f"RECORDED {key}: {summary['steps']} steps, ok={summary['ok']} ({summary['reason']}), "
        f"depth {summary['depth_mm']:.1f} mm, GLB {len(glb) / 1e6:.1f} MB, {time.time() - started:.1f}s",
        flush=True,
    )
    for name, path in paths.items():
        print(f"  {name}: {rel(path)}")


if __name__ == "__main__":
    main()
