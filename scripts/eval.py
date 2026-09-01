#!/usr/bin/env python3
"""Evaluate the SFT SmolVLA checkpoint on PegInsertionSideThreeCamera-v1.

Success is read from ManiSkill's official ``info["success"]`` signal.  Use
``--check`` for a CPU-only metadata preflight that loads neither CUDA nor the
simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smolvla_rltoken.paths import SFT_LAST_PRETRAINED  # noqa: E402
from smolvla_rltoken.benchmark import write_benchmark_markdown  # noqa: E402
from smolvla_rltoken.vla.evaluation import (  # noqa: E402
    CAMERA_NAMES,
    DEFAULT_CHUNK_EXECUTION_RATIO,
    DEFAULT_RECORD_FAILURES,
    DEFAULT_RECORD_SUCCESSES,
    TASK_PROMPT,
    EpisodeResult,
    OutcomeVideoBudget,
    SFTPolicyRunner,
    check_eval_inputs,
    delete_video_artifacts,
    extract_policy_observation_batch,
    make_maniskill_env,
    reconcile_recorded_video,
    run_episode_batch,
    summarize_results,
)

DEFAULT_CAMERA_CONFIG = REPO_ROOT / "configs/vla/peg_insertion_three_cameras.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/eval/sft_peg_insertion"
FPS = 20
SFT_PIDFILE = REPO_ROOT / "outputs/sft/peg_insertion.pid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=SFT_LAST_PRETRAINED)
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=10_000)
    parser.add_argument("--seeds", type=int, nargs="+", help="Explicit evaluation seeds.")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=1, help="Parallel ManiSkill environments.")
    parser.add_argument(
        "--chunk-execution-ratio",
        type=float,
        default=DEFAULT_CHUNK_EXECUTION_RATIO,
        help="Fraction of each predicted action chunk to execute before replanning (default: 0.70).",
    )
    parser.add_argument("--task", default=TASK_PROMPT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sim-backend",
        choices=("physx_cuda", "physx_cpu"),
        default="physx_cuda",
    )
    parser.add_argument(
        "--record-successes",
        type=int,
        default=DEFAULT_RECORD_SUCCESSES,
        help="Keep three-camera videos for the first N successful episodes (default: 5).",
    )
    parser.add_argument(
        "--record-failures",
        type=int,
        default=DEFAULT_RECORD_FAILURES,
        help="Keep three-camera videos for the first N failed episodes (default: 5).",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="CPU-only metadata validation; do not load torch, policy, or simulator.",
    )
    return parser.parse_args()


def evaluation_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds is not None:
        if not args.seeds:
            raise ValueError("--seeds must not be empty")
        if len(set(args.seeds)) != len(args.seeds):
            raise ValueError("--seeds must be unique")
        return list(args.seeds)
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    return list(range(args.start_seed, args.start_seed + args.episodes))


def default_output_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("run_%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / timestamp


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class EpisodeVideoRecorder:
    """Stream a labelled three-camera montage to MP4 without retaining frames."""

    def __init__(self, target: Path, *, fps: int = FPS):
        import imageio.v2 as imageio

        self.target = target
        self.temporary = target.with_suffix(".partial.mp4")
        self.target.parent.mkdir(parents=True, exist_ok=True)
        if self.target.exists() or self.temporary.exists():
            raise FileExistsError(f"Video already exists: {self.target}")
        self.writer = imageio.get_writer(
            self.temporary,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=2,
        )
        self._closed = False

    def append(self, observation: Mapping[str, Any], step: int, env_index: int = 0) -> None:
        extracted = extract_policy_observation_batch(observation)
        frames = [
            Image.fromarray(extracted[f"observation.images.{camera_name}"][env_index])
            for camera_name in CAMERA_NAMES
        ]
        header = 32
        montage = Image.new("RGB", (sum(frame.width for frame in frames), frames[0].height + header))
        draw = ImageDraw.Draw(montage)
        x = 0
        for camera_name, frame in zip(CAMERA_NAMES, frames, strict=True):
            montage.paste(frame, (x, header))
            draw.text((x + 8, 9), camera_name, fill="white")
            x += frame.width
        draw.text((montage.width - 100, 9), f"step {step}", fill="white")
        self.writer.append_data(np.asarray(montage))

    def close(self, *, completed: bool) -> None:
        if self._closed:
            return
        self.writer.close()
        self._closed = True
        if completed:
            self.temporary.replace(self.target)


def validate_args(args: argparse.Namespace) -> None:
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.record_successes < 0:
        raise ValueError("--record-successes must be non-negative")
    if args.record_failures < 0:
        raise ValueError("--record-failures must be non-negative")
    if not 0.0 < args.chunk_execution_ratio <= 1.0:
        raise ValueError("--chunk-execution-ratio must be in (0, 1]")


def active_sft_pid() -> int | None:
    """Return the active official SFT worker PID, ignoring a stale PID file."""
    try:
        pid = int(SFT_PIDFILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    try:
        os.kill(pid, 0)
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ")
    except (OSError, PermissionError):
        return None
    if b"train_sft.sh" not in command and b"lerobot-train" not in command:
        return None
    return pid


def run_config(args: argparse.Namespace, seeds: list[int], camera_bytes: bytes) -> dict[str, Any]:
    return {
        "checkpoint": str(args.checkpoint),
        "resolved_checkpoint": str(args.checkpoint.resolve()),
        "camera_config": str(args.camera_config),
        "camera_config_sha256": hashlib.sha256(camera_bytes).hexdigest(),
        "seeds": seeds,
        "max_steps": args.max_steps,
        "num_envs": args.num_envs,
        "chunk_execution_ratio_requested": args.chunk_execution_ratio,
        "task": args.task,
        "device": args.device,
        "sim_backend": args.sim_backend,
        "record_successes": args.record_successes,
        "record_failures": args.record_failures,
        "fps": FPS,
        "success_signal": "ManiSkill info.success",
        "started_at_utc": datetime.now(UTC).isoformat(),
    }


def main() -> None:
    args = parse_args()
    seeds = evaluation_seeds(args)
    validate_args(args)

    check = check_eval_inputs(args.checkpoint, args.camera_config)
    if args.check:
        print(json.dumps(check, indent=2, sort_keys=True))
        if not check["ok"]:
            raise SystemExit(1)
        return
    if not check["ok"]:
        formatted = "\n".join(f"  - {error}" for error in check["errors"])
        raise FileNotFoundError(f"Evaluation preflight failed:\n{formatted}")
    training_pid = active_sft_pid()
    if training_pid is not None:
        raise RuntimeError(
            f"SFT worker PID {training_pid} is still active; wait for it to exit before GPU evaluation"
        )

    output_dir = args.output_dir or default_output_dir()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Evaluation output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_bytes = args.camera_config.read_bytes()
    camera_specs = json.loads(camera_bytes)
    run_configuration = run_config(args, seeds, camera_bytes)
    write_json(output_dir / "run_config.json", run_configuration)
    benchmark_name = output_dir.name
    benchmark_path = write_benchmark_markdown(
        stage="sft",
        run_name=benchmark_name,
        status="running",
        config=run_configuration,
        summary=None,
        episodes=[],
        output_dir=output_dir,
    )

    print(f"Loading policy from {args.checkpoint} on {args.device}", flush=True)
    try:
        policy_runner = SFTPolicyRunner(
            args.checkpoint,
            device=args.device,
            task=args.task,
            chunk_execution_ratio=args.chunk_execution_ratio,
        )
    except BaseException:
        write_benchmark_markdown(
            stage="sft",
            run_name=benchmark_name,
            status="failed",
            config=run_configuration,
            summary=None,
            episodes=[],
            output_dir=output_dir,
        )
        raise
    run_configuration["chunk_size"] = policy_runner.chunk_size
    run_configuration["action_steps_per_inference"] = policy_runner.action_steps
    run_configuration["chunk_execution_ratio_effective"] = policy_runner.chunk_execution_ratio
    write_json(output_dir / "run_config.json", run_configuration)
    print(
        f"Action chunk: predict {policy_runner.chunk_size}, execute {policy_runner.action_steps} "
        f"({policy_runner.chunk_execution_ratio:.1%}) before replanning",
        flush=True,
    )
    print(
        f"Keeping videos for the first {args.record_successes} successes and "
        f"{args.record_failures} failures",
        flush=True,
    )
    try:
        env = make_maniskill_env(
            camera_specs,
            sim_backend=args.sim_backend,
            max_episode_steps=args.max_steps,
            num_envs=args.num_envs,
        )
    except BaseException:
        write_benchmark_markdown(
            stage="sft",
            run_name=benchmark_name,
            status="failed",
            config=run_configuration,
            summary=None,
            episodes=[],
            output_dir=output_dir,
        )
        raise
    results: list[EpisodeResult] = []
    video_budget = OutcomeVideoBudget(args.record_successes, args.record_failures)
    episodes_path = output_dir / "episodes.jsonl"
    try:
        with episodes_path.open("x") as episodes_file:
            for batch_start in range(0, len(seeds), args.num_envs):
                actual_seeds = seeds[batch_start : batch_start + args.num_envs]
                actual_count = len(actual_seeds)
                # ManiSkill keeps a fixed vector width. Pad only the final batch;
                # padded results are discarded and never enter the metric.
                padding = args.num_envs - actual_count
                padded_seeds = actual_seeds + [actual_seeds[-1] + i + 1 for i in range(padding)]
                episode_indices = list(range(batch_start, batch_start + actual_count)) + [-1] * padding
                record_batch = video_budget.still_recording()

                video_paths: list[str | None] = []
                recorders: list[EpisodeVideoRecorder | None] = []
                for slot, seed in enumerate(padded_seeds):
                    episode_index = episode_indices[slot]
                    video_relative = None
                    recorder = None
                    if record_batch and episode_index >= 0:
                        video_relative = f"videos/episode_{episode_index:04d}_seed_{seed}.mp4"
                        recorder = EpisodeVideoRecorder(output_dir / video_relative)
                    video_paths.append(video_relative)
                    recorders.append(recorder)

                completed = False
                try:
                    batch_results = run_episode_batch(
                        env,
                        policy_runner,
                        episode_indices=episode_indices,
                        seeds=padded_seeds,
                        max_steps=args.max_steps,
                        frame_callbacks=[
                            recorder.append if recorder is not None else None for recorder in recorders
                        ],
                        videos=video_paths,
                    )
                    completed = True
                finally:
                    for recorder in recorders:
                        if recorder is not None:
                            recorder.close(completed=completed)
                            if not completed:
                                delete_video_artifacts(recorder.target)

                for result in batch_results[:actual_count]:
                    decided = video_budget.decide(result)
                    reconcile_recorded_video(result, decided, output_dir)
                    results.append(decided)
                    episodes_file.write(json.dumps(decided.to_dict(), sort_keys=True) + "\n")
                    episodes_file.flush()
                    summary = summarize_results(results)
                    write_json(output_dir / "summary.json", summary)
                    write_benchmark_markdown(
                        stage="sft",
                        run_name=benchmark_name,
                        status="running",
                        config=run_configuration,
                        summary=summary,
                        episodes=[episode.to_dict() for episode in results],
                        output_dir=output_dir,
                    )
                    print(
                        f"episode={result.episode_index + 1}/{len(seeds)} seed={result.seed} "
                        f"success={result.success} steps={result.steps} "
                        f"running_success={summary['success_rate_percent']:.1f}%",
                        flush=True,
                    )
    except KeyboardInterrupt:
        partial_summary = summarize_results(results) if results else None
        write_benchmark_markdown(
            stage="sft",
            run_name=benchmark_name,
            status="interrupted",
            config=run_configuration,
            summary=partial_summary,
            episodes=[episode.to_dict() for episode in results],
            output_dir=output_dir,
        )
        raise
    except BaseException:
        partial_summary = summarize_results(results) if results else None
        write_benchmark_markdown(
            stage="sft",
            run_name=benchmark_name,
            status="failed",
            config=run_configuration,
            summary=partial_summary,
            episodes=[episode.to_dict() for episode in results],
            output_dir=output_dir,
        )
        raise
    finally:
        env.close()

    summary = summarize_results(results)
    benchmark_path = write_benchmark_markdown(
        stage="sft",
        run_name=benchmark_name,
        status="complete",
        config=run_configuration,
        summary=summary,
        episodes=[episode.to_dict() for episode in results],
        output_dir=output_dir,
    )
    print(
        f"Completed {summary['episodes']} episodes: {summary['successes']} successes, "
        f"success rate {summary['success_rate_percent']:.2f}%"
    )
    print(f"Results: {output_dir / 'summary.json'}")
    print(f"Benchmark Markdown: {benchmark_path}")


if __name__ == "__main__":
    main()
