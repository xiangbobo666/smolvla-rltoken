"""Stage 2 GPU smoke: parallel ManiSkill + frozen VLA, with GPU/VRAM/RAM peaks.

Does not write into ``outputs/online_rl/`` (the formal Stage 2 parent).
Each run creates ``outputs/online_rl_smoke/<tag>_YYYYMMDD_HHMMSS/``.
Formal training currently uses ``num_envs=16`` in ``configs/rl/actor_critic.yaml``.
This smoke is a short resource probe, not that run.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from smolvla_rltoken.benchmark import _markdown_value, _relative_link
from smolvla_rltoken.paths import BENCHAMRK_DIR, ONLINE_RL_OUTPUT_DIR, ONLINE_RL_SMOKE_OUTPUT_DIR
from smolvla_rltoken.rl.config import OnlineRLConfig
from smolvla_rltoken.rl.train import (
    allocate_smoke_run_dir,
    is_smoke_output_root,
    is_throwaway_smoke_output,
    train_online_rl,
)
from smolvla_rltoken.rlt.pressure import NvidiaSampler, _is_cuda_oom, _nvidia_smi_snapshot

SUCCESS_DEFINITION = (
    "GPU smoke succeeds if parallel ManiSkill (`physx_cuda`) plus frozen SmolVLA/"
    "RL-Token collect, and at least one Actor-Critic update, finish without CUDA "
    "OOM or host OOM. This is a resource probe, not a success-rate evaluation. "
    "Formal training currently uses `num_envs=16` in `configs/rl/actor_critic.yaml`."
)
NVIDIA_SAMPLE_INTERVAL_S = 0.2


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _parse_kib(line: str) -> float | None:
    parts = line.split()
    if len(parts) < 2:
        return None
    try:
        kib = float(parts[1])
    except ValueError:
        return None
    unit = parts[2].lower() if len(parts) > 2 else "kb"
    if unit in {"kb", "kib"}:
        return kib / 1024.0
    if unit in {"mb", "mib"}:
        return kib
    if unit in {"gb", "gib"}:
        return kib * 1024.0
    return kib / 1024.0


def host_memory_snapshot() -> dict[str, float | None]:
    rss_mib = None
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss_mib = _parse_kib(line)
                break
    except OSError:
        rss_mib = None

    mem_total_mib = None
    mem_available_mib = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_total_mib = _parse_kib(line)
            elif line.startswith("MemAvailable:"):
                mem_available_mib = _parse_kib(line)
    except OSError:
        pass

    cgroup_mib = None
    for candidate in (
        Path("/sys/fs/cgroup/memory.current"),
        Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        if not candidate.is_file():
            continue
        try:
            cgroup_mib = int(candidate.read_text().strip()) / (1024 * 1024)
            break
        except (OSError, ValueError):
            continue

    return {
        "rss_mib": rss_mib,
        "mem_total_mib": mem_total_mib,
        "mem_available_mib": mem_available_mib,
        "cgroup_mib": cgroup_mib,
    }


class ResourceSampler:
    """nvidia-smi plus host RSS / cgroup / MemAvailable."""

    def __init__(self, interval_s: float = NVIDIA_SAMPLE_INTERVAL_S):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.nvidia = NvidiaSampler(interval_s=interval_s)
        self.peak_rss_mib = 0.0
        self.peak_cgroup_mib = 0.0
        self.min_mem_available_mib: float | None = None
        self.mem_total_mib = 0.0
        self.host_samples = 0

    def start(self) -> None:
        self.nvidia.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float | int | None]:
        nvidia = self.nvidia.stop()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return {
            **nvidia,
            "rss_peak_mib": self.peak_rss_mib or None,
            "cgroup_peak_mib": self.peak_cgroup_mib or None,
            "mem_available_min_mib": self.min_mem_available_mib,
            "mem_total_mib": self.mem_total_mib or None,
            "host_samples": self.host_samples,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            snap = host_memory_snapshot()
            rss = snap.get("rss_mib")
            cgroup = snap.get("cgroup_mib")
            available = snap.get("mem_available_mib")
            total = snap.get("mem_total_mib")
            if rss:
                self.peak_rss_mib = max(self.peak_rss_mib, float(rss))
            if cgroup:
                self.peak_cgroup_mib = max(self.peak_cgroup_mib, float(cgroup))
            if available is not None:
                current = float(available)
                if self.min_mem_available_mib is None:
                    self.min_mem_available_mib = current
                else:
                    self.min_mem_available_mib = min(self.min_mem_available_mib, current)
            if total:
                self.mem_total_mib = float(total)
            self.host_samples += 1
            self._stop.wait(self.interval_s)


def render_gpu_smoke_markdown(
    *,
    run_name: str,
    status: str,
    config: Mapping[str, Any],
    summary: Mapping[str, Any] | None,
    output_dir: str | Path,
    summary_path: Path,
    updated_at: str | None = None,
) -> str:
    now = updated_at or datetime.now(UTC).isoformat()
    output_link = _relative_link(output_dir, summary_path)
    raw_summary_link = _relative_link(Path(output_dir) / "summary.json", summary_path)
    config_link = _relative_link(Path(output_dir) / "run_config.json", summary_path)
    lines = [
        f"# RL benchmark: {run_name}",
        "",
        "## Run",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Status | {_markdown_value(status)} |",
        f"| Updated (UTC) | {_markdown_value(now)} |",
        f"| Stage | rl |",
        f"| Method | stage2_gpu_smoke |",
        f"| Output artifacts | [{output_dir}]({output_link}) |",
        f"| Raw summary | [summary.json]({raw_summary_link}) |",
        f"| Run config | [run_config.json]({config_link}) |",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "|---|---|",
    ]
    for key in sorted(config):
        lines.append(f"| {key} | {_markdown_value(config[key])} |")
    lines.extend(
        [
            "",
            "## Success definition",
            "",
            SUCCESS_DEFINITION,
            "",
            "## Aggregate metrics",
            "",
            "| Metric | Value |",
            "|---|---|",
        ]
    )
    if summary:
        for key in sorted(summary):
            lines.append(f"| {key} | {_markdown_value(summary[key])} |")
    else:
        lines.append("| env_steps | — |")
    lines.append("")
    return "\n".join(lines)


def write_gpu_smoke_markdown(
    *,
    run_name: str,
    status: str,
    config: Mapping[str, Any],
    summary: Mapping[str, Any] | None,
    output_dir: str | Path,
) -> Path:
    path = BENCHAMRK_DIR / "rl" / f"{run_name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = render_gpu_smoke_markdown(
        run_name=run_name,
        status=status,
        config=config,
        summary=summary,
        output_dir=output_dir,
        summary_path=path,
    )
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(text)
    temporary.replace(path)
    return path


def gpu_smoke_online_rl(
    cfg: OnlineRLConfig,
    *,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Load frozen VLA + encoder, run a short parallel collect/update, record peaks."""
    import torch

    if Path(cfg.output_dir).resolve() == ONLINE_RL_OUTPUT_DIR.resolve():
        raise ValueError(
            f"GPU smoke refuses to write into the formal Stage 2 dir {cfg.output_dir}"
        )
    if not is_throwaway_smoke_output(cfg.output_dir):
        raise ValueError(
            f"GPU smoke must write under {ONLINE_RL_SMOKE_OUTPUT_DIR}, got {cfg.output_dir}"
        )
    if is_smoke_output_root(cfg.output_dir):
        allocate_smoke_run_dir(cfg)
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU smoke requested but torch.cuda.is_available() is false")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = run_name or out_dir.name
    started = datetime.now(UTC)
    config = {
        **cfg.to_dict(),
        "run_name": name,
        "success_definition": SUCCESS_DEFINITION,
        "resolved_vla_checkpoint": str(Path(cfg.vla_checkpoint).resolve()),
        "resolved_rl_token_checkpoint": str(Path(cfg.rl_token_checkpoint).resolve()),
        # The YAML value is 0 ("use explore_std"), which reads as "disabled".
        "resolved_critic_residual_scale": cfg.resolved_critic_residual_scale(),
        "pid": os.getpid(),
    }
    _write_json(out_dir / "run_config.json", config)
    summary: dict[str, Any] = {"status": "running", "env_steps": 0}
    markdown_path = write_gpu_smoke_markdown(
        run_name=name,
        status="running",
        config=config,
        summary=summary,
        output_dir=out_dir,
    )
    print(f"[stage2-gpu-smoke] markdown {markdown_path}", flush=True)
    print(
        f"[stage2-gpu-smoke] num_envs={cfg.num_envs} chunk_len={cfg.chunk_len} "
        f"total_env_steps={cfg.total_env_steps} device={cfg.device}",
        flush=True,
    )

    sampler = ResourceSampler()
    sampler.start()
    status = "failed"
    error = None
    train_result = None
    torch_peak_mib = None
    if cfg.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        train_result = train_online_rl(cfg)
        if cfg.device.startswith("cuda"):
            torch.cuda.synchronize()
        status = "complete"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, KeyboardInterrupt):
            status = "interrupted"
        elif _is_cuda_oom(exc):
            status = "failed"
            print(f"[stage2-gpu-smoke] CUDA OOM: {exc}", flush=True)
        else:
            status = "failed"
        raise
    finally:
        elapsed_s = time.perf_counter() - t0
        peaks = sampler.stop()
        if cfg.device.startswith("cuda") and torch.cuda.is_available():
            torch_peak_mib = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        nvidia_now = _nvidia_smi_snapshot()
        summary = {
            "status": status,
            "elapsed_s": elapsed_s,
            "env_steps": None if train_result is None else train_result.env_steps,
            "buffer_size": None if train_result is None else train_result.buffer_size,
            "did_offline": None if train_result is None else train_result.did_offline,
            "offline_updates": None if train_result is None else train_result.offline_updates,
            "used_actor": None if train_result is None else train_result.used_actor,
            "warmup_reference_matched": (
                None if train_result is None else train_result.warmup_reference_matched
            ),
            "episodes": None if train_result is None else train_result.episodes,
            "successes": None if train_result is None else train_result.successes,
            "vla_episodes": None if train_result is None else train_result.vla_episodes,
            "vla_successes": None if train_result is None else train_result.vla_successes,
            "actor_episodes": None if train_result is None else train_result.actor_episodes,
            "actor_successes": None if train_result is None else train_result.actor_successes,
            "gradient_steps": None if train_result is None else train_result.gradient_steps,
            "reconfigures": None if train_result is None else train_result.reconfigures,
            "num_envs": cfg.num_envs,
            "chunk_len": cfg.chunk_len,
            "torch_peak_mib": torch_peak_mib,
            "nvidia_peak_mib": peaks.get("nvidia_peak_mib") or None,
            "nvidia_total_mib": peaks.get("nvidia_total_mib") or None,
            "gpu_util_peak": peaks.get("gpu_util_peak") or None,
            "power_peak_w": peaks.get("power_peak_w") or None,
            "nvidia_samples": peaks.get("nvidia_samples") or 0,
            "rss_peak_mib": peaks.get("rss_peak_mib"),
            "cgroup_peak_mib": peaks.get("cgroup_peak_mib"),
            "mem_available_min_mib": peaks.get("mem_available_min_mib"),
            "mem_total_mib": peaks.get("mem_total_mib"),
            "host_samples": peaks.get("host_samples") or 0,
            "nvidia_final": nvidia_now,
            "error": error,
            "started_at_utc": started.isoformat(),
            "finished_at_utc": datetime.now(UTC).isoformat(),
        }
        _write_json(out_dir / "summary.json", summary)
        markdown_path = write_gpu_smoke_markdown(
            run_name=name,
            status=status,
            config=config,
            summary=summary,
            output_dir=out_dir,
        )
        print(
            f"[stage2-gpu-smoke] done status={status} "
            f"offline_updates={summary['offline_updates']} "
            f"episodes={summary['episodes']} successes={summary['successes']} "
            f"nvidia_peak_mib={summary['nvidia_peak_mib']} "
            f"torch_peak_mib={summary['torch_peak_mib']} "
            f"rss_peak_mib={summary['rss_peak_mib']} "
            f"cgroup_peak_mib={summary['cgroup_peak_mib']} "
            f"gpu_util_peak={summary['gpu_util_peak']} "
            f"markdown {markdown_path}",
            flush=True,
        )
    return summary
