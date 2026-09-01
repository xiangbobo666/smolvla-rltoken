"""Stage 1 GPU pressure sweep: batch size, loss_ro, and memory.

Does not write into ``outputs/rl_token/`` (the formal Stage 1 parent).
"""

from __future__ import annotations

import json
import math
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from smolvla_rltoken.benchmark import _markdown_value, _relative_link
from smolvla_rltoken.paths import BENCHAMRK_DIR, RL_TOKEN_OUTPUT_DIR, RL_TOKEN_PRESSURE_OUTPUT_DIR
from smolvla_rltoken.rlt.train import (
    RL_TOKEN_PRESSURE_BATCH_SIZES,
    RLTokenTrainConfig,
    make_stage1_loader,
    masked_token_rms,
    _move_batch_tensors,
)

HEADROOM_FRACTION = 0.20
NVIDIA_SAMPLE_INTERVAL_S = 0.2
SUCCESS_DEFINITION = (
    "A batch size succeeds if every pressure step finishes without CUDA OOM. "
    "`loss_ro` is masked mean-MSE over image tokens (not the paper token-sum). "
    "Untrained reconstruction on RMSNorm embeddings should start near `z_rms^2` "
    "(typically O(1)), not NaN/Inf, and not ~0."
)


def parse_batch_sizes(text: str | None) -> tuple[int, ...]:
    if text is None:
        return RL_TOKEN_PRESSURE_BATCH_SIZES
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if not parts:
        raise ValueError("batch sizes must not be empty")
    sizes = tuple(int(part) for part in parts)
    if any(size < 1 for size in sizes):
        raise ValueError(f"batch sizes must be positive: {sizes}")
    return sizes


def resolve_pressure_batch_sizes(args: Any) -> tuple[int, ...]:
    raw = getattr(args, "batch_sizes", None)
    if raw:
        return parse_batch_sizes(raw)
    if getattr(args, "batch_size", None) is not None:
        return (int(args.batch_size),)
    return RL_TOKEN_PRESSURE_BATCH_SIZES


def _is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch

        oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
        if oom_type and isinstance(exc, oom_type):
            return True
    except Exception:
        pass
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def _nvidia_smi_snapshot() -> dict[str, float] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw",
                "--format=csv,nounits,noheader",
            ],
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = output.strip().splitlines()[0]
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 4:
        return None
    try:
        return {
            "memory_used_mib": float(parts[0]),
            "memory_total_mib": float(parts[1]),
            "utilization": float(parts[2]),
            "power_w": float(parts[3]),
        }
    except ValueError:
        return None


class NvidiaSampler:
    """0.2 s nvidia-smi sampler used only for pressure peaks."""

    def __init__(self, interval_s: float = NVIDIA_SAMPLE_INTERVAL_S):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_mem_mib = 0.0
        self.peak_util = 0.0
        self.peak_power_w = 0.0
        self.memory_total_mib = 0.0
        self.samples = 0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float | int]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return {
            "nvidia_peak_mib": self.peak_mem_mib,
            "nvidia_total_mib": self.memory_total_mib,
            "gpu_util_peak": self.peak_util,
            "power_peak_w": self.peak_power_w,
            "nvidia_samples": self.samples,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            snap = _nvidia_smi_snapshot()
            if snap is not None:
                self.peak_mem_mib = max(self.peak_mem_mib, snap["memory_used_mib"])
                self.peak_util = max(self.peak_util, snap["utilization"])
                self.peak_power_w = max(self.peak_power_w, snap["power_w"])
                self.memory_total_mib = snap["memory_total_mib"]
                self.samples += 1
            self._stop.wait(self.interval_s)


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def recommend_batch_size(
    results: Sequence[Mapping[str, Any]],
    headroom_fraction: float = HEADROOM_FRACTION,
) -> dict[str, Any]:
    ok = [row for row in results if row.get("status") == "ok"]
    max_ok = max((int(row["batch_size"]) for row in ok), default=None)
    total = 0.0
    for row in results:
        total = max(total, float(row.get("nvidia_total_mib") or 0.0))
    limit = total * (1.0 - headroom_fraction) if total else None
    recommended = None
    for row in ok:
        peak = float(row.get("nvidia_peak_mib") or row.get("torch_peak_mib") or 0.0)
        if limit is None or peak <= limit:
            recommended = int(row["batch_size"])
    if recommended is None:
        recommended = max_ok
    return {
        "max_ok_batch_size": max_ok,
        "recommended_batch_size": recommended,
        "memory_limit_mib": limit,
        "headroom_fraction": headroom_fraction,
        "gpu_total_mib": total or None,
    }


def render_pressure_markdown(
    *,
    run_name: str,
    status: str,
    config: Mapping[str, Any],
    summary: Mapping[str, Any] | None,
    results: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    summary_path: Path,
    updated_at: str | None = None,
) -> str:
    now = updated_at or datetime.now(UTC).isoformat()
    output_link = _relative_link(output_dir, summary_path)
    raw_summary_link = _relative_link(Path(output_dir) / "summary.json", summary_path)
    steps_link = _relative_link(Path(output_dir) / "steps.jsonl", summary_path)
    config_link = _relative_link(Path(output_dir) / "run_config.json", summary_path)
    lines = [
        f"# RLT benchmark: {run_name}",
        "",
        "## Run",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Status | {_markdown_value(status)} |",
        f"| Updated (UTC) | {_markdown_value(now)} |",
        f"| Stage | rlt |",
        f"| Method | stage1_pressure |",
        f"| Output artifacts | [{output_dir}]({output_link}) |",
        f"| Raw summary | [summary.json]({raw_summary_link}) |",
        f"| Per-step JSONL | [steps.jsonl]({steps_link}) |",
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
        lines.append("| max_ok_batch_size | — |")
    lines.extend(
        [
            "",
            "## Batch sizes",
            "",
            "| batch_size | status | steps | z_shape | first_loss_ro | last_loss_ro | mean_loss_ro | z_rms | mean_step_s | torch_peak_MiB | nvidia_peak_MiB | gpu_util_peak | power_peak_W |",
            "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if results:
        for row in results:
            lines.append(
                "| {batch_size} | {status} | {steps} | {z_shape} | {first_loss_ro} | {last_loss_ro} | {mean_loss_ro} | {z_rms} | {mean_step_s} | {torch_peak_mib} | {nvidia_peak_mib} | {gpu_util_peak} | {power_peak_w} |".format(
                    batch_size=_markdown_value(row.get("batch_size")),
                    status=_markdown_value(row.get("status")),
                    steps=_markdown_value(row.get("steps")),
                    z_shape=_markdown_value(row.get("z_shape")),
                    first_loss_ro=_markdown_value(row.get("first_loss_ro")),
                    last_loss_ro=_markdown_value(row.get("last_loss_ro")),
                    mean_loss_ro=_markdown_value(row.get("mean_loss_ro")),
                    z_rms=_markdown_value(row.get("z_rms")),
                    mean_step_s=_markdown_value(row.get("mean_step_s")),
                    torch_peak_mib=_markdown_value(row.get("torch_peak_mib")),
                    nvidia_peak_mib=_markdown_value(row.get("nvidia_peak_mib")),
                    gpu_util_peak=_markdown_value(row.get("gpu_util_peak")),
                    power_peak_w=_markdown_value(row.get("power_peak_w")),
                )
            )
    else:
        lines.append("| — | — | — | — | — | — | — | — | — | — | — | — | — |")
    lines.append("")
    return "\n".join(lines)


def write_pressure_markdown(
    *,
    run_name: str,
    status: str,
    config: Mapping[str, Any],
    summary: Mapping[str, Any] | None,
    results: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> Path:
    path = BENCHAMRK_DIR / "rlt" / f"{run_name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = render_pressure_markdown(
        run_name=run_name,
        status=status,
        config=config,
        summary=summary,
        results=results,
        output_dir=output_dir,
        summary_path=path,
    )
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(text)
    temporary.replace(path)
    return path


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def pressure_rl_token(
    cfg: RLTokenTrainConfig,
    batch_sizes: Sequence[int] | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Load the frozen VLA once and sweep reconstruction batch sizes."""
    import gc

    import torch

    from smolvla_rltoken.rlt.module import RLTokenModule
    from smolvla_rltoken.vla.dataset import build_dataset_and_processors
    from smolvla_rltoken.vla.extractor import SmolVLAPrefixExtractor
    from smolvla_rltoken.vla.load import load_smolvla_policy

    sizes = tuple(batch_sizes or RL_TOKEN_PRESSURE_BATCH_SIZES)
    if Path(cfg.output_dir).resolve() == RL_TOKEN_OUTPUT_DIR.resolve():
        raise ValueError(
            f"pressure refuses to write into the formal Stage 1 dir {cfg.output_dir}"
        )
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = run_name or datetime.now(UTC).strftime("pressure_%Y%m%d_%H%M%S")
    config = {
        **cfg.to_dict(),
        "batch_sizes": list(sizes),
        "steps_per_size": cfg.steps,
        "headroom_fraction": HEADROOM_FRACTION,
        "resolved_checkpoint": str(Path(cfg.checkpoint).resolve()),
        "run_name": name,
        "success_definition": SUCCESS_DEFINITION,
        "task": "PegInsertionSide-v1 Stage 1 reconstruction (LeRobot demos)",
    }
    _write_json(out_dir / "run_config.json", config)
    steps_path = out_dir / "steps.jsonl"
    steps_path.write_text("")
    results: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"status": "running", "completed_sizes": 0}
    markdown_path = write_pressure_markdown(
        run_name=name,
        status="running",
        config=config,
        summary=summary,
        results=results,
        output_dir=out_dir,
    )
    print(f"[stage1-pressure] markdown {markdown_path}", flush=True)

    torch.manual_seed(cfg.seed)
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    print(f"[stage1-pressure] loading SmolVLA from {cfg.checkpoint} ...", flush=True)
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

    module_cfg = cfg.to_module_config()
    status = "complete"
    error_text = None
    try:
        for batch_size in sizes:
            row = _run_one_size(
                cfg=cfg,
                module_cfg=module_cfg,
                dataset=dataset,
                preprocessor=preprocessor,
                extractor=extractor,
                batch_size=batch_size,
                steps_path=steps_path,
            )
            results.append(row)
            rec = recommend_batch_size(results)
            summary = {
                "status": "running",
                "completed_sizes": len(results),
                "loss_name": "loss_ro",
                **rec,
            }
            _write_json(out_dir / "summary.json", {"summary": summary, "results": results})
            write_pressure_markdown(
                run_name=name,
                status="running",
                config=config,
                summary=summary,
                results=results,
                output_dir=out_dir,
            )
            print(
                f"[stage1-pressure] batch_size={batch_size} status={row['status']} "
                f"loss_ro={row.get('last_loss_ro')} z_shape={row.get('z_shape')} "
                f"nvidia_peak_MiB={row.get('nvidia_peak_mib')}",
                flush=True,
            )
            if row["status"] == "oom":
                break
            if row["status"] not in {"ok", "oom"}:
                status = "failed"
                error_text = str(row.get("error"))
                break
    except Exception as exc:
        status = "failed"
        error_text = str(exc)
        raise
    finally:
        rec = recommend_batch_size(results)
        if status == "complete" and not any(row.get("status") == "ok" for row in results):
            status = "failed"
        summary = {
            "status": status,
            "completed_sizes": len(results),
            "loss_name": "loss_ro",
            "error": error_text,
            **rec,
        }
        _write_json(out_dir / "summary.json", {"summary": summary, "results": results})
        write_pressure_markdown(
            run_name=name,
            status=status,
            config=config,
            summary=summary,
            results=results,
            output_dir=out_dir,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"[stage1-pressure] done status={status} recommended={summary.get('recommended_batch_size')}", flush=True)
    return {"run_name": name, "status": status, "summary": summary, "results": results}


def _run_one_size(
    *,
    cfg: RLTokenTrainConfig,
    module_cfg,
    dataset,
    preprocessor,
    extractor,
    batch_size: int,
    steps_path: Path,
) -> dict[str, Any]:
    import gc

    import torch

    from smolvla_rltoken.rlt.module import RLTokenModule

    cfg.batch_size = batch_size
    sampler = NvidiaSampler()
    losses: list[float] = []
    step_times: list[float] = []
    z_shape = None
    z_rms = None
    n_valid = None
    completed = 0
    loader = None
    rl_token = None
    try:
        if cfg.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        rl_token = RLTokenModule(module_cfg).to(cfg.device)
        params = list(rl_token.parameters())
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        loader = make_stage1_loader(dataset, cfg)
        data_iter = iter(loader)
        sampler.start()
        for step in range(1, cfg.steps + 1):
            t0 = time.time()
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
            z_rms = masked_token_rms(z, mask)
            n_valid = int(mask.sum().item())
            z_shape = list(z.shape)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
            opt.step()
            elapsed = time.time() - t0
            loss_value = float(loss.item())
            losses.append(loss_value)
            if step > 1:
                step_times.append(elapsed)
            completed = step
            record = {
                "batch_size": batch_size,
                "step": step,
                "loss_ro": loss_value,
                "z_rms": z_rms,
                "n_valid": n_valid,
                "z_shape": z_shape,
                "grad_norm": float(grad_norm),
                "step_s": elapsed,
            }
            with steps_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"[stage1-pressure] bs={batch_size} step {step}/{cfg.steps} "
                f"loss_ro={loss_value:.5f} z_shape={tuple(z_shape)} "
                f"z_rms={z_rms:.4f} n_valid={n_valid} "
                f"grad_norm={float(grad_norm):.4f} ({elapsed:.2f} s)",
                flush=True,
            )
            if not _finite(loss_value):
                nvidia = sampler.stop()
                return {
                    "batch_size": batch_size,
                    "status": "failed",
                    "steps": completed,
                    "z_shape": z_shape,
                    "first_loss_ro": losses[0],
                    "last_loss_ro": losses[-1],
                    "mean_loss_ro": _mean(losses),
                    "z_rms": z_rms,
                    "n_valid": n_valid,
                    "mean_step_s": _mean(step_times),
                    "torch_peak_mib": _torch_peak_mib(),
                    "error": "loss_ro was NaN/Inf",
                    **nvidia,
                }
        nvidia = sampler.stop()
        return {
            "batch_size": batch_size,
            "status": "ok",
            "steps": completed,
            "z_shape": z_shape,
            "first_loss_ro": losses[0] if losses else None,
            "last_loss_ro": losses[-1] if losses else None,
            "mean_loss_ro": _mean(losses),
            "z_rms": z_rms,
            "n_valid": n_valid,
            "mean_step_s": _mean(step_times),
            "torch_peak_mib": _torch_peak_mib(),
            **nvidia,
        }
    except Exception as exc:
        sampler.stop()
        if _is_cuda_oom(exc):
            print(f"[stage1-pressure] batch_size={batch_size} CUDA OOM: {exc}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {
                "batch_size": batch_size,
                "status": "oom",
                "steps": completed,
                "z_shape": z_shape,
                "first_loss_ro": losses[0] if losses else None,
                "last_loss_ro": losses[-1] if losses else None,
                "mean_loss_ro": _mean(losses),
                "z_rms": z_rms,
                "n_valid": n_valid,
                "mean_step_s": _mean(step_times),
                "torch_peak_mib": _torch_peak_mib(),
                "error": str(exc),
                "nvidia_peak_mib": sampler.peak_mem_mib or None,
                "nvidia_total_mib": sampler.memory_total_mib or None,
                "gpu_util_peak": sampler.peak_util or None,
                "power_peak_w": sampler.peak_power_w or None,
            }
        raise
    finally:
        if loader is not None:
            del loader
        if rl_token is not None:
            del rl_token
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _torch_peak_mib() -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
