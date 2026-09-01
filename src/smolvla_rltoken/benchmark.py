"""Repository-wide Markdown benchmark summaries.

The directory name intentionally follows the repository's existing
``benchamrk/`` spelling.  SFT, RLT, RL, ablations, and future evaluators should
all use this module (or emit the same fields) so results remain discoverable.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from smolvla_rltoken.paths import BENCHAMRK_DIR, REPO_ROOT


def _markdown_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        return f"`{json.dumps(value, ensure_ascii=False, sort_keys=True)}`"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _relative_link(target: str | Path, summary_path: Path) -> str:
    target_path = Path(target)
    if not target_path.is_absolute():
        target_path = REPO_ROOT / target_path
    relative = os.path.relpath(target_path, start=summary_path.parent)
    return Path(relative).as_posix()


def benchmark_markdown_path(stage: str, run_name: str) -> Path:
    normalized_stage = stage.strip().lower().replace(" ", "_")
    stage_path = Path(normalized_stage)
    if (
        not normalized_stage
        or stage_path.is_absolute()
        or len(stage_path.parts) != 1
        or any(part in {".", ".."} for part in stage_path.parts)
    ):
        raise ValueError(f"Invalid benchmark stage: {stage!r}")
    safe_name = run_name.strip().replace("/", "_").replace(" ", "_")
    if not safe_name:
        raise ValueError("Benchmark run name must not be empty")
    return BENCHAMRK_DIR / normalized_stage / f"{safe_name}.md"


def render_benchmark_markdown(
    *,
    stage: str,
    run_name: str,
    status: str,
    config: Mapping[str, Any],
    summary: Mapping[str, Any] | None,
    episodes: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    summary_path: Path,
) -> str:
    """Render a stable human-readable summary for any evaluation stage."""
    now = datetime.now(UTC).isoformat()
    output_link = _relative_link(output_dir, summary_path)
    raw_summary_link = _relative_link(Path(output_dir) / "summary.json", summary_path)
    episodes_link = _relative_link(Path(output_dir) / "episodes.jsonl", summary_path)
    config_link = _relative_link(Path(output_dir) / "run_config.json", summary_path)

    lines = [
        f"# {stage.upper()} benchmark: {run_name}",
        "",
        "## Run",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Status | {_markdown_value(status)} |",
        f"| Updated (UTC) | {_markdown_value(now)} |",
        f"| Stage | {_markdown_value(stage)} |",
        f"| Output artifacts | [{output_dir}]({output_link}) |",
        f"| Raw summary | [summary.json]({raw_summary_link}) |",
        f"| Per-episode JSONL | [episodes.jsonl]({episodes_link}) |",
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
        lines.append("| completed_episodes | 0 |")

    lines.extend(
        [
            "",
            "## Episodes",
            "",
            "| Episode | Seed | Success | Steps | Termination | Reward sum | Video |",
            "|---:|---:|:---:|---:|---|---:|---|",
        ]
    )
    if episodes:
        for episode in episodes:
            video = episode.get("video")
            if video:
                video_target = Path(output_dir) / str(video)
                video_cell = f"[video]({_relative_link(video_target, summary_path)})"
            else:
                video_cell = "—"
            lines.append(
                "| {episode} | {seed} | {success} | {steps} | {termination} | {reward} | {video} |".format(
                    episode=_markdown_value(episode.get("episode_index")),
                    seed=_markdown_value(episode.get("seed")),
                    success="✅" if episode.get("success") else "❌",
                    steps=_markdown_value(episode.get("steps")),
                    termination=_markdown_value(episode.get("termination")),
                    reward=_markdown_value(episode.get("reward_sum")),
                    video=video_cell,
                )
            )
    else:
        lines.append("| — | — | — | — | — | — | — |")
    lines.append("")
    return "\n".join(lines)


def write_benchmark_markdown(
    *,
    stage: str,
    run_name: str,
    status: str,
    config: Mapping[str, Any],
    summary: Mapping[str, Any] | None,
    episodes: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> Path:
    """Atomically write/update ``benchamrk/<stage>/<run>.md``."""
    path = benchmark_markdown_path(stage, run_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = render_benchmark_markdown(
        stage=stage,
        run_name=run_name,
        status=status,
        config=config,
        summary=summary,
        episodes=episodes,
        output_dir=output_dir,
        summary_path=path,
    )
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(text)
    temporary.replace(path)
    return path
