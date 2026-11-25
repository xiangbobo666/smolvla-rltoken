"""Standard run files for rl-experiment-site (``adapter: standard``).

The site reads these files from every run directory registered in
``docs/site/progress.yaml``; the protocol is defined in
``../rl-experiment-site/docs/integration-guide.md``:

  metrics.jsonl  training curve, one JSON object per log point
  eval.jsonl     periodic evaluation curve (Stage 1 val, Stage 2 probe pairs)
  tables.json    summary / evaluation tables ``[{title, note, columns, rows}]``

Standard library only, so every trainer and evaluator can write them without a
site dependency. The x-axis field is always ``step``; what a step means per
stage (optimizer step, env step) is declared in ``docs/site/schemas/*.json``.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

METRICS_FILENAME = "metrics.jsonl"
EVAL_FILENAME = "eval.jsonl"
TABLES_FILENAME = "tables.json"


def _clean(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Keep finite numbers, bools and strings; NaN/inf would make the line invalid JSON."""
    out: dict[str, Any] = {}
    for key, value in fields.items():
        if hasattr(value, "item") and callable(value.item):
            value = value.item()
        if isinstance(value, bool) or isinstance(value, str):
            out[key] = value
        elif isinstance(value, (int, float)):
            if math.isfinite(value):
                out[key] = value
    return out


def append_jsonl(path: str | Path, fields: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_clean(fields), ensure_ascii=False) + "\n")


def append_metrics(run_dir: str | Path, **fields: Any) -> None:
    append_jsonl(Path(run_dir) / METRICS_FILENAME, fields)


def append_eval(run_dir: str | Path, **fields: Any) -> None:
    append_jsonl(Path(run_dir) / EVAL_FILENAME, fields)


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Rewrite a whole JSONL file atomically (used by log converters)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_clean(record), ensure_ascii=False) + "\n")
    temporary.replace(path)


def table(
    title: str, columns: Sequence[str], rows: Sequence[Sequence[Any]], note: str = ""
) -> dict[str, Any]:
    return {"title": title, "note": note, "columns": list(columns), "rows": [list(r) for r in rows]}


def write_tables(run_dir: str | Path, tables: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(run_dir) / TABLES_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(list(tables), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def wilson_interval(successes: int, episodes: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if episodes <= 0:
        return (0.0, 0.0)
    phat = successes / episodes
    denom = 1.0 + z * z / episodes
    center = (phat + z * z / (2 * episodes)) / denom
    half = z * math.sqrt(phat * (1 - phat) / episodes + z * z / (4 * episodes * episodes)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def rate_cell(successes: int, episodes: int) -> str:
    """``22/100 = 22.0% [15.0%, 31.1%]`` (Wilson 95%); ``—`` when there are no episodes."""
    if episodes <= 0:
        return "—"
    lower, upper = wilson_interval(successes, episodes)
    return f"{successes}/{episodes} = {100 * successes / episodes:.1f}% [{100 * lower:.1f}%, {100 * upper:.1f}%]"


def eval_summary_tables(
    summary: Mapping[str, Any], *, title: str, status: str, setting: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Tables for a ``vla.evaluation.summarize_results`` summary (SFT now, RL eval later)."""
    episodes = int(summary.get("episodes", 0))
    successes = int(summary.get("successes", 0))
    mean_success = summary.get("mean_success_steps")
    main = table(
        title,
        ["状态", "局数", "成功率（Wilson 95%）", "平均步数", "成功局平均步数"],
        [
            [
                status,
                episodes,
                rate_cell(successes, episodes),
                f"{float(summary.get('mean_episode_steps', 0.0)):.1f}",
                "—" if mean_success is None else f"{float(mean_success):.1f}",
            ]
        ],
        note=" · ".join(f"{k}={v}" for k, v in setting.items()),
    )
    terminations = table(
        "终止原因",
        ["原因", "局数"],
        sorted((summary.get("termination_counts") or {}).items()),
    )
    return [main, terminations]


# lerobot-train prints ``step:20K smpl:640K ep:4K epch:4.29 loss:0.031 grdn:0.412
# lr:1.0e-05 updt_s:0.412 data_s:0.003``; step/smpl/ep go through
# ``format_big_number`` and lose precision, so exact steps are rebuilt from log_freq.
_LEROBOT_FIELDS = {
    "loss": "loss",
    "grdn": "grad_norm",
    "lr": "lr",
    "updt_s": "update_s",
    "data_s": "data_s",
    "epch": "epochs",
}
_LEROBOT_LINE = re.compile(r"\bstep:(\S+)\s+smpl:\S+\s+ep:\S+\s+epch:\S+")


def _format_big_number(num: float, precision: int = 0) -> str:
    for suffix in ("", "K", "M", "B", "T", "Q"):
        if abs(num) < 1000.0:
            return f"{num:.{precision}f}{suffix}"
        num /= 1000.0
    return str(num)


def parse_lerobot_train_log(text: str, log_freq: int) -> list[dict[str, float]]:
    """Turn lerobot-train stdout into metrics rows with exact ``step`` values."""
    if log_freq <= 0:
        raise ValueError("log_freq must be positive")
    rows: list[dict[str, float]] = []
    last = 0
    for line in text.splitlines():
        match = _LEROBOT_LINE.search(line)
        if not match:
            continue
        shown = match.group(1)
        step = last + log_freq
        if _format_big_number(step) != shown:
            # Resumed or restarted run (the log jumped): take the first multiple
            # of log_freq that prints the same. Exact again from the next line.
            step = next(
                (
                    k * log_freq
                    for k in range(1, 10_000_000 // log_freq + 1)
                    if _format_big_number(k * log_freq) == shown
                ),
                0,
            )
            if step == 0:
                continue
        row: dict[str, float] = {"step": float(step)}
        for token in line[match.start():].split():
            key, _, value = token.partition(":")
            if key in _LEROBOT_FIELDS:
                try:
                    row[_LEROBOT_FIELDS[key]] = float(value)
                except ValueError:
                    pass
        rows.append(row)
        last = step
    return rows
