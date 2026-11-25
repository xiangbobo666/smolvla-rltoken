"""CPU tests for the rl-experiment-site standard run files."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from smolvla_rltoken import site_log
from smolvla_rltoken.vla.evaluation import _wilson_interval


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_append_metrics_drops_non_finite_values_and_keeps_valid_json(tmp_path: Path):
    site_log.append_metrics(tmp_path, step=1, loss=0.5, bad=float("nan"), inf=math.inf, flag=True)
    site_log.append_metrics(tmp_path, step=2, loss=0.25)
    rows = _lines(tmp_path / "metrics.jsonl")
    assert rows == [{"step": 1, "loss": 0.5, "flag": True}, {"step": 2, "loss": 0.25}]


def test_append_metrics_unwraps_tensor_like_scalars(tmp_path: Path):
    torch = pytest.importorskip("torch")
    site_log.append_eval(tmp_path, step=3, val=torch.tensor(0.125))
    assert _lines(tmp_path / "eval.jsonl") == [{"step": 3, "val": 0.125}]


def test_write_tables_matches_the_site_table_shape(tmp_path: Path):
    path = site_log.write_tables(tmp_path, [site_log.table("T", ["a", "b"], [(1, "x")], note="n")])
    assert json.loads(path.read_text()) == [
        {"title": "T", "note": "n", "columns": ["a", "b"], "rows": [[1, "x"]]}
    ]
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("successes,episodes", [(0, 16), (22, 100), (16, 16), (2, 16)])
def test_wilson_interval_matches_the_evaluation_module(successes: int, episodes: int):
    assert site_log.wilson_interval(successes, episodes) == pytest.approx(
        _wilson_interval(successes, episodes)
    )


def test_rate_cell_reports_the_sft_baseline_interval():
    assert site_log.rate_cell(22, 100) == "22/100 = 22.0% [15.0%, 31.1%]"
    assert site_log.rate_cell(0, 0) == "—"


def test_eval_summary_tables_from_summarize_results_shape():
    summary = {
        "episodes": 100,
        "successes": 22,
        "mean_episode_steps": 186.06,
        "mean_success_steps": 136.636,
        "termination_counts": {"time_limit": 78, "success": 22},
    }
    main, terminations = site_log.eval_summary_tables(
        summary, title="SFT 评测", status="complete", setting={"num_envs": 8}
    )
    assert main["rows"] == [["complete", 100, "22/100 = 22.0% [15.0%, 31.1%]", "186.1", "136.6"]]
    assert main["note"] == "num_envs=8"
    assert terminations["rows"] == [["success", 22], ["time_limit", 78]]


def _lerobot_line(step: str, loss: float) -> str:
    return (
        f"INFO 2026-11-25 12:00:00 ot_train.py:351 step:{step} smpl:6K ep:40 epch:0.04 "
        f"loss:{loss:.3f} grdn:2.100 lr:1.0e-04 updt_s:0.412 data_s:0.003"
    )


def test_parse_lerobot_log_rebuilds_exact_steps_from_log_freq():
    # 1000/1200/1400 all print as "1K"; exact steps come from the log cadence.
    shown = ["200", "400", "600", "800", "1K", "1K", "1K", "2K"]
    text = "\n".join(["INFO creating dataset"] + [_lerobot_line(s, 0.5) for s in shown])
    rows = site_log.parse_lerobot_train_log(text, log_freq=200)
    assert [row["step"] for row in rows] == [200, 400, 600, 800, 1000, 1200, 1400, 1600]
    assert rows[0] == {
        "step": 200.0,
        "epochs": 0.04,
        "loss": 0.5,
        "grad_norm": 2.1,
        "lr": 1e-4,
        "update_s": 0.412,
        "data_s": 0.003,
    }


def test_parse_lerobot_log_resyncs_after_a_resume_jump():
    text = "\n".join(_lerobot_line(s, 0.1) for s in ["200", "400", "10K", "10K"])
    steps = [row["step"] for row in site_log.parse_lerobot_train_log(text, log_freq=200)]
    assert steps[:2] == [200, 400]
    assert site_log._format_big_number(steps[2]) == "10K"
    assert steps[3] == steps[2] + 200


def test_parse_lerobot_log_rejects_bad_log_freq():
    with pytest.raises(ValueError):
        site_log.parse_lerobot_train_log("", log_freq=0)
