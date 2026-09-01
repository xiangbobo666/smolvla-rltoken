"""CPU tests for Stage 1 pressure sweep records (no GPU, no VLA load)."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import torch

from smolvla_rltoken.paths import RL_TOKEN_CONFIG_PATH, RL_TOKEN_OUTPUT_DIR
from smolvla_rltoken.rlt import pressure as pressure_module
from smolvla_rltoken.rlt.pressure import (
    parse_batch_sizes,
    pressure_rl_token,
    recommend_batch_size,
    render_pressure_markdown,
    resolve_pressure_batch_sizes,
    write_pressure_markdown,
)
from smolvla_rltoken.rlt.train import RL_TOKEN_PRESSURE_BATCH_SIZES, RLTokenTrainConfig, masked_token_rms


def test_parse_batch_sizes_default_and_csv():
    assert parse_batch_sizes(None) == RL_TOKEN_PRESSURE_BATCH_SIZES
    assert parse_batch_sizes("16, 32,64") == (16, 32, 64)


def test_parse_batch_sizes_rejects_empty_or_nonpositive():
    with pytest.raises(ValueError):
        parse_batch_sizes(" ")
    with pytest.raises(ValueError):
        parse_batch_sizes("0,16")


def test_resolve_pressure_batch_sizes_prefers_csv():
    args = Namespace(batch_sizes="16,48", batch_size=32)
    assert resolve_pressure_batch_sizes(args) == (16, 48)
    args = Namespace(batch_sizes=None, batch_size=24)
    assert resolve_pressure_batch_sizes(args) == (24,)
    args = Namespace(batch_sizes=None, batch_size=None)
    assert resolve_pressure_batch_sizes(args) == RL_TOKEN_PRESSURE_BATCH_SIZES


def test_recommend_batch_size_keeps_20pct_headroom():
    results = [
        {"batch_size": 16, "status": "ok", "nvidia_peak_mib": 8000, "nvidia_total_mib": 24564},
        {"batch_size": 32, "status": "ok", "nvidia_peak_mib": 15000, "nvidia_total_mib": 24564},
        {"batch_size": 64, "status": "ok", "nvidia_peak_mib": 22000, "nvidia_total_mib": 24564},
        {"batch_size": 80, "status": "oom", "nvidia_peak_mib": 24500, "nvidia_total_mib": 24564},
    ]
    rec = recommend_batch_size(results)
    assert rec["max_ok_batch_size"] == 64
    # 20% of 24564 is ~4913, so limit ~19651; 22000 is over, 15000 is under.
    assert rec["recommended_batch_size"] == 32


def test_pressure_markdown_includes_loss_and_z_shape(tmp_path: Path):
    output_dir = tmp_path / "outputs/rl_token_pressure"
    output_dir.mkdir(parents=True)
    for name in ("summary.json", "steps.jsonl", "run_config.json"):
        (output_dir / name).write_text("{}\n")
    summary_path = tmp_path / "benchamrk/rlt/pressure_test.md"
    text = render_pressure_markdown(
        run_name="pressure_test",
        status="complete",
        config={"checkpoint": "/ckpt", "batch_sizes": [16, 32]},
        summary={"recommended_batch_size": 32, "max_ok_batch_size": 32, "loss_name": "loss_ro"},
        results=[
            {
                "batch_size": 16,
                "status": "ok",
                "steps": 8,
                "z_shape": [16, 192, 960],
                "first_loss_ro": 1.2,
                "last_loss_ro": 1.1,
                "mean_loss_ro": 1.15,
                "z_rms": 1.01,
                "mean_step_s": 0.4,
                "torch_peak_mib": 9000,
                "nvidia_peak_mib": 10000,
                "gpu_util_peak": 90,
                "power_peak_w": 250,
            }
        ],
        output_dir=output_dir,
        summary_path=summary_path,
    )
    assert "# RLT benchmark: pressure_test" in text
    assert "| Status | complete |" in text
    assert "| Method | stage1_pressure |" in text
    assert "loss_ro" in text
    assert "masked mean-MSE" in text
    assert "| 16 | ok | 8 | `[16, 192, 960]` |" in text
    assert "z_rms" in text


def test_write_pressure_markdown_uses_rlt_stage(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(pressure_module, "BENCHAMRK_DIR", tmp_path / "benchamrk")
    output_dir = tmp_path / "outputs/rl_token_pressure"
    output_dir.mkdir(parents=True)
    for name in ("summary.json", "steps.jsonl", "run_config.json"):
        (output_dir / name).write_text("{}\n")
    path = write_pressure_markdown(
        run_name="pressure_cpu",
        status="complete",
        config={"seed": 1000},
        summary={"recommended_batch_size": 16},
        results=[],
        output_dir=output_dir,
    )
    assert path == tmp_path / "benchamrk/rlt/pressure_cpu.md"
    text = path.read_text()
    assert "| Stage | rlt |" in text
    assert "| recommended_batch_size | 16 |" in text


def test_pressure_refuses_formal_output_dir():
    cfg = RLTokenTrainConfig.from_yaml(RL_TOKEN_CONFIG_PATH)
    cfg.output_dir = str(RL_TOKEN_OUTPUT_DIR)
    with pytest.raises(ValueError, match="formal Stage 1"):
        pressure_rl_token(cfg, batch_sizes=(16,))


def test_masked_token_rms_ignores_padded_tokens():
    z = torch.zeros(2, 3, 4)
    z[:, 0] = 2.0
    mask = torch.tensor([[True, False, False], [True, False, False]])
    assert masked_token_rms(z, mask) == pytest.approx(2.0)
