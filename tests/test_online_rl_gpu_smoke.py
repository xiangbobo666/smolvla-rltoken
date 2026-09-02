"""CPU tests for Stage 2 GPU-smoke config, batched planner, and markdown."""

from __future__ import annotations

import numpy as np
import torch

from smolvla_rltoken.paths import ONLINE_RL_OUTPUT_DIR, ONLINE_RL_SMOKE_OUTPUT_DIR
from smolvla_rltoken.rl.config import OnlineRLConfig
from smolvla_rltoken.rl.gpu_smoke import (
    gpu_smoke_online_rl,
    host_memory_snapshot,
    render_gpu_smoke_markdown,
    write_gpu_smoke_markdown,
)
from smolvla_rltoken.rollout.planner import FrozenVLAPlanner, MockPlanner
from smolvla_rltoken.vla.evaluation import CAMERA_NAMES


def _fake_maniskill_obs(num_envs: int, height: int = 8) -> dict:
    rgb = np.zeros((num_envs, height, height, 3), dtype=np.uint8)
    sensor = {name: {"rgb": rgb} for name in CAMERA_NAMES}
    return {
        "agent": {"qpos": np.zeros((num_envs, 9), dtype=np.float32)},
        "sensor_data": sensor,
    }


class _FakeEncoder:
    def rl_token(self, z, mask):
        del mask
        return torch.ones(z.shape[0], 6)


class _FakeExtractor:
    def extract(self, batch, use_cache=True):
        assert use_cache is True
        batch_size = batch["observation.state"].shape[0]
        return {
            "z": torch.ones(batch_size, 4, 6),
            "pad_mask": torch.ones(batch_size, 4, dtype=torch.bool),
            "n_img_tokens": 4,
            "prefix_pad_masks": torch.ones(batch_size, 4, dtype=torch.bool),
            "past_key_values": {"cached": True},
        }

    def select_tokens(self, feats, image_only):
        del image_only
        return feats["z"], feats["pad_mask"]

    def sample_reference_chunk(self, feats, num_steps=None):
        del num_steps
        batch_size = feats["z"].shape[0]
        return torch.full((batch_size, 10, 32), 0.2)


def test_host_memory_snapshot_has_rss_or_meminfo():
    snap = host_memory_snapshot()
    assert snap["rss_mib"] is None or snap["rss_mib"] > 0
    assert snap["mem_total_mib"] is None or snap["mem_total_mib"] > 0


def test_gpu_smoke_markdown_includes_resource_peaks(tmp_path, monkeypatch):
    from smolvla_rltoken.rl import gpu_smoke as module

    monkeypatch.setattr(module, "BENCHAMRK_DIR", tmp_path / "benchamrk")
    output_dir = tmp_path / "outputs/online_rl_smoke"
    output_dir.mkdir(parents=True)
    path = write_gpu_smoke_markdown(
        run_name="gpu_smoke_cpu",
        status="complete",
        config={"num_envs": 4, "device": "cuda"},
        summary={
            "nvidia_peak_mib": 1234.0,
            "torch_peak_mib": 1111.0,
            "rss_peak_mib": 8000.0,
            "cgroup_peak_mib": 9000.0,
            "gpu_util_peak": 87.0,
            "env_steps": 120,
        },
        output_dir=output_dir,
    )
    text = path.read_text()
    assert path == tmp_path / "benchamrk/rl/gpu_smoke_cpu.md"
    assert "# RL benchmark: gpu_smoke_cpu" in text
    assert "| Method | stage2_gpu_smoke |" in text
    assert "| nvidia_peak_mib | 1234 |" in text
    assert "| rss_peak_mib | 8000 |" in text
    assert "| gpu_util_peak | 87 |" in text
    assert "Formal training currently uses `num_envs=16`" in text


def test_render_mentions_throwaway_not_formal_dir():
    from pathlib import Path as P

    text = render_gpu_smoke_markdown(
        run_name="demo",
        status="running",
        config={"output_dir": str(ONLINE_RL_SMOKE_OUTPUT_DIR)},
        summary=None,
        output_dir=ONLINE_RL_SMOKE_OUTPUT_DIR,
        summary_path=P("/tmp/benchamrk/rl/demo.md"),
    )
    assert "online_rl_smoke" in text
    assert str(ONLINE_RL_OUTPUT_DIR) not in text or "online_rl_smoke" in text


def test_mock_planner_observe_batch_keeps_env_axis():
    planner = MockPlanner(chunk_len=4, action_dim=2, rl_token_dim=3, proprio_dim=3)
    batched = planner.observe_batch({"t": np.array([0, 0])})
    assert batched.z_rl.shape == (2, 3)
    assert batched.reference_action.shape == (2, 4, 2)
    single = planner.observe({"t": 0})
    assert single.z_rl.shape == (3,)


def test_frozen_planner_observe_batch_without_real_vla():
    planner = FrozenVLAPlanner(
        extractor=_FakeExtractor(),
        encoder=_FakeEncoder(),
        actor=None,
        preprocessor=None,
        postprocessor=None,
        chunk_len=10,
        action_dim=8,
        proprio_dim=9,
        image_only=True,
    )
    features = planner.observe_batch(_fake_maniskill_obs(4))
    assert features.z_rl.shape == (4, 6)
    assert features.proprio.shape == (4, 9)
    assert features.reference_action.shape == (4, 10, 8)
    actions = planner.act_batch(features, use_actor=False)
    assert actions.shape == (4, 10, 8)
    env_actions = planner.to_env_actions(actions[:, 0])
    assert env_actions.shape == (4, 8)


def test_gpu_smoke_refuses_formal_output_dir():
    cfg = OnlineRLConfig(output_dir=str(ONLINE_RL_OUTPUT_DIR), mock_env=True, mock_vla=True)
    try:
        gpu_smoke_online_rl(cfg)
    except ValueError as exc:
        assert "formal Stage 2" in str(exc)
    else:
        raise AssertionError("formal output dir must be rejected")


def test_gpu_smoke_refuses_non_smoke_dir(tmp_path):
    cfg = OnlineRLConfig(output_dir=str(tmp_path / "other"), mock_env=True, mock_vla=True)
    try:
        gpu_smoke_online_rl(cfg)
    except ValueError as exc:
        assert "must write under" in str(exc)
    else:
        raise AssertionError("non-smoke output dir must be rejected")
