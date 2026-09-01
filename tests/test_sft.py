"""CPU tests for Stage 0 SFT launch config (no GPU, no VLA load)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from smolvla_rltoken.paths import (
    DATASET_ROOT,
    SFT_CONFIG_PATH,
    SFT_IMAGE_RENAME_MAP,
    SFT_SMOKE_OUTPUT_DIR,
    SMOLVLA_BASE,
    WANDB_PROJECT,
)
from smolvla_rltoken.vla.sft import (
    EXPECTED_EPISODES,
    EXPECTED_FRAMES,
    SFT_SMOKE_BATCH_SIZE,
    SFT_SMOKE_JOB_NAME,
    SFT_SMOKE_LOG_FREQ,
    SFT_SMOKE_STEPS,
    SFTConfig,
    apply_smoke_overrides,
    build_lerobot_train_argv,
    check_sft,
    is_throwaway_smoke_output,
    torchcodec_loads,
)


def test_yaml_matches_paths():
    cfg = SFTConfig.from_yaml(SFT_CONFIG_PATH)
    assert cfg.rename_map == SFT_IMAGE_RENAME_MAP
    assert cfg.freeze_vision_encoder is True
    assert cfg.train_expert_only is True
    assert cfg.push_to_hub is False
    assert cfg.eval_freq == 0
    assert cfg.wandb_enable is True
    assert cfg.wandb_project == WANDB_PROJECT
    assert cfg.wandb_disable_artifact is False
    assert cfg.batch_size == 32
    assert cfg.num_workers == 4
    assert cfg.video_backend == "torchcodec"
    assert Path(cfg.dataset_root) == DATASET_ROOT
    assert Path(cfg.policy_path) == SMOLVLA_BASE


def test_lerobot_argv_contains_rename_and_freeze():
    cfg = SFTConfig.from_yaml(SFT_CONFIG_PATH)
    argv = build_lerobot_train_argv(cfg)
    joined = " ".join(argv)
    assert argv[0] == "lerobot-train"
    assert "--policy.freeze_vision_encoder=true" in argv
    assert "--policy.train_expert_only=true" in argv
    assert "--policy.push_to_hub=false" in argv
    assert "--eval_freq=0" in argv
    assert "--wandb.enable=true" in argv
    assert f"--wandb.project={WANDB_PROJECT}" in argv
    assert "--wandb.disable_artifact=false" in argv
    assert "environment_camera" in joined
    assert "observation.images.camera3" in joined
    assert "--batch_size=32" in argv
    assert "--num_workers=4" in argv
    assert "--dataset.video_backend=torchcodec" in argv
    extra = build_lerobot_train_argv(cfg, extra=["--batch_size=4"])
    assert extra[-1] == "--batch_size=4"
    no_empty = build_lerobot_train_argv(cfg, extra=["", "--batch_size=4", ""])
    assert "" not in no_empty
    assert no_empty[-1] == "--batch_size=4"


def test_smoke_overrides_short_loop_and_throwaway_dir():
    cfg = SFTConfig.from_yaml(SFT_CONFIG_PATH)
    apply_smoke_overrides(cfg)
    assert cfg.steps == SFT_SMOKE_STEPS
    assert cfg.log_freq == SFT_SMOKE_LOG_FREQ
    assert cfg.batch_size == SFT_SMOKE_BATCH_SIZE
    assert cfg.job_name == SFT_SMOKE_JOB_NAME
    assert Path(cfg.output_dir) == SFT_SMOKE_OUTPUT_DIR
    assert cfg.wandb_enable is True
    assert cfg.wandb_project == WANDB_PROJECT
    assert cfg.wandb_disable_artifact is True
    assert is_throwaway_smoke_output(cfg.output_dir)
    assert not is_throwaway_smoke_output("/tmp/not-smoke")
    argv = build_lerobot_train_argv(cfg)
    assert f"--steps={SFT_SMOKE_STEPS}" in argv
    assert f"--log_freq={SFT_SMOKE_LOG_FREQ}" in argv
    assert "--wandb.disable_artifact=true" in argv
    assert f"--wandb.project={WANDB_PROJECT}" in argv


def test_sft_launcher_script():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts" / "train_sft.sh").read_text()
    assert "ulimit -n 1048576" in text
    assert "setsid" in text
    assert "CURSOR_AGENT" in text
    assert "CODEX_SESSION_ID" in text
    assert "CODEX_THREAD_ID" in text
    assert "CODEX_INTERNAL_ORIGINATOR_OVERRIDE" in text
    assert " AgentHost " not in text
    assert "process ancestor codex" in text
    assert "TRAIN_SIGNAL=SIGKILL" in text
    assert "peg_insertion.pid" in text
    assert "train_sft.py" in text
    assert "conda activate smolvla-rlt" in text
    assert "printf '%s\\0'" not in text
    assert "start_sft_detached" not in text
    assert "train_sft_daemon" not in text
    assert not (root / "scripts" / "start_sft_detached.sh").exists()
    assert not (root / "scripts" / "train_sft_daemon.sh").exists()


def test_sft_launcher_refuses_agent_owned_official_run():
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["CODEX_SESSION_ID"] = "test-agent-session"
    result = subprocess.run(
        ["bash", str(root / "scripts" / "train_sft.sh")],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 1
    assert "Refusing to start official SFT" in result.stderr
    assert "CODEX_SESSION_ID" in result.stderr
    assert "AutoDL web terminal" in result.stderr


def test_resume_argv_points_at_last_checkpoint():
    cfg = SFTConfig(output_dir="/tmp/sft-out")
    argv = build_lerobot_train_argv(cfg, resume=True)
    assert "--resume=true" in argv
    assert any(a.endswith("checkpoints/last/pretrained_model/train_config.json") for a in argv)


def test_check_sft_on_local_dataset():
    if not (DATASET_ROOT / "meta" / "info.json").is_file():
        return
    if not (SMOLVLA_BASE / "config.json").is_file():
        return
    cfg = SFTConfig.from_yaml(SFT_CONFIG_PATH)
    cfg.output_dir = str(DATASET_ROOT.parent / "_sft_out_does_not_exist")
    result = check_sft(cfg)
    assert result.info["episodes"] == EXPECTED_EPISODES
    assert result.info["frames"] == EXPECTED_FRAMES
    assert result.info["video_counts"]["insertion_camera"] == EXPECTED_EPISODES
    assert result.info["video_backend"] == "torchcodec"
    assert torchcodec_loads()
    assert result.ok, result.errors


def test_check_sft_rejects_missing_camera(tmp_path: Path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (tmp_path / "data" / "chunk-000").mkdir(parents=True)
    (tmp_path / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"x")
    info = {
        "total_episodes": EXPECTED_EPISODES,
        "total_frames": EXPECTED_FRAMES,
        "features": {
            "action": {"dtype": "float32", "shape": [8]},
            "observation.state": {"dtype": "float32", "shape": [9]},
            "observation.images.environment_camera": {
                "dtype": "video",
                "shape": [512, 512, 3],
            },
            "observation.images.hand_camera": {"dtype": "video", "shape": [512, 512, 3]},
        },
    }
    (meta / "info.json").write_text(json.dumps(info))
    cfg = SFTConfig.from_yaml(SFT_CONFIG_PATH)
    cfg.dataset_root = str(tmp_path)
    cfg.output_dir = str(tmp_path / "out")
    result = check_sft(cfg)
    assert not result.ok
    assert any("insertion_camera" in err for err in result.errors)
