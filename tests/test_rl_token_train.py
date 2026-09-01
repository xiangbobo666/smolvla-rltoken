"""CPU tests for Stage 1 RL Token launch config (no GPU, no VLA load)."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from lerobot.processor.rename_processor import RenameObservationsProcessorStep

from smolvla_rltoken.paths import (
    DATASET_ROOT,
    RL_TOKEN_CONFIG_PATH,
    RL_TOKEN_OUTPUT_DIR,
    RL_TOKEN_SMOKE_OUTPUT_DIR,
    SFT_IMAGE_RENAME_MAP,
    SFT_LAST_PRETRAINED,
    SFT_OUTPUT_DIR,
    SMOLVLA_BASE,
    WANDB_PROJECT,
)
from smolvla_rltoken.rlt.train import (
    RL_TOKEN_SMOKE_BATCH_SIZE,
    RL_TOKEN_SMOKE_JOB_NAME,
    RL_TOKEN_SMOKE_LOG_FREQ,
    RL_TOKEN_SMOKE_STEPS,
    RLTokenTrainConfig,
    apply_cli_overrides,
    apply_smoke_overrides,
    check_rl_token,
    is_throwaway_smoke_output,
)
from smolvla_rltoken.vla.dataset import apply_camera_rename, dataset_delta_timestamps
from smolvla_rltoken.vla.sft import EXPECTED_EPISODES, EXPECTED_FRAMES, torchcodec_loads


def _cli_ns(**kwargs) -> Namespace:
    defaults = dict(
        smoke=False,
        checkpoint=None,
        output_dir=None,
        batch_size=None,
        steps=None,
        num_workers=None,
        device=None,
        dtype=None,
        all_prefix_tokens=False,
    )
    defaults.update(kwargs)
    return Namespace(**defaults)


def test_yaml_matches_paths():
    cfg = RLTokenTrainConfig.from_yaml(RL_TOKEN_CONFIG_PATH)
    assert cfg.rename_map == SFT_IMAGE_RENAME_MAP
    assert cfg.use_image_tokens_only is True
    assert cfg.wandb_enable is True
    assert cfg.wandb_project == WANDB_PROJECT
    assert cfg.wandb_disable_artifact is False
    assert cfg.batch_size == 16
    assert cfg.num_workers == 16
    assert cfg.steps == 5000
    assert cfg.d_model == 512
    assert cfg.n_encoder_layers == 2
    assert cfg.video_backend == "torchcodec"
    assert cfg.dtype is None
    assert Path(cfg.dataset_root) == DATASET_ROOT
    assert Path(cfg.checkpoint) == SFT_LAST_PRETRAINED
    assert Path(cfg.output_dir) == RL_TOKEN_OUTPUT_DIR
    assert Path(cfg.checkpoint) != SFT_OUTPUT_DIR


def test_cli_none_does_not_override_yaml():
    cfg = RLTokenTrainConfig.from_yaml(RL_TOKEN_CONFIG_PATH)
    out = apply_cli_overrides(cfg, _cli_ns())
    assert out.steps == 5000
    assert out.batch_size == 16
    assert out.num_workers == 16
    assert Path(out.checkpoint) == SFT_LAST_PRETRAINED


def test_cli_overrides_only_passed_flags():
    cfg = RLTokenTrainConfig.from_yaml(RL_TOKEN_CONFIG_PATH)
    out = apply_cli_overrides(cfg, _cli_ns(steps=3, batch_size=4))
    assert out.steps == 3
    assert out.batch_size == 4
    assert out.num_workers == 16
    assert Path(out.checkpoint) == SFT_LAST_PRETRAINED


def test_smoke_overrides_short_loop_and_throwaway_dir():
    cfg = RLTokenTrainConfig.from_yaml(RL_TOKEN_CONFIG_PATH)
    apply_smoke_overrides(cfg)
    assert cfg.steps == RL_TOKEN_SMOKE_STEPS
    assert cfg.log_freq == RL_TOKEN_SMOKE_LOG_FREQ
    assert cfg.batch_size == RL_TOKEN_SMOKE_BATCH_SIZE
    assert cfg.job_name == RL_TOKEN_SMOKE_JOB_NAME
    assert Path(cfg.output_dir) == RL_TOKEN_SMOKE_OUTPUT_DIR
    assert cfg.wandb_enable is True
    assert cfg.wandb_project == WANDB_PROJECT
    assert cfg.wandb_disable_artifact is True
    assert is_throwaway_smoke_output(cfg.output_dir)
    assert not is_throwaway_smoke_output("/tmp/not-smoke")


def test_module_config_excludes_launch_fields():
    cfg = RLTokenTrainConfig.from_yaml(RL_TOKEN_CONFIG_PATH)
    module = cfg.to_module_config()
    payload = module.to_dict()
    assert "checkpoint" not in payload
    assert payload["d_model"] == 512
    assert payload["steps"] == 5000
    assert payload["use_image_tokens_only"] is True


def test_apply_camera_rename_sets_sft_map():
    class DummyPipeline:
        def __init__(self):
            self.steps = [RenameObservationsProcessorStep(rename_map={})]

    pipe = DummyPipeline()
    apply_camera_rename(pipe)
    assert pipe.steps[0].rename_map == SFT_IMAGE_RENAME_MAP


def test_delta_timestamps_use_dataset_camera_keys():
    meta = SimpleNamespace(
        fps=20,
        features={
            "observation.images.environment_camera": {},
            "observation.images.hand_camera": {},
            "observation.images.insertion_camera": {},
            "observation.state": {},
            "action": {},
        },
    )
    policy = SimpleNamespace(config=SimpleNamespace(action_delta_indices=[0, 1, 2]))
    delta = dataset_delta_timestamps(policy, meta)
    assert "observation.images.environment_camera" in delta
    assert "observation.images.camera1" not in delta
    assert delta["action"] == [0.0, 0.05, 0.1]


def test_check_with_smolvla_base_does_not_load_vla():
    if not (DATASET_ROOT / "meta" / "info.json").is_file():
        return
    if not (SMOLVLA_BASE / "config.json").is_file():
        return
    cfg = RLTokenTrainConfig.from_yaml(RL_TOKEN_CONFIG_PATH)
    cfg.checkpoint = str(SMOLVLA_BASE)
    result = check_rl_token(cfg)
    assert result.info["episodes"] == EXPECTED_EPISODES
    assert result.info["frames"] == EXPECTED_FRAMES
    assert result.info["video_counts"]["insertion_camera"] == EXPECTED_EPISODES
    assert result.info["video_backend"] == "torchcodec"
    assert torchcodec_loads()
    assert result.ok, result.errors


def test_check_missing_sft_last_hints_smolvla_base():
    cfg = RLTokenTrainConfig.from_yaml(RL_TOKEN_CONFIG_PATH)
    cfg.checkpoint = str(SFT_LAST_PRETRAINED)
    if (SFT_LAST_PRETRAINED / "model.safetensors").is_file():
        return
    result = check_rl_token(cfg)
    assert not result.ok
    joined = " ".join(result.errors)
    assert "smolvla_base" in joined
    assert "pretrained_model" in joined or "last" in joined


def test_check_rejects_sft_output_root_as_checkpoint():
    cfg = RLTokenTrainConfig.from_yaml(RL_TOKEN_CONFIG_PATH)
    cfg.checkpoint = str(SFT_OUTPUT_DIR)
    result = check_rl_token(cfg)
    assert not result.ok
    assert any("pretrained_model" in err for err in result.errors)


def test_check_rejects_wrong_rename_map():
    cfg = RLTokenTrainConfig.from_yaml(RL_TOKEN_CONFIG_PATH)
    cfg.checkpoint = str(SMOLVLA_BASE)
    cfg.rename_map = {"observation.images.environment_camera": "observation.images.camera2"}
    result = check_rl_token(cfg)
    assert not result.ok
    assert any("rename_map" in err for err in result.errors)
