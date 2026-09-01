"""CPU tests for Stage 1 RL Token launch config (no GPU, no VLA load)."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lerobot.processor.rename_processor import RenameObservationsProcessorStep

from smolvla_rltoken.paths import (
    DATASET_ROOT,
    RL_TOKEN_CONFIG_PATH,
    RL_TOKEN_OUTPUT_DIR,
    RL_TOKEN_PRESSURE_OUTPUT_DIR,
    RL_TOKEN_SMOKE_OUTPUT_DIR,
    SFT_IMAGE_RENAME_MAP,
    SFT_LAST_PRETRAINED,
    SFT_OUTPUT_DIR,
    SMOLVLA_BASE,
    WANDB_PROJECT,
)
from smolvla_rltoken.rlt.config import RLTokenConfig
from smolvla_rltoken.rlt.module import RLTokenModule
from smolvla_rltoken.rlt.train import (
    RL_TOKEN_PRESSURE_JOB_NAME,
    RL_TOKEN_PRESSURE_NUM_WORKERS,
    RL_TOKEN_PRESSURE_STEPS,
    RL_TOKEN_SMOKE_BATCH_SIZE,
    RL_TOKEN_SMOKE_JOB_NAME,
    RL_TOKEN_SMOKE_LOG_FREQ,
    RL_TOKEN_SMOKE_MAX_VAL_BATCHES,
    RL_TOKEN_SMOKE_STEPS,
    RL_TOKEN_SMOKE_VAL_FREQ,
    RLTokenTrainConfig,
    allocate_formal_run_dir,
    apply_cli_overrides,
    apply_pressure_overrides,
    apply_smoke_overrides,
    check_rl_token,
    evaluate_reconstruction,
    is_formal_output_root,
    is_throwaway_output,
    is_throwaway_pressure_output,
    is_throwaway_smoke_output,
    new_run_directory,
    split_episode_indices,
)
from smolvla_rltoken.vla.dataset import apply_camera_rename, dataset_delta_timestamps
from smolvla_rltoken.vla.sft import EXPECTED_EPISODES, EXPECTED_FRAMES, torchcodec_loads


def _cli_ns(**kwargs) -> Namespace:
    defaults = dict(
        smoke=False,
        pressure=False,
        checkpoint=None,
        output_dir=None,
        batch_size=None,
        batch_sizes=None,
        steps=None,
        num_workers=None,
        device=None,
        dtype=None,
        all_prefix_tokens=False,
        val_freq=None,
        val_ratio=None,
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
    assert cfg.batch_size == 64
    assert cfg.num_workers == 4
    assert cfg.steps == 5000
    assert cfg.val_ratio == 0.1
    assert cfg.val_freq == 100
    assert cfg.max_val_batches == 16
    assert cfg.val_num_workers == 0
    assert cfg.d_model == 512
    assert cfg.n_encoder_layers == 2
    assert cfg.video_backend == "torchcodec"
    assert cfg.dtype is None
    assert Path(cfg.dataset_root) == DATASET_ROOT
    assert Path(cfg.checkpoint) == SFT_LAST_PRETRAINED
    assert Path(cfg.output_dir) == RL_TOKEN_OUTPUT_DIR
    assert Path(cfg.checkpoint) != SFT_OUTPUT_DIR
    assert is_formal_output_root(cfg.output_dir)


def test_new_run_directory_creates_unique_child(tmp_path: Path):
    from datetime import datetime, timezone

    when = datetime(2026, 9, 1, 12, 35, 40, tzinfo=timezone.utc)
    first = new_run_directory(tmp_path, when=when)
    second = new_run_directory(tmp_path, when=when)
    assert first == tmp_path / "run_20260901_123540"
    assert first.is_dir()
    assert second == tmp_path / "run_20260901_123540_2"
    assert second.is_dir()


def test_allocate_formal_run_dir_only_for_parent(tmp_path: Path):
    from datetime import datetime, timezone

    when = datetime(2026, 9, 1, 20, 47, 8, tzinfo=timezone.utc)
    cfg = RLTokenTrainConfig(output_dir=str(tmp_path), job_name="smolvla_rltoken_stage1")
    allocate_formal_run_dir(cfg, when=when, root=tmp_path)
    assert Path(cfg.output_dir) == tmp_path / "run_20260901_204708"
    assert Path(cfg.output_dir).is_dir()
    assert cfg.job_name == "smolvla_rltoken_stage1_20260901_204708"

    pinned = RLTokenTrainConfig(output_dir=str(tmp_path / "custom_run"))
    allocate_formal_run_dir(pinned, when=when, root=tmp_path)
    assert Path(pinned.output_dir) == tmp_path / "custom_run"

    smoke = RLTokenTrainConfig.from_yaml(RL_TOKEN_CONFIG_PATH)
    apply_smoke_overrides(smoke)
    before = smoke.output_dir
    allocate_formal_run_dir(smoke, when=when, root=tmp_path)
    assert smoke.output_dir == before


def test_cli_none_does_not_override_yaml():
    cfg = RLTokenTrainConfig.from_yaml(RL_TOKEN_CONFIG_PATH)
    out = apply_cli_overrides(cfg, _cli_ns())
    assert out.steps == 5000
    assert out.batch_size == 64
    assert out.num_workers == 4
    assert out.val_freq == 100
    assert Path(out.checkpoint) == SFT_LAST_PRETRAINED


def test_cli_overrides_only_passed_flags():
    cfg = RLTokenTrainConfig.from_yaml(RL_TOKEN_CONFIG_PATH)
    out = apply_cli_overrides(cfg, _cli_ns(steps=3, batch_size=4, val_freq=50, val_ratio=0.2))
    assert out.steps == 3
    assert out.batch_size == 4
    assert out.num_workers == 4
    assert out.val_freq == 50
    assert out.val_ratio == 0.2
    assert Path(out.checkpoint) == SFT_LAST_PRETRAINED


def test_smoke_overrides_short_loop_and_throwaway_dir():
    cfg = RLTokenTrainConfig.from_yaml(RL_TOKEN_CONFIG_PATH)
    apply_smoke_overrides(cfg)
    assert cfg.steps == RL_TOKEN_SMOKE_STEPS
    assert cfg.log_freq == RL_TOKEN_SMOKE_LOG_FREQ
    assert cfg.batch_size == RL_TOKEN_SMOKE_BATCH_SIZE
    assert cfg.job_name == RL_TOKEN_SMOKE_JOB_NAME
    assert cfg.val_freq == RL_TOKEN_SMOKE_VAL_FREQ
    assert cfg.max_val_batches == RL_TOKEN_SMOKE_MAX_VAL_BATCHES
    assert cfg.val_ratio == 0.1
    assert Path(cfg.output_dir) == RL_TOKEN_SMOKE_OUTPUT_DIR
    assert cfg.wandb_enable is True
    assert cfg.wandb_project == WANDB_PROJECT
    assert cfg.wandb_disable_artifact is True
    assert is_throwaway_smoke_output(cfg.output_dir)
    assert not is_throwaway_smoke_output("/tmp/not-smoke")


def test_pressure_overrides_throwaway_dir_and_disables_wandb():
    cfg = RLTokenTrainConfig.from_yaml(RL_TOKEN_CONFIG_PATH)
    apply_pressure_overrides(cfg)
    assert cfg.steps == RL_TOKEN_PRESSURE_STEPS
    assert cfg.log_freq == 1
    assert cfg.num_workers == RL_TOKEN_PRESSURE_NUM_WORKERS
    assert cfg.job_name == RL_TOKEN_PRESSURE_JOB_NAME
    assert cfg.val_ratio == 0.0
    assert cfg.val_freq == 0
    assert Path(cfg.output_dir) == RL_TOKEN_PRESSURE_OUTPUT_DIR
    assert cfg.wandb_enable is False
    assert is_throwaway_pressure_output(cfg.output_dir)
    assert is_throwaway_output(cfg.output_dir)
    assert not is_throwaway_pressure_output(str(RL_TOKEN_OUTPUT_DIR))


def test_pressure_cli_does_not_use_formal_output_dir():
    cfg = RLTokenTrainConfig.from_yaml(RL_TOKEN_CONFIG_PATH)
    out = apply_cli_overrides(cfg, _cli_ns(pressure=True))
    assert Path(out.output_dir) == RL_TOKEN_PRESSURE_OUTPUT_DIR
    assert Path(out.output_dir) != RL_TOKEN_OUTPUT_DIR
    assert out.wandb_enable is False
    assert out.steps == RL_TOKEN_PRESSURE_STEPS


def test_module_config_excludes_launch_fields():
    cfg = RLTokenTrainConfig.from_yaml(RL_TOKEN_CONFIG_PATH)
    module = cfg.to_module_config()
    payload = module.to_dict()
    assert "checkpoint" not in payload
    assert "val_freq" not in payload
    assert "val_ratio" not in payload
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
    assert result.info["train_episodes"] == 900
    assert result.info["val_episodes"] == 100
    assert result.info["val_freq"] == 100
    assert result.info["output_root"] == str(RL_TOKEN_OUTPUT_DIR)
    assert "run_YYYYMMDD_HHMMSS" in result.info["output_dir_note"]
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


def test_check_rejects_invalid_val_ratio():
    cfg = RLTokenTrainConfig.from_yaml(RL_TOKEN_CONFIG_PATH)
    cfg.checkpoint = str(SMOLVLA_BASE)
    cfg.val_ratio = 1.0
    result = check_rl_token(cfg)
    assert not result.ok
    assert any("val_ratio" in err for err in result.errors)


def test_split_episode_indices_is_disjoint_and_seeded():
    train_a, val_a = split_episode_indices(1000, 0.1, seed=1000)
    train_b, val_b = split_episode_indices(1000, 0.1, seed=1000)
    train_c, val_c = split_episode_indices(1000, 0.1, seed=1001)
    assert len(train_a) == 900
    assert len(val_a) == 100
    assert set(train_a).isdisjoint(val_a)
    assert set(train_a) | set(val_a) == set(range(1000))
    assert train_a == train_b and val_a == val_b
    assert val_a != val_c
    assert split_episode_indices(10, 0.0, seed=0) == (list(range(10)), [])
    with pytest.raises(ValueError, match="val_ratio"):
        split_episode_indices(10, 1.0, seed=0)


def test_split_episode_indices_keeps_at_least_one_train():
    train_ids, val_ids = split_episode_indices(2, 0.9, seed=0)
    assert len(train_ids) == 1
    assert len(val_ids) == 1


def test_evaluate_reconstruction_is_token_weighted_and_restores_train_mode():
    torch.manual_seed(0)
    module = RLTokenModule(
        RLTokenConfig(
            vla_width=16,
            d_model=32,
            n_heads=4,
            n_encoder_layers=1,
            n_decoder_layers=1,
            max_recon_tokens=16,
            dropout=0.0,
        )
    )
    module.train()
    z1 = torch.randn(1, 4, 16)
    mask1 = torch.ones(1, 4, dtype=torch.bool)
    z2 = torch.randn(1, 4, 16)
    mask2 = torch.tensor([[True, True, False, False]])
    with torch.no_grad():
        loss1, _ = module.reconstruction_loss(z1, mask1)
        loss2, _ = module.reconstruction_loss(z2, mask2)
    expected = (float(loss1) * 4 + float(loss2) * 2) / 6

    class _ZDataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            z, mask = self.items[index]
            return {"z": z.squeeze(0), "mask": mask.squeeze(0)}

    class _FakeExtractor:
        def extract(self, batch):
            return batch

        def select_tokens(self, feats, image_only):
            return feats["z"], feats["mask"]

    loader = torch.utils.data.DataLoader(
        _ZDataset([(z1, mask1), (z2, mask2)]),
        batch_size=1,
        shuffle=False,
    )
    cfg = RLTokenTrainConfig(device="cpu", use_image_tokens_only=True)
    metrics = evaluate_reconstruction(
        _FakeExtractor(),
        module,
        loader,
        preprocessor=lambda batch: batch,
        cfg=cfg,
    )
    assert module.training
    assert metrics["val_batches"] == 2
    assert metrics["val_n_valid"] == 6
    assert abs(metrics["val_loss_ro"] - expected) < 1e-6

    capped = evaluate_reconstruction(
        _FakeExtractor(),
        module,
        loader,
        preprocessor=lambda batch: batch,
        cfg=cfg,
        max_batches=1,
    )
    assert capped["val_batches"] == 1
    assert capped["val_n_valid"] == 4
    assert abs(capped["val_loss_ro"] - float(loss1)) < 1e-6
