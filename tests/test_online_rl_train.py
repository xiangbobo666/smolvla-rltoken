"""CPU tests for Stage 2 online RL launch config and mock loop."""

from __future__ import annotations

import json
import os
import subprocess
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import torch

from smolvla_rltoken.paths import (
    CAMERA_CONFIG_PATH,
    ONLINE_RL_CONFIG_PATH,
    ONLINE_RL_OUTPUT_DIR,
    ONLINE_RL_SMOKE_OUTPUT_DIR,
    RL_TOKEN_STAGE1_RUN,
    SFT_LAST_PRETRAINED,
    SFT_OUTPUT_DIR,
    WANDB_PROJECT,
)
from smolvla_rltoken.rl.config import OnlineRLConfig
from smolvla_rltoken.rl.train import (
    EPISODES_FILENAME,
    ONLINE_RL_GPU_SMOKE_CHUNKS,
    ONLINE_RL_GPU_SMOKE_JOB_NAME,
    ONLINE_RL_GPU_SMOKE_NUM_ENVS,
    ONLINE_RL_GPU_SMOKE_WARMUP_CHUNKS,
    ONLINE_RL_SMOKE_CHUNK_LEN,
    ONLINE_RL_SMOKE_ENV_STEPS,
    ONLINE_RL_SMOKE_JOB_NAME,
    ONLINE_RL_SMOKE_SUCCESS_AT,
    EpisodeTracker,
    allocate_formal_run_dir,
    allocate_smoke_run_dir,
    apply_cli_overrides,
    apply_gpu_smoke_overrides,
    apply_smoke_overrides,
    check_online_rl,
    is_formal_output_root,
    is_smoke_output_root,
    is_throwaway_smoke_output,
    smoke_run_tag,
    train_online_rl,
)
from smolvla_rltoken.vla.evaluation import TASK_PROMPT


def _cli_ns(**kwargs) -> Namespace:
    defaults = dict(
        smoke=False,
        gpu_smoke=False,
        vla_checkpoint=None,
        rl_token_checkpoint=None,
        output_dir=None,
        device=None,
        total_env_steps=None,
        num_envs=None,
    )
    defaults.update(kwargs)
    return Namespace(**defaults)


def test_yaml_matches_locked_defaults():
    cfg = OnlineRLConfig.from_yaml(ONLINE_RL_CONFIG_PATH)
    assert cfg.chunk_len == 10
    assert cfg.vla_horizon == 50
    assert cfg.action_dim == 8
    assert cfg.proprio_dim == 9
    assert cfg.rl_token_dim == 512
    assert cfg.stride == 1
    assert cfg.use_residual_actor is False
    assert cfg.human_intervention is False
    assert cfg.num_envs == 16
    assert cfg.reconfigure_every_episodes == 16
    assert cfg.total_env_steps == 1_000_000
    assert cfg.buffer_capacity == 200_000
    assert cfg.max_episode_steps == 200
    assert cfg.reward == "sparse_success"
    assert cfg.dense_reward_debug is False
    assert cfg.ref_dropout == 0.5
    assert cfg.bc_beta == 1.0
    assert cfg.gamma == 0.99
    assert cfg.utd == 5
    assert cfg.critic_updates_per_actor == 2
    # One executed chunk is one transition, so warmup must fill a whole batch.
    # 32000 is a multiple of num_envs * max_episode_steps so warmup does not
    # cut mid-episode (those mixed episodes are counted as Actor).
    assert cfg.warmup_env_steps == 32000
    assert cfg.warmup_env_steps >= cfg.batch_size * cfg.chunk_len
    assert cfg.warmup_env_steps % (cfg.num_envs * cfg.max_episode_steps) == 0
    assert cfg.action_bound_margin == 1.5
    assert cfg.mock_success_at is None
    assert cfg.offline_updates_after_warmup == 1000
    assert cfg.task == TASK_PROMPT
    assert cfg.wandb_project == WANDB_PROJECT
    assert Path(cfg.vla_checkpoint) == SFT_LAST_PRETRAINED
    assert Path(cfg.rl_token_checkpoint) == RL_TOKEN_STAGE1_RUN
    assert Path(cfg.rl_token_checkpoint).name == "rl_token_best.pt"
    assert Path(cfg.output_dir) == ONLINE_RL_OUTPUT_DIR
    assert Path(cfg.camera_config) == CAMERA_CONFIG_PATH
    assert is_formal_output_root(cfg.output_dir)


def test_check_accepts_repo_checkpoints():
    cfg = OnlineRLConfig.from_yaml(ONLINE_RL_CONFIG_PATH)
    result = check_online_rl(cfg)
    assert result.ok, result.errors


def test_check_reports_per_dim_action_bounds():
    cfg = OnlineRLConfig.from_yaml(ONLINE_RL_CONFIG_PATH)
    bounds = check_online_rl(cfg).info["action_bounds"]
    assert bounds["mode"] == "MEAN_STD"
    assert bounds["margin"] == cfg.action_bound_margin
    assert len(bounds["lo"]) == cfg.action_dim
    assert len(bounds["hi"]) == cfg.action_dim
    # The normalized action space is not [-1, 1]; a unit clip would be wrong.
    assert max(bounds["hi"]) > 2.0


def test_check_rejects_warmup_shorter_than_one_batch():
    cfg = OnlineRLConfig.from_yaml(ONLINE_RL_CONFIG_PATH)
    cfg.warmup_env_steps = cfg.batch_size * cfg.chunk_len - cfg.chunk_len
    result = check_online_rl(cfg)
    assert not result.ok
    assert any("warmup_env_steps" in item for item in result.errors)


def test_check_rejects_non_positive_bound_margin():
    cfg = OnlineRLConfig.from_yaml(ONLINE_RL_CONFIG_PATH)
    cfg.action_bound_margin = 0.0
    result = check_online_rl(cfg)
    assert not result.ok
    assert any("action_bound_margin" in item for item in result.errors)


def test_check_rejects_stride_residual_and_sft_root():
    cfg = OnlineRLConfig.from_yaml(ONLINE_RL_CONFIG_PATH)
    cfg.stride = 2
    cfg.use_residual_actor = True
    cfg.human_intervention = True
    cfg.vla_checkpoint = str(SFT_OUTPUT_DIR)
    result = check_online_rl(cfg)
    assert not result.ok
    joined = " ".join(result.errors)
    assert "stride=1" in joined
    assert "non-residual" in joined
    assert "human intervention" in joined
    assert "SFT run root" in joined


def test_check_accepts_parallel_envs_with_reconfigure():
    cfg = OnlineRLConfig.from_yaml(ONLINE_RL_CONFIG_PATH)
    cfg.num_envs = 8
    cfg.reconfigure_every_episodes = 16
    result = check_online_rl(cfg)
    assert result.ok, result.errors
    assert result.info["num_envs"] == 8
    assert result.info["reconfigure_every_episodes"] == 16


def test_check_rejects_parallel_envs_without_reconfigure():
    # Parallel envs freeze PegInsertion geometry unless a reconfigure is forced.
    cfg = OnlineRLConfig.from_yaml(ONLINE_RL_CONFIG_PATH)
    cfg.num_envs = 8
    cfg.reconfigure_every_episodes = 0
    result = check_online_rl(cfg)
    assert not result.ok
    assert any("freezes PegInsertion geometry" in item for item in result.errors)


def test_check_warns_when_reconfigure_is_more_frequent_than_a_wave():
    cfg = OnlineRLConfig.from_yaml(ONLINE_RL_CONFIG_PATH)
    cfg.num_envs = 8
    cfg.reconfigure_every_episodes = 2
    result = check_online_rl(cfg)
    assert result.ok, result.errors
    assert any("below num_envs" in item for item in result.warnings)


def test_check_warns_when_reconfigure_set_for_single_env():
    cfg = OnlineRLConfig.from_yaml(ONLINE_RL_CONFIG_PATH)
    cfg.num_envs = 1
    cfg.reconfigure_every_episodes = 16
    result = check_online_rl(cfg)
    assert result.ok, result.errors
    assert any("ignored for num_envs=1" in item for item in result.warnings)


def test_smoke_overrides_are_throwaway_and_mock():
    cfg = apply_smoke_overrides(OnlineRLConfig.from_yaml(ONLINE_RL_CONFIG_PATH))
    assert cfg.mock_env is True
    assert cfg.mock_vla is True
    assert cfg.device == "cpu"
    assert cfg.chunk_len == ONLINE_RL_SMOKE_CHUNK_LEN
    assert cfg.total_env_steps == ONLINE_RL_SMOKE_ENV_STEPS
    # The mock episodes must succeed, otherwise the smoke never exercises the
    # sparse reward -> TD -> episode metric path.
    assert cfg.mock_success_at == ONLINE_RL_SMOKE_SUCCESS_AT
    assert cfg.mock_success_at < cfg.max_episode_steps
    assert cfg.num_envs == 1
    assert cfg.reconfigure_every_episodes == 0
    # The smoke must also reach the post-warmup offline warm-start.
    assert cfg.warmup_env_steps >= cfg.batch_size * cfg.chunk_len
    assert cfg.total_env_steps > cfg.warmup_env_steps
    assert cfg.job_name == ONLINE_RL_SMOKE_JOB_NAME
    assert is_throwaway_smoke_output(cfg.output_dir)
    assert Path(cfg.output_dir).resolve() != ONLINE_RL_OUTPUT_DIR.resolve()
    result = check_online_rl(cfg)
    assert result.ok, result.errors


def test_throwaway_smoke_output_includes_tagged_child():
    child = ONLINE_RL_SMOKE_OUTPUT_DIR / "gpu_smoke_env32_20260901_164334"
    assert is_throwaway_smoke_output(ONLINE_RL_SMOKE_OUTPUT_DIR)
    assert is_throwaway_smoke_output(child)
    assert is_smoke_output_root(ONLINE_RL_SMOKE_OUTPUT_DIR)
    assert not is_smoke_output_root(child)
    assert not is_throwaway_smoke_output("/tmp/not-smoke")


def test_smoke_run_tag_distinguishes_mock_and_gpu():
    mock = OnlineRLConfig(mock_env=True, mock_vla=True, num_envs=1)
    gpu = OnlineRLConfig(mock_env=False, mock_vla=False, num_envs=32)
    assert smoke_run_tag(mock) == "mock_smoke"
    assert smoke_run_tag(gpu) == "gpu_smoke_env32"


def test_allocate_formal_run_dir_only_for_parent(tmp_path: Path):
    when = datetime(2026, 9, 2, 1, 2, 3, tzinfo=timezone.utc)
    cfg = OnlineRLConfig(output_dir=str(tmp_path), job_name="smolvla_rltoken_stage2")
    allocate_formal_run_dir(cfg, when=when, root=tmp_path)
    assert Path(cfg.output_dir) == tmp_path / "run_20260902_010203"
    assert Path(cfg.output_dir).is_dir()
    smoke = OnlineRLConfig(output_dir=str(ONLINE_RL_SMOKE_OUTPUT_DIR))
    allocate_formal_run_dir(smoke, when=when, root=tmp_path)
    assert Path(smoke.output_dir).resolve() == ONLINE_RL_SMOKE_OUTPUT_DIR.resolve()


def test_allocate_smoke_run_dir_tags_and_keeps_siblings(tmp_path: Path):
    when = datetime(2026, 9, 2, 1, 2, 3, tzinfo=timezone.utc)
    mock = OnlineRLConfig(output_dir=str(tmp_path), mock_env=True, mock_vla=True)
    allocate_smoke_run_dir(mock, when=when, root=tmp_path)
    mock_dir = Path(mock.output_dir)
    assert mock_dir == tmp_path / "mock_smoke_20260902_010203"
    assert mock_dir.is_dir()
    assert mock.job_name == "mock_smoke_20260902_010203"

    gpu4 = OnlineRLConfig(output_dir=str(tmp_path), mock_env=False, mock_vla=False, num_envs=4)
    allocate_smoke_run_dir(gpu4, when=when, root=tmp_path)
    gpu32 = OnlineRLConfig(output_dir=str(tmp_path), mock_env=False, mock_vla=False, num_envs=32)
    allocate_smoke_run_dir(gpu32, when=when, root=tmp_path)
    assert Path(gpu4.output_dir) == tmp_path / "gpu_smoke_env4_20260902_010203"
    assert Path(gpu32.output_dir) == tmp_path / "gpu_smoke_env32_20260902_010203"
    assert mock_dir.is_dir()
    assert Path(gpu4.output_dir).is_dir()
    assert Path(gpu32.output_dir).is_dir()
    already = OnlineRLConfig(output_dir=str(gpu32.output_dir), num_envs=32)
    allocate_smoke_run_dir(already, when=when, root=tmp_path)
    assert Path(already.output_dir) == Path(gpu32.output_dir)


def test_cli_smoke_does_not_keep_formal_output_dir():
    cfg = apply_cli_overrides(OnlineRLConfig.from_yaml(ONLINE_RL_CONFIG_PATH), _cli_ns(smoke=True))
    assert is_throwaway_smoke_output(cfg.output_dir)


def test_gpu_smoke_overrides_are_throwaway_real_and_parallel():
    cfg = apply_gpu_smoke_overrides(OnlineRLConfig.from_yaml(ONLINE_RL_CONFIG_PATH))
    assert cfg.mock_env is False
    assert cfg.mock_vla is False
    assert cfg.device == "cuda"
    assert cfg.num_envs == ONLINE_RL_GPU_SMOKE_NUM_ENVS
    assert cfg.chunk_len == 10
    assert cfg.vla_horizon == 50
    assert cfg.max_episode_steps == 200
    assert cfg.job_name == ONLINE_RL_GPU_SMOKE_JOB_NAME
    assert cfg.total_env_steps == cfg.num_envs * cfg.chunk_len * ONLINE_RL_GPU_SMOKE_CHUNKS
    assert (
        cfg.warmup_env_steps
        == cfg.num_envs * cfg.chunk_len * ONLINE_RL_GPU_SMOKE_WARMUP_CHUNKS
    )
    # Two warmup rounds so the late-written buffer holds a full batch when
    # warmup ends; the GPU smoke must reach the offline warm-start.
    assert cfg.warmup_env_steps >= cfg.batch_size * cfg.chunk_len
    assert is_throwaway_smoke_output(cfg.output_dir)
    # Parallel collect must carry a reconfigure schedule or geometry stays frozen.
    assert cfg.reconfigure_every_episodes == cfg.num_envs
    result = check_online_rl(cfg)
    assert result.ok, result.errors


def test_cli_gpu_smoke_num_envs_override():
    cfg = apply_cli_overrides(
        OnlineRLConfig.from_yaml(ONLINE_RL_CONFIG_PATH),
        _cli_ns(gpu_smoke=True, num_envs=8),
    )
    assert cfg.num_envs == 8
    assert cfg.total_env_steps == 8 * 10 * ONLINE_RL_GPU_SMOKE_CHUNKS
    assert cfg.warmup_env_steps == 8 * 10 * ONLINE_RL_GPU_SMOKE_WARMUP_CHUNKS
    assert is_throwaway_smoke_output(cfg.output_dir)


def _mock_train_config(tmp_path: Path, **overrides) -> OnlineRLConfig:
    payload = dict(
        output_dir=str(tmp_path / "run"),
        mock_env=True,
        mock_vla=True,
        mock_success_at=7,
        device="cpu",
        wandb_enable=False,
        chunk_len=4,
        action_dim=8,
        proprio_dim=9,
        rl_token_dim=8,
        hidden_dim=16,
        n_layers=2,
        max_episode_steps=12,
        total_env_steps=60,
        warmup_env_steps=24,
        offline_updates_after_warmup=3,
        batch_size=4,
        buffer_capacity=64,
        utd=2,
        log_freq=12,
        save_freq=60,
        stride=1,
        use_residual_actor=False,
        human_intervention=False,
        num_envs=1,
        reward="sparse_success",
        task=TASK_PROMPT,
    )
    payload.update(overrides)
    return OnlineRLConfig(**payload)


def _episode_records(output_dir: Path) -> list[dict]:
    path = output_dir / EPISODES_FILENAME
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_mock_train_warmup_then_actor(tmp_path: Path):
    cfg = _mock_train_config(tmp_path)
    result = train_online_rl(cfg)
    assert result.env_steps >= cfg.total_env_steps
    assert result.warmup_reference_matched is True
    assert result.used_actor is True
    assert result.did_offline is True
    assert result.offline_updates == cfg.offline_updates_after_warmup
    assert result.buffer_size >= 1
    assert (tmp_path / "run" / "online_rl.pt").is_file()
    saved = torch.load(tmp_path / "run" / "online_rl.pt", map_location="cpu", weights_only=False)
    assert "agent" in saved
    assert saved["config"]["use_residual_actor"] is False
    assert saved["config"]["stride"] == 1
    assert Path(result.output_dir).resolve() != ONLINE_RL_OUTPUT_DIR.resolve()


def test_mock_train_records_episode_success_metrics(tmp_path: Path):
    cfg = _mock_train_config(tmp_path)
    result = train_online_rl(cfg)
    assert result.episodes > 0
    assert result.successes == result.episodes  # the mock env always succeeds
    assert result.vla_episodes > 0
    assert result.actor_episodes > 0
    assert result.vla_episodes + result.actor_episodes == result.episodes
    assert result.vla_successes + result.actor_successes == result.successes

    records = _episode_records(tmp_path / "run")
    assert len(records) == result.episodes
    assert {record["use_actor"] for record in records} == {False, True}
    assert all(record["success"] is True for record in records)
    assert all(record["steps"] == cfg.mock_success_at for record in records)
    assert all(record["terminated"] is True for record in records)
    assert [record["env_steps"] for record in records] == sorted(
        record["env_steps"] for record in records
    )


def test_mock_train_reports_skipped_offline_warm_start(tmp_path: Path, capsys):
    # Warmup shorter than one batch of transitions: the warm-start cannot run and
    # must be reported instead of silently claiming success.
    cfg = _mock_train_config(
        tmp_path,
        mock_success_at=None,
        warmup_env_steps=4,
        batch_size=8,
        total_env_steps=16,
    )
    result = train_online_rl(cfg)
    assert result.offline_updates == 0
    assert result.did_offline is False
    assert "offline warm-start skipped" in capsys.readouterr().out


def test_mock_train_parallel_envs(tmp_path: Path):
    cfg = _mock_train_config(
        tmp_path,
        num_envs=2,
        max_episode_steps=20,
        total_env_steps=120,
        warmup_env_steps=48,
        log_freq=24,
        save_freq=120,
    )
    result = train_online_rl(cfg)
    assert result.env_steps >= cfg.total_env_steps
    assert result.used_actor is True
    assert result.warmup_reference_matched is True
    assert result.did_offline is True
    assert result.buffer_size >= 2
    assert result.episodes > 0
    assert len(_episode_records(tmp_path / "run")) == result.episodes
    assert (tmp_path / "run" / "online_rl.pt").is_file()


def test_gradient_steps_scale_with_parallel_envs(tmp_path: Path):
    # UTD is per transition. N envs add N transitions per chunk, so the gradient
    # count must scale with N instead of staying fixed per collect call.
    def run(name: str, num_envs: int) -> int:
        cfg = _mock_train_config(
            tmp_path / name,
            num_envs=num_envs,
            mock_success_at=None,
            max_episode_steps=1000,  # no episode ends: added == num_envs per chunk
            chunk_len=4,
            batch_size=1,
            utd=3,
            # Scale the horizon with num_envs so both runs execute the same number
            # of chunks per env and the warmup boundary lands on the same round.
            warmup_env_steps=8 * num_envs,
            total_env_steps=40 * num_envs,
            log_freq=10_000,
            save_freq=0,
        )
        return train_online_rl(cfg).gradient_steps

    single = run("one", 1)
    parallel = run("four", 4)
    assert single > 0
    assert parallel == 4 * single


def test_mock_train_parallel_reconfigures_and_keeps_ids_unique(tmp_path: Path):
    cfg = _mock_train_config(
        tmp_path,
        num_envs=4,
        mock_success_at=5,
        max_episode_steps=20,
        chunk_len=4,
        reconfigure_every_episodes=4,
        total_env_steps=200,
        warmup_env_steps=40,
        batch_size=4,
        log_freq=1000,
        save_freq=0,
    )
    result = train_online_rl(cfg)
    assert result.episodes > 4
    # One reconfigure for the initial reset plus at least one periodic refresh.
    assert result.reconfigures >= 2
    records = _episode_records(tmp_path / "run")
    ids = [record["episode_id"] for record in records]
    assert len(ids) == len(set(ids)), ids
    assert set(record["env_index"] for record in records) == {0, 1, 2, 3}


def test_episode_tracker_splits_vla_and_actor_rates(tmp_path: Path):
    from smolvla_rltoken.rollout.collector import EpisodeOutcome

    tracker = EpisodeTracker(path=tmp_path / EPISODES_FILENAME, window=2)
    outcomes = [
        EpisodeOutcome(0, 10, True, True, False, False),
        EpisodeOutcome(1, 20, False, False, True, False),
        EpisodeOutcome(2, 30, True, True, False, True),
        EpisodeOutcome(3, 40, True, True, False, True),
    ]
    for index, outcome in enumerate(outcomes):
        tracker.record(outcome, env_steps=index)
    metrics = tracker.metrics()
    assert metrics["episodes"] == 4.0
    assert metrics["successes"] == 3.0
    assert metrics["vla_success_rate"] == 0.5
    assert metrics["actor_success_rate"] == 1.0
    assert metrics["success_rate_2"] == 1.0
    assert metrics["episode_steps_2"] == 35.0
    assert len(_episode_records(tmp_path)) == 4


def test_online_rl_launcher_script():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts" / "train_online_rl.sh").read_text()
    assert "CURSOR_AGENT" in text
    assert "CODEX_SESSION_ID" in text
    assert "CODEX_THREAD_ID" in text
    assert "CODEX_INTERNAL_ORIGINATOR_OVERRIDE" in text
    assert "process ancestor codex" in text
    assert "train_online_rl.py" in text
    assert "conda activate smolvla-rlt" in text
    assert "--check" in text
    assert "--smoke" in text
    assert "--gpu-smoke" in text
    launcher = (root / "scripts" / "train_online_rl.py").read_text()
    assert "allocate_smoke_run_dir" in launcher
    assert "shutil.rmtree" not in launcher
    assert "removing throwaway dir" not in launcher


def test_online_rl_launcher_refuses_agent_owned_official_run():
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["CODEX_SESSION_ID"] = "test-agent-session"
    result = subprocess.run(
        ["bash", str(root / "scripts" / "train_online_rl.sh")],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 1
    assert "Refusing to start official Stage 2" in result.stderr
    assert "environment marker" in result.stderr
    assert "AutoDL web terminal" in result.stderr
