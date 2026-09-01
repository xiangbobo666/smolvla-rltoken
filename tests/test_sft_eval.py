"""CPU-only tests for automatic SFT success-rate evaluation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

import smolvla_rltoken.benchmark as benchmark_module
from smolvla_rltoken.benchmark import write_benchmark_markdown
from smolvla_rltoken.paths import SFT_IMAGE_RENAME_MAP
from smolvla_rltoken.vla.evaluation import (
    CAMERA_NAMES,
    EpisodeResult,
    OutcomeVideoBudget,
    action_steps_for_ratio,
    check_eval_inputs,
    extract_policy_observation,
    extract_policy_observation_batch,
    prepare_policy_observation,
    prepare_policy_observation_batch,
    reconcile_recorded_video,
    run_episode_batch,
    run_episode,
    summarize_results,
)


def fake_observation() -> dict:
    sensor_data = {}
    for index, camera_name in enumerate(CAMERA_NAMES):
        image = np.full((1, 4, 5, 4), (index + 1) / 4, dtype=np.float32)
        sensor_data[camera_name] = {"rgb": image}
    return {
        "agent": {"qpos": np.arange(9, dtype=np.float32)[None]},
        "sensor_data": sensor_data,
    }


def fake_batch_observation(batch_size: int) -> dict:
    single = fake_observation()
    return {
        "agent": {"qpos": np.repeat(single["agent"]["qpos"], batch_size, axis=0)},
        "sensor_data": {
            camera_name: {
                "rgb": np.repeat(single["sensor_data"][camera_name]["rgb"], batch_size, axis=0)
            }
            for camera_name in CAMERA_NAMES
        },
    }


class FakePolicyRunner:
    def __init__(self, action: np.ndarray | None = None):
        self.action = np.zeros(8, dtype=np.float32) if action is None else action
        self.reset_count = 0
        self.call_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def __call__(self, observation) -> np.ndarray:
        self.call_count += 1
        return self.action


class FakeEnv:
    def __init__(self, outcomes: list[str]):
        self.outcomes = outcomes
        self.index = 0
        self.seed = None

    def reset(self, *, seed: int):
        self.index = 0
        self.seed = seed
        return fake_observation(), {"success": np.array([False])}

    def step(self, action: np.ndarray):
        outcome = self.outcomes[self.index]
        self.index += 1
        success = outcome == "success"
        terminated = outcome in {"success", "failure"}
        truncated = outcome == "truncated"
        return (
            fake_observation(),
            np.array([1.0], dtype=np.float32),
            np.array([terminated]),
            np.array([truncated]),
            {"success": np.array([success])},
        )


class FakeBatchPolicyRunner(FakePolicyRunner):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

    def __call__(self, observation) -> np.ndarray:
        self.call_count += 1
        return np.zeros((self.batch_size, 8), dtype=np.float32)


class FakeBatchEnv:
    def __init__(self):
        self.step_index = 0
        self.reset_seeds = None
        self.reset_options = None

    def reset(self, *, seed: list[int], options: dict):
        self.step_index = 0
        self.reset_seeds = seed
        self.reset_options = options
        return fake_batch_observation(3), {"success": np.zeros(3, dtype=bool)}

    def step(self, action: np.ndarray):
        self.step_index += 1
        # env0 succeeds at step 2; env1 times out at step 3; env2 succeeds at step 1.
        success = np.array([self.step_index == 2, False, self.step_index == 1])
        terminated = success.copy()
        truncated = np.array([False, self.step_index == 3, False])
        return (
            fake_batch_observation(3),
            np.ones(3, dtype=np.float32),
            terminated,
            truncated,
            {"success": success},
        )


def test_extract_policy_observation_matches_training_features():
    extracted = extract_policy_observation(fake_observation())
    assert extracted["observation.state"].shape == (9,)
    assert extracted["observation.state"].dtype == np.float32
    for camera_name in CAMERA_NAMES:
        image = extracted[f"observation.images.{camera_name}"]
        assert image.shape == (4, 5, 3)
        assert image.dtype == np.uint8

    prepared = prepare_policy_observation(fake_observation())
    assert prepared["observation.state"].shape == (9,)
    assert prepared["observation.images.environment_camera"].shape == (3, 4, 5)
    assert prepared["task"].startswith("Insert the peg")

    extracted_batch = extract_policy_observation_batch(fake_batch_observation(3))
    prepared_batch = prepare_policy_observation_batch(fake_batch_observation(3))
    assert extracted_batch["observation.state"].shape == (3, 9)
    assert extracted_batch["observation.images.environment_camera"].shape == (3, 4, 5, 3)
    assert prepared_batch["observation.state"].shape == (3, 9)
    assert prepared_batch["observation.images.environment_camera"].shape == (3, 3, 4, 5)
    assert len(prepared_batch["task"]) == 3


def test_seventy_percent_chunk_executes_35_of_50_actions():
    assert action_steps_for_ratio(50, 0.70) == 35
    assert action_steps_for_ratio(50, 1.0) == 50
    assert action_steps_for_ratio(3, 0.01) == 1
    with pytest.raises(ValueError, match="ratio"):
        action_steps_for_ratio(50, 0.0)


def test_run_episode_stops_immediately_on_official_success():
    env = FakeEnv(["continue", "success", "continue"])
    runner = FakePolicyRunner()
    frames: list[int] = []
    result = run_episode(
        env,
        runner,
        episode_index=3,
        seed=10_003,
        max_steps=200,
        frame_callback=lambda observation, step: frames.append(step),
    )
    assert result.success is True
    assert result.steps == 2
    assert result.termination == "success"
    assert result.reward_sum == 2.0
    assert runner.reset_count == 1
    assert runner.call_count == 2
    assert frames == [0, 1, 2]


def test_run_episode_distinguishes_time_limit_and_local_max_steps():
    truncated = run_episode(
        FakeEnv(["continue", "truncated"]),
        FakePolicyRunner(),
        episode_index=0,
        seed=10_000,
        max_steps=10,
    )
    assert not truncated.success
    assert truncated.termination == "time_limit"
    assert truncated.steps == 2

    maxed = run_episode(
        FakeEnv(["continue", "continue"]),
        FakePolicyRunner(),
        episode_index=1,
        seed=10_001,
        max_steps=2,
    )
    assert not maxed.success
    assert maxed.termination == "max_steps"


def test_run_episode_rejects_wrong_action_shape():
    with pytest.raises(ValueError, match="Policy action"):
        run_episode(
            FakeEnv(["continue"]),
            FakePolicyRunner(np.zeros(7, dtype=np.float32)),
            episode_index=0,
            seed=10_000,
            max_steps=1,
        )


def test_parallel_batch_freezes_each_first_done_result():
    env = FakeBatchEnv()
    runner = FakeBatchPolicyRunner(3)
    frames = [[], [], []]
    callbacks = [
        (lambda observation, step, env_index, slot=slot: frames[slot].append((step, env_index)))
        for slot in range(3)
    ]
    results = run_episode_batch(
        env,
        runner,
        episode_indices=[0, 1, 2],
        seeds=[10_000, 10_001, 10_002],
        max_steps=10,
        frame_callbacks=callbacks,
    )
    assert env.reset_seeds == [10_000, 10_001, 10_002]
    assert env.reset_options == {"reconfigure": True}
    assert runner.reset_count == 1
    assert runner.call_count == 3
    assert [(result.success, result.steps, result.termination) for result in results] == [
        (True, 2, "success"),
        (False, 3, "time_limit"),
        (True, 1, "success"),
    ]
    assert [result.reward_sum for result in results] == [2.0, 3.0, 1.0]
    assert frames[0] == [(0, 0), (1, 0), (2, 0)]
    assert frames[1] == [(0, 1), (1, 1), (2, 1), (3, 1)]
    assert frames[2] == [(0, 2), (1, 2)]


def test_summary_counts_successes_and_reports_interval():
    results = [
        EpisodeResult(0, 10_000, True, 50, "success", 10.0),
        EpisodeResult(1, 10_001, False, 200, "time_limit", 2.0),
        EpisodeResult(2, 10_002, True, 100, "success", 10.0),
        EpisodeResult(3, 10_003, False, 40, "terminated_failure", 1.0),
    ]
    summary = summarize_results(results)
    assert summary["episodes"] == 4
    assert summary["successes"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["mean_success_steps"] == 75
    assert summary["termination_counts"] == {
        "success": 2,
        "time_limit": 1,
        "terminated_failure": 1,
    }
    lower, upper = summary["success_rate_wilson_95"]
    assert 0 < lower < 0.5 < upper < 1


def test_outcome_video_budget_keeps_first_successes_and_failures():
    budget = OutcomeVideoBudget(max_successes=2, max_failures=2)
    outcomes = [False, False, True, False, True, True, False]
    decided = []
    for index, success in enumerate(outcomes):
        result = EpisodeResult(
            episode_index=index,
            seed=10_000 + index,
            success=success,
            steps=10,
            termination="success" if success else "time_limit",
            reward_sum=1.0,
            video=f"videos/episode_{index:04d}_seed_{10000 + index}.mp4",
        )
        decided.append(budget.decide(result).video)

    assert decided == [
        "videos/failure/episode_0000_seed_10000.mp4",
        "videos/failure/episode_0001_seed_10001.mp4",
        "videos/success/episode_0002_seed_10002.mp4",
        None,
        "videos/success/episode_0004_seed_10004.mp4",
        None,
        None,
    ]
    assert not budget.still_recording()


def test_zero_outcome_video_budget_never_records():
    budget = OutcomeVideoBudget(max_successes=0, max_failures=0)
    assert not budget.still_recording()
    result = EpisodeResult(0, 10_000, False, 200, "time_limit", 1.0, video="videos/episode_0000_seed_10000.mp4")
    assert budget.decide(result).video is None


def test_outcome_video_budget_ignores_episodes_without_video():
    budget = OutcomeVideoBudget(max_successes=1, max_failures=1)
    skipped = EpisodeResult(0, 10_000, False, 200, "time_limit", 1.0, video=None)
    kept = EpisodeResult(1, 10_001, False, 200, "time_limit", 1.0, video="videos/episode_0001_seed_10001.mp4")
    assert budget.decide(skipped).video is None
    assert budget.decide(kept).video == "videos/failure/episode_0001_seed_10001.mp4"
    assert budget.kept_failures == 1


def test_reconcile_recorded_video_moves_kept_and_deletes_discarded(tmp_path: Path):
    recorded = tmp_path / "videos"
    recorded.mkdir()
    kept_source = recorded / "episode_0000_seed_10000.mp4"
    dropped_source = recorded / "episode_0001_seed_10001.mp4"
    dropped_partial = recorded / "episode_0001_seed_10001.partial.mp4"
    kept_source.write_bytes(b"keep")
    dropped_source.write_bytes(b"drop")
    dropped_partial.write_bytes(b"partial")

    kept = EpisodeResult(0, 10_000, True, 80, "success", 1.0, video="videos/episode_0000_seed_10000.mp4")
    kept_decided = EpisodeResult(
        0, 10_000, True, 80, "success", 1.0, video="videos/success/episode_0000_seed_10000.mp4"
    )
    dropped = EpisodeResult(1, 10_001, False, 200, "time_limit", 1.0, video="videos/episode_0001_seed_10001.mp4")
    dropped_decided = EpisodeResult(1, 10_001, False, 200, "time_limit", 1.0, video=None)

    reconcile_recorded_video(kept, kept_decided, tmp_path)
    reconcile_recorded_video(dropped, dropped_decided, tmp_path)

    destination = tmp_path / "videos/success/episode_0000_seed_10000.mp4"
    assert destination.read_bytes() == b"keep"
    assert not kept_source.exists()
    assert not dropped_source.exists()
    assert not dropped_partial.exists()


def test_cpu_preflight_checks_checkpoint_and_camera_contract(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "smolvla",
                "chunk_size": 50,
                "n_action_steps": 50,
                "output_features": {"action": {"shape": [8]}},
            }
        )
    )
    (checkpoint / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "rename_observations_processor",
                        "config": {"rename_map": SFT_IMAGE_RENAME_MAP},
                    }
                ]
            }
        )
    )
    (checkpoint / "policy_postprocessor.json").write_text(json.dumps({"steps": []}))
    camera_config = tmp_path / "cameras.json"
    camera_config.write_text(
        json.dumps({camera_name: {"width": 512, "height": 512} for camera_name in CAMERA_NAMES})
    )

    result = check_eval_inputs(checkpoint, camera_config)
    assert result["ok"], result["errors"]
    assert result["chunk_size"] == 50
    assert result["n_action_steps"] == 50

    camera_config.write_text(json.dumps({"environment_camera": {"width": 128, "height": 128}}))
    result = check_eval_inputs(checkpoint, camera_config)
    assert not result["ok"]
    assert any("512x512" in error for error in result["errors"])


def test_eval_shell_launcher_is_valid_and_forwards_to_python():
    root = Path(__file__).resolve().parents[1]
    launcher = root / "scripts/eval_sft.sh"
    result = subprocess.run(
        ["bash", "-n", str(launcher)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    text = launcher.read_text()
    assert "conda activate smolvla-rlt" in text
    assert 'exec python "$ROOT/scripts/eval.py"' in text
    assert 'EVAL_DEVICE="cuda"' in text
    assert 'EVAL_SIM_BACKEND="physx_cuda"' in text
    assert "EVAL_NUM_ENVS=8" in text
    assert "EVAL_EPISODES=100" in text
    assert "EVAL_START_SEED=10000" in text
    assert "EVAL_MAX_STEPS=200" in text
    assert "EVAL_CHUNK_EXECUTION_RATIO=0.70" in text
    assert "EVAL_RECORD_SUCCESSES=5" in text
    assert "EVAL_RECORD_FAILURES=5" in text
    assert "--record-successes" in text
    assert "--record-failures" in text
    assert '"$@"' in text
    assert "HTTP_PROXY" in text
    assert "HF_HOME" in text


def test_benchmark_markdown_contains_config_metrics_and_episodes(tmp_path: Path, monkeypatch):
    benchmark_root = tmp_path / "benchamrk"
    monkeypatch.setattr(benchmark_module, "BENCHAMRK_DIR", benchmark_root)
    output_dir = tmp_path / "outputs/eval/run_test"
    output_dir.mkdir(parents=True)
    for filename in ("summary.json", "episodes.jsonl", "run_config.json"):
        (output_dir / filename).write_text("{}\n")

    path = write_benchmark_markdown(
        stage="sft",
        run_name="run_test",
        status="complete",
        config={"checkpoint": "/checkpoints/020000", "num_envs": 4, "success_signal": "info.success"},
        summary={"episodes": 2, "successes": 1, "success_rate_percent": 50.0},
        episodes=[
            {
                "episode_index": 0,
                "seed": 10_000,
                "success": True,
                "steps": 80,
                "termination": "success",
                "reward_sum": 12.5,
                "video": None,
            },
            {
                "episode_index": 1,
                "seed": 10_001,
                "success": False,
                "steps": 200,
                "termination": "time_limit",
                "reward_sum": 4.0,
                "video": None,
            },
        ],
        output_dir=output_dir,
    )
    assert path == benchmark_root / "sft/run_test.md"
    text = path.read_text()
    assert "# SFT benchmark: run_test" in text
    assert "| Status | complete |" in text
    assert "| checkpoint | /checkpoints/020000 |" in text
    assert "| success_rate_percent | 50 |" in text
    assert "| 0 | 10000 | ✅ | 80 | success |" in text
    assert "| 1 | 10001 | ❌ | 200 | time_limit |" in text
