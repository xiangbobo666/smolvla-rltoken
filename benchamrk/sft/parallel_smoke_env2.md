# SFT benchmark: parallel_smoke_env2

## Run

| Field | Value |
|---|---|
| Status | complete |
| Updated (UTC) | 2026-09-01T12:10:05.056753+00:00 |
| Stage | sft |
| Output artifacts | [outputs/eval/parallel_smoke_env2](../../outputs/eval/parallel_smoke_env2) |
| Raw summary | [summary.json](../../outputs/eval/parallel_smoke_env2/summary.json) |
| Per-episode JSONL | [episodes.jsonl](../../outputs/eval/parallel_smoke_env2/episodes.jsonl) |
| Run config | [run_config.json](../../outputs/eval/parallel_smoke_env2/run_config.json) |

## Configuration

| Parameter | Value |
|---|---|
| camera_config | /root/autodl-tmp/smolvla-rltoken/configs/vla/peg_insertion_three_cameras.json |
| camera_config_sha256 | 53a117271c77f4470ac0aae677f65a209e1be10f705a6bb90dde392732c8c2fa |
| checkpoint | /root/autodl-tmp/smolvla-rltoken/outputs/sft/peg_insertion/checkpoints/last/pretrained_model |
| chunk_execution_ratio_requested | 0.7 |
| device | cuda |
| fps | 20 |
| max_steps | 2 |
| num_envs | 2 |
| record_episodes | 0 |
| resolved_checkpoint | /root/autodl-tmp/smolvla-rltoken/outputs/sft/peg_insertion/checkpoints/020000/pretrained_model |
| seeds | `[10000, 10001]` |
| sim_backend | physx_cuda |
| started_at_utc | 2026-09-01T12:05:44.672753+00:00 |
| success_signal | ManiSkill info.success |
| task | Insert the peg into the hole from the side. |

## Aggregate metrics

| Metric | Value |
|---|---|
| episodes | 2 |
| failures | 2 |
| mean_episode_steps | 2 |
| mean_success_steps | — |
| median_episode_steps | 2 |
| success_rate | 0 |
| success_rate_percent | 0 |
| success_rate_wilson_95 | `[0.0, 0.6576197724933469]` |
| successes | 0 |
| termination_counts | `{"time_limit": 2}` |

## Episodes

| Episode | Seed | Success | Steps | Termination | Reward sum | Video |
|---:|---:|:---:|---:|---|---:|---|
| 0 | 10000 | ❌ | 2 | time_limit | 0.74913 | — |
| 1 | 10001 | ❌ | 2 | time_limit | 0.290695 | — |
