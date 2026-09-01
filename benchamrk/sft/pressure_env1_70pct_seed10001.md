# SFT benchmark: pressure_env1_70pct_seed10001

## Run

| Field | Value |
|---|---|
| Status | complete |
| Updated (UTC) | 2026-09-01T12:10:05.056189+00:00 |
| Stage | sft |
| Output artifacts | [outputs/eval/pressure_env1_70pct_seed10001](../../outputs/eval/pressure_env1_70pct_seed10001) |
| Raw summary | [summary.json](../../outputs/eval/pressure_env1_70pct_seed10001/summary.json) |
| Per-episode JSONL | [episodes.jsonl](../../outputs/eval/pressure_env1_70pct_seed10001/episodes.jsonl) |
| Run config | [run_config.json](../../outputs/eval/pressure_env1_70pct_seed10001/run_config.json) |

## Configuration

| Parameter | Value |
|---|---|
| camera_config | /root/autodl-tmp/smolvla-rltoken/configs/vla/peg_insertion_three_cameras.json |
| camera_config_sha256 | 53a117271c77f4470ac0aae677f65a209e1be10f705a6bb90dde392732c8c2fa |
| checkpoint | /root/autodl-tmp/smolvla-rltoken/outputs/sft/peg_insertion/checkpoints/last/pretrained_model |
| chunk_execution_ratio_requested | 0.7 |
| device | cuda |
| fps | 20 |
| max_steps | 200 |
| record_episodes | 0 |
| resolved_checkpoint | /root/autodl-tmp/smolvla-rltoken/outputs/sft/peg_insertion/checkpoints/020000/pretrained_model |
| seeds | `[10001]` |
| sim_backend | physx_cuda |
| started_at_utc | 2026-09-01T11:53:51.218201+00:00 |
| success_signal | ManiSkill info.success |
| task | Insert the peg into the hole from the side. |

## Aggregate metrics

| Metric | Value |
|---|---|
| episodes | 1 |
| failures | 1 |
| mean_episode_steps | 200 |
| mean_success_steps | — |
| median_episode_steps | 200 |
| success_rate | 0 |
| success_rate_percent | 0 |
| success_rate_wilson_95 | `[0.0, 0.7934506856227626]` |
| successes | 0 |
| termination_counts | `{"time_limit": 1}` |

## Episodes

| Episode | Seed | Success | Steps | Termination | Reward sum | Video |
|---:|---:|:---:|---:|---|---:|---|
| 0 | 10001 | ❌ | 200 | time_limit | 617.067 | — |
