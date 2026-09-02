# Project instructions

## Download policy

For every future download, including packages, datasets, model weights, and
other artifacts, use the local HTTP proxy:

```bash
HTTP_PROXY=http://127.0.0.1:18082
HTTPS_PROXY=http://127.0.0.1:18082
```

Download directly from official upstream sources. Do not use domestic mirrors,
including Aliyun, Tsinghua, or similar mirror services.

## Storage layout

Store all large artifacts—including datasets, checkpoints, model weights,
download caches, and generated training outputs—on the data disk under
`/root/autodl-tmp`, not under the system disk or home directory. The repository
itself is located directly at `/root/autodl-tmp/smolvla-rltoken`; keep project
source, configuration, and other small repository files there.

## Runtime environment and local resources

- **Conda environment:** `smolvla-rlt`
  (`/root/miniconda3/envs/smolvla-rlt`; Python 3.11; `lerobot[smolvla]==0.4.4`;
  `torch==2.8.0+cu128`; `torchcodec==0.7.0` — must stay on 0.7.x with this torch;
  0.10.x does not load).
  Activate it with `conda activate smolvla-rlt` before running project commands.
- **Package download cache:**
  `/root/autodl-tmp/smolvla-rltoken/.cache/pip`
- **Hugging Face Hub cache:**
  `/root/autodl-tmp/smolvla-rltoken/.cache/huggingface`
- **SmolVLA model:**
  `/root/autodl-tmp/smolvla-rltoken/models/lerobot/smolvla_base`
- **ManiSkill PegInsertionSide-v1 demonstrations:**
  - Motion-planning data:
    `/root/autodl-tmp/smolvla-rltoken/data/maniskill/demos/PegInsertionSide-v1/motionplanning`
  - RL data:
    `/root/autodl-tmp/smolvla-rltoken/data/maniskill/demos/PegInsertionSide-v1/rl`
- **Legacy PegInsertionSide-v1 RGB replay (motion-planning, `pd_joint_pos`):**
  `/root/autodl-tmp/smolvla-rltoken/data/maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.rgb.pd_joint_pos.physx_cpu.h5`
  (the earlier two-camera `128×128` replay; do not use it for the current three-camera SFT set).
- **LeRobot training dataset (motion-planning, `pd_joint_pos`, three-camera `512×512`):**
  `/root/autodl-tmp/smolvla-rltoken/data/lerobot/PegInsertionSide-v1/motionplanning_rgb_pd_joint_pos`
  (1000 episodes / 149,055 frames; cameras `environment_camera`, `hand_camera`, `insertion_camera`. Published at `wkal/smolvla-rlt` on Hugging Face. The previous two-camera `128×128` LeRobot set has been removed locally and replaced on the Hub.)
- **SmolVLA SFT output (PegInsertion):**
  `/root/autodl-tmp/smolvla-rltoken/outputs/sft/peg_insertion`
  Start the official 20k-step run from the AutoDL web terminal with
  `bash scripts/train_sft.sh`. Do not launch a long SFT from a Cursor or Codex
  agent shell: agent-owned processes are killed after about 20 minutes. The
  launcher rejects known agent environments; `--check` and `--smoke` remain
  available there.
- **SmolVLA SFT last weights (LeRobot `pretrained_model`; Stage 1 `--checkpoint`):**
  `/root/autodl-tmp/smolvla-rltoken/outputs/sft/peg_insertion/checkpoints/last/pretrained_model`
  (created when SFT saves; until then Stage 1 `--check` can use `smolvla_base`)
- **RL Token Stage 1 output root:**
  `/root/autodl-tmp/smolvla-rltoken/outputs/rl_token`
  Formal training writes a new `run_YYYYMMDD_HHMMSS/` subdirectory (checkpoint
  `rl_token.pt`, optional `rl_token_best.pt`). Do not write into the parent.
- **RL Token Stage 1 first completed run (5000 step, train `loss_ro` only, no val split):**
  `/root/autodl-tmp/smolvla-rltoken/outputs/rl_token/run_20260901_204708/rl_token.pt`
- **RL Token Stage 1 val run (interrupted ~3300/5000, use this for Stage 2):**
  `/root/autodl-tmp/smolvla-rltoken/outputs/rl_token/run_20260901_145328/rl_token_best.pt`
  (`val_loss_ro≈0.0747` at step 3300). Latest-at-interrupt is `rl_token.pt` (step 3000).
  Subsequent RL Token / Stage 2 loading uses **best**, not latest.
- **Stage 2 online RL config:**
  `/root/autodl-tmp/smolvla-rltoken/configs/rl/actor_critic.yaml`
  CPU preflight `python scripts/train_online_rl.py --check`. Mock smoke
  `python scripts/train_online_rl.py --smoke`. GPU smoke with parallel envs
  `python scripts/train_online_rl.py --gpu-smoke` (default `num_envs=4`;
  override `--num-envs`). Official train from the AutoDL web terminal with
  `bash scripts/train_online_rl.sh`; the launcher rejects Cursor/Codex agent
  shells. Do not start a long Stage 2 run from an agent, and do not load
  SmolVLA/ManiSkill on the GPU while Stage 1 still holds it.
  `--check` prints the per-dimension normalized action bounds it derived from
  the checkpoint; confirm they are not +/-1 before starting a formal run.
- **Stage 2 online RL smoke parent (tagged per-run dirs, not a dump):**
  `/root/autodl-tmp/smolvla-rltoken/outputs/online_rl_smoke`
  Mock `--smoke` writes `mock_smoke_YYYYMMDD_HHMMSS/`. GPU `--gpu-smoke`
  writes `gpu_smoke_envN_YYYYMMDD_HHMMSS/` and `benchamrk/rl/` resource
  Markdown. Do not write checkpoints into the parent; do not delete sibling
  runs. Formal training defaults to `num_envs=16` and
  `total_env_steps=1000000` under `outputs/online_rl/` (`reconfigure_every_episodes`
  must stay >= `num_envs`).
- **Stage 2 online RL formal output root:**
  `/root/autodl-tmp/smolvla-rltoken/outputs/online_rl`
  Formal training writes a new `run_YYYYMMDD_HHMMSS/` subdirectory holding
  `online_rl.pt` and `episodes.jsonl` (per-episode success / steps / whether the
  Actor was in control). Do not write checkpoints into the parent. Evaluation
  Markdown for later RL evals belongs under `benchamrk/rl/` (not implemented
  yet).
- **Benchmark Markdown summaries (repository records):**
  `/root/autodl-tmp/smolvla-rltoken/benchamrk`
  (the directory name intentionally uses the existing `benchamrk` spelling)

Treat this section as the authoritative local-resource registry. When changing
the Conda environment or adding, moving, replacing, or removing a dataset,
model, checkpoint, or cache that future agents need, update the exact paths in
this section as part of the same change. Verify the replacement path exists
before recording it, and remove or clearly mark obsolete entries so agents do
not search for stale resources.

## RLT knowledge base

The repository-owned RLT knowledge base is located in `docs/knowledge-base/`.
It is an independent copy; do not assume that it is synchronized with the
author's Obsidian vault.

Stage 1 training status (what is adapted vs still deferred) lives in
`docs/rltoken-training/`.

Stage 2 (online chunk-level Actor-Critic) survey and locked user decisions
live in `docs/online-rl/`. Read `docs/online-rl/stage2_survey.md` before
implementing rollout, replay, actor, or critic. That file overrides older
plan defaults where they disagree (non-residual actor, no stride in V1,
sparse `info["success"]` reward, full-episode control after warmup). Sections
13.1 and 13.2 override the rest of that file:

- 13.1 (post-review corrections): the normalized action space is `MEAN_STD`, so
  Actor clipping uses per-dimension bounds derived from the checkpoint (never a
  hard-coded +/-1) and the warmup reference is executed unclipped;
  `warmup_env_steps` must be at least `batch_size * chunk_len`; the training
  loop must report episode success rates.
- 13.2 (parallel envs unlocked, replacing the original "V1 single env"):
  `num_envs > 1` is allowed for formal training. UTD is accounted per
  transition, finished envs restart via ManiSkill partial reset, and
  `episode_id` is globally unique. `num_envs > 1` additionally REQUIRES
  `reconfigure_every_episodes > 0`, because ManiSkill uses
  `reconfiguration_freq=0` for parallel envs and PegInsertion only randomizes
  peg geometry during reconfiguration.

Before making or reviewing changes related to system design, algorithms,
training objectives, actor-critic behavior, or data pipelines, read the
relevant notes in `docs/knowledge-base/`. In particular:

- Use `SmolVLA_RLT_plan.md` for the overall design and implementation plan.
- Use `算法实现.md` and the notes under `详解/` for algorithm and loss details.
- Use `数据管线.md` for dataset and data-pipeline behavior.

Treat these notes as project context rather than unquestionable truth. Resolve
disagreements in favor of verified code behavior, tests, experiment results,
and explicit user decisions, and record important discrepancies in the
relevant note.

When an implementation change makes the repository knowledge base inaccurate,
update the corresponding note in the same change. Keep relative links and
files under `附件/` intact when moving or renaming notes.

## Evaluation benchmark records

Every evaluation run must produce a Markdown summary under the repository's
`benchamrk/` directory in addition to raw machine-readable artifacts under
`outputs/`. This rule applies to the SFT baseline and all later RL Token (RLT),
RL, ablation, regression, and comparison evaluations.

- Organize summaries by stage, for example `benchamrk/sft/`,
  `benchamrk/rlt/`, and `benchamrk/rl/`.
- Create or update the Markdown record while an evaluation is running and mark
  its final status explicitly as `complete`, `failed`, or `interrupted`.
- Include at minimum: UTC time, stage/method, model and resolved checkpoint,
  task/environment, success definition, seed set, episode count, parallel
  environment count, max steps, action-chunk execution settings, relevant
  configuration, aggregate metrics, termination counts, per-episode results,
  and links to JSON/JSONL/videos or other raw artifacts.
- Evaluation code is not complete unless this Markdown output is implemented
  and covered by a CPU test. Future RLT and RL evaluators must follow the same
  record format rather than leaving results only in logs, W&B, or JSON files.

## Reference RLT repositories

The following sibling repositories under `/root/autodl-tmp/` are related RL
Token (RLT) work. Use them as references for techniques, code structure, and
implementation details when building or reviewing this project. Do not copy
them blindly: adapt to this repository's design, tests, configs, and explicit
user decisions. When a sibling repo disagrees with this repo's knowledge base
or verified behavior, prefer this repo.

- **`/root/autodl-tmp/RL-Token-SmolVLA`**
  (`https://github.com/RajatDandekar/RL-Token-SmolVLA`): RLT on SmolVLA /
  LeRobot v0.4.4 for SO-101, including the `smolvla_rlt` policy module,
  LeRobot patches, and Stage 1 / Stage 2 training scripts.
- **`/root/autodl-tmp/rlt-openpi`**
  (`https://github.com/yknxh/rlt-openpi`): RLT on OpenPI public checkpoints,
  including encoder/decoder, residual actor, twin Q-critic, Stage 1 / Stage 2
  trainers, replay buffer, and VLA embedding hooks.
- **`/root/autodl-tmp/smollvla_rltoken`**
  (`https://github.com/afengleafs/smollvla_rltoken`): another SmolVLA RLT
  reproduction (LeRobot 0.5.1), useful for SmolVLM prefix embedding extraction,
  RL-token sizing, and actor-critic / replay / online-training correspondence
  with a PI0.5-style implementation.
