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
  (`/root/miniconda3/envs/smolvla-rlt`; Python 3.11; `lerobot[smolvla]==0.4.4`).
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
  (1000 episodes / 149,055 frames; cameras `environment_camera`, `hand_camera`, `insertion_camera`. Switched locally on 2026-08-30. Hugging Face `wkal/smolvla-rlt` still holds the previous two-camera `128×128` set until it is re-uploaded; SFT and Stage 1 must use this local `--dataset.root`, not a Hub download.)
- **Previous two-camera `128×128` LeRobot dataset (local rollback backup):**
  `/root/autodl-tmp/smolvla-rltoken/data/lerobot/PegInsertionSide-v1/motionplanning_rgb_pd_joint_pos.previous_128px_20260830`
  (1000 episodes / 149,055 frames; cameras `base_camera`, `hand_camera`. Restore by renaming this directory back to `motionplanning_rgb_pd_joint_pos` after moving the current `512×512` directory aside.)
- **SmolVLA SFT output (PegInsertion):**
  `/root/autodl-tmp/smolvla-rltoken/outputs/sft/peg_insertion`
- **RL Token Stage 1 checkpoint:**
  `/root/autodl-tmp/smolvla-rltoken/outputs/rl_token/rl_token.pt`

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
