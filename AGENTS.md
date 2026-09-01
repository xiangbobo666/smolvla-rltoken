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

Stage 1 training status (what is adapted vs still deferred) lives in
`docs/rltoken-training/`.

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
