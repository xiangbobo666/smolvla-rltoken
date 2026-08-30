#!/usr/bin/env bash
# Stage 0: SmolVLA SFT on PegInsertion (LeRobot train loop, not a custom trainer).
set -euo pipefail

ROOT="/root/autodl-tmp/smolvla-rltoken"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:18082}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:18082}"

POLICY_PATH="${POLICY_PATH:-$ROOT/models/lerobot/smolvla_base}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT/data/lerobot/PegInsertionSide-v1/motionplanning_rgb_pd_joint_pos}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/sft/peg_insertion}"

mkdir -p "$OUTPUT_DIR"

exec lerobot-train \
  --policy.path="$POLICY_PATH" \
  --policy.push_to_hub=false \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --dataset.repo_id=wkal/smolvla-rlt \
  --dataset.root="$DATASET_ROOT" \
  --output_dir="$OUTPUT_DIR" \
  --batch_size=8 \
  --steps=20000 \
  --save_freq=5000 \
  --log_freq=200 \
  --num_workers=4 \
  --job_name=smolvla_sft_peg_insertion \
  --rename_map='{"observation.images.environment_camera": "observation.images.camera1", "observation.images.hand_camera": "observation.images.camera2", "observation.images.insertion_camera": "observation.images.camera3"}' \
  "$@"
