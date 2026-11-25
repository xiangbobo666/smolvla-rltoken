#!/usr/bin/env bash
# Evaluate the 20k-step SmolVLA SFT checkpoint on PegInsertion.
#
# CPU-only preflight:
#   bash scripts/eval_sft.sh --check
#
# One-episode pressure/smoke evaluation (uses GPU, no video):
#   bash scripts/eval_sft.sh --episodes 1 --record-successes 0 --record-failures 0
#
# Formal evaluation (100 held-out seeds, 70% = 35/50 chunk execution):
#   bash scripts/eval_sft.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

_resolve_conda_sh() {
  if [[ -n "${CONDA_SH:-}" && -f "$CONDA_SH" ]]; then
    echo "$CONDA_SH"
    return 0
  fi
  local base=""
  if command -v conda >/dev/null 2>&1; then
    base="$(conda info --base 2>/dev/null || true)"
  fi
  local candidate
  for candidate in \
    "${base:+$base/etc/profile.d/conda.sh}" \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    if [[ -n "$candidate" && -f "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

# Formal evaluation parameters. Edit these values to change the default run.
# Matching command-line options are appended last and override these defaults.
EVAL_CHECKPOINT="$ROOT/outputs/sft/peg_insertion/checkpoints/last/pretrained_model"
EVAL_CAMERA_CONFIG="$ROOT/configs/vla/peg_insertion_three_cameras.json"
EVAL_DEVICE="cuda"
EVAL_SIM_BACKEND="physx_cuda"
EVAL_NUM_ENVS=8
EVAL_EPISODES=100
EVAL_START_SEED=10000
EVAL_MAX_STEPS=200
EVAL_CHUNK_EXECUTION_RATIO=0.70
EVAL_RECORD_SUCCESSES=5
EVAL_RECORD_FAILURES=5
EVAL_TASK="Insert the peg into the hole from the side."
EVAL_OUTPUT_DIR="$ROOT/outputs/eval/sft_peg_insertion/run_$(date -u +%Y%m%d_%H%M%S)"

CONDA_SH="$(_resolve_conda_sh)" || {
  echo "Conda initialization script not found. Set CONDA_SH or install miniforge3/miniconda3." >&2
  exit 1
}

# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate smolvla-rlt

# Optional proxy: inherit from the caller environment when configured.
if [[ -n "${HTTP_PROXY:-}" ]]; then export HTTP_PROXY; fi
if [[ -n "${HTTPS_PROXY:-}" ]]; then export HTTPS_PROXY; fi
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$ROOT"
exec python "$ROOT/scripts/eval.py" \
  --checkpoint "$EVAL_CHECKPOINT" \
  --camera-config "$EVAL_CAMERA_CONFIG" \
  --device "$EVAL_DEVICE" \
  --sim-backend "$EVAL_SIM_BACKEND" \
  --num-envs "$EVAL_NUM_ENVS" \
  --episodes "$EVAL_EPISODES" \
  --start-seed "$EVAL_START_SEED" \
  --max-steps "$EVAL_MAX_STEPS" \
  --chunk-execution-ratio "$EVAL_CHUNK_EXECUTION_RATIO" \
  --record-successes "$EVAL_RECORD_SUCCESSES" \
  --record-failures "$EVAL_RECORD_FAILURES" \
  --task "$EVAL_TASK" \
  --output-dir "$EVAL_OUTPUT_DIR" \
  "$@"
