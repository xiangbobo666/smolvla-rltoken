#!/usr/bin/env bash
# Stage 0: SmolVLA SFT on PegInsertion (forwards to scripts/train_sft.py).
# Activate conda env smolvla-rlt first. Use --check on CPU-only machines.
# GPU smoke: bash scripts/train_sft.sh --smoke
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:18082}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:18082}"

exec python "$ROOT/scripts/train_sft.py" "$@"
