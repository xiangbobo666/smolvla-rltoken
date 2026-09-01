#!/usr/bin/env bash
# Stage 1: RL Token encoder/decoder on PegInsertion (forwards to scripts/train_rltoken.py).
# Activate conda env smolvla-rlt first. Use --check on CPU-only machines.
# GPU smoke (after SFT releases the GPU): bash scripts/train_rltoken.sh --smoke
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:18082}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:18082}"

exec python "$ROOT/scripts/train_rltoken.py" "$@"
