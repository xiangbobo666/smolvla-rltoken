#!/usr/bin/env bash
# Stage 2: online chunk-level Actor-Critic (forwards to scripts/train_online_rl.py).
#
# AutoDL web terminal (official run):
#   bash scripts/train_online_rl.sh
# Do not start the official run from a Cursor/Codex agent shell.
#
# Foreground helpers (allowed in agent shells):
#   bash scripts/train_online_rl.sh --check
#   bash scripts/train_online_rl.sh --smoke
#   bash scripts/train_online_rl.sh --gpu-smoke
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"

export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:18082}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:18082}"

CHECK=0
SMOKE=0
GPU_SMOKE=0
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --check) CHECK=1 ;;
    --smoke) SMOKE=1 ;;
    --gpu-smoke) GPU_SMOKE=1 ;;
    *) EXTRA+=("$arg") ;;
  esac
done

_activate() {
  # shellcheck source=/dev/null
  source "$CONDA_SH"
  conda activate smolvla-rlt
}

_fill_python_args() {
  PY_ARGS=()
  if [[ "$CHECK" -eq 1 ]]; then
    PY_ARGS+=(--check)
  fi
  if [[ "$SMOKE" -eq 1 ]]; then
    PY_ARGS+=(--smoke)
  fi
  if [[ "$GPU_SMOKE" -eq 1 ]]; then
    PY_ARGS+=(--gpu-smoke)
  fi
  if [[ "${#EXTRA[@]}" -gt 0 ]]; then
    PY_ARGS+=("${EXTRA[@]}")
  fi
}

_agent_shell_reason() {
  # Long-running descendants started by an IDE coding agent retain external
  # ownership metadata even after nohup/setsid.  The agent runner later kills
  # them when its turn ends.  Marker names vary between Cursor and Codex, so a
  # single CURSOR_AGENT check is not sufficient.
  local marker
  for marker in \
    CURSOR_AGENT CURSOR_CONVERSATION_ID AGENT_TRANSCRIPTS \
    CODEX_SESSION_ID CODEX_THREAD_ID CODEX_CI \
    CODEX_INTERNAL_ORIGINATOR_OVERRIDE \
    __CURSOR_SANDBOX_ENV_RESTORE; do
    if [[ -n "${!marker:-}" ]]; then
      echo "environment marker $marker"
      return 0
    fi
  done

  local pid="$PPID"
  local command_line
  local command_lower
  local depth=0
  while [[ "$pid" =~ ^[0-9]+$ ]] && (( pid > 1 && depth < 16 )); do
    command_line="$(ps -o args= -p "$pid" 2>/dev/null || true)"
    command_lower="${command_line,,}"
    case "$command_lower" in
      *"/codex "*)
        echo "process ancestor codex"
        return 0
        ;;
      *"openai.chatgpt-"*)
        echo "process ancestor openai.chatgpt"
        return 0
        ;;
      *"cursor"*"agent"*)
        echo "process ancestor cursor-agent"
        return 0
        ;;
    esac
    pid="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    ((depth += 1))
  done
  return 1
}

if [[ "$CHECK" -eq 1 || "$SMOKE" -eq 1 || "$GPU_SMOKE" -eq 1 ]]; then
  _activate
  cd "$ROOT"
  _fill_python_args
  exec python "$ROOT/scripts/train_online_rl.py" "${PY_ARGS[@]}"
fi

if agent_reason="$(_agent_shell_reason)"; then
  echo "Refusing to start official Stage 2 online RL from a Cursor/Codex agent shell ($agent_reason)." >&2
  echo "Agent-owned processes are externally SIGKILLed after roughly 20 minutes; nohup/setsid cannot detach that ownership." >&2
  echo "Use the AutoDL web terminal:" >&2
  echo "  cd $ROOT && bash scripts/train_online_rl.sh" >&2
  exit 1
fi

_activate
cd "$ROOT"
_fill_python_args
exec python "$ROOT/scripts/train_online_rl.py" "${PY_ARGS[@]}"
