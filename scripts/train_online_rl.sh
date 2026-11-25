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

export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

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
  local conda_sh
  conda_sh="$(_resolve_conda_sh)" || {
    echo "Conda initialization script not found. Set CONDA_SH or install miniforge3/miniconda3." >&2
    exit 1
  }
  # shellcheck source=/dev/null
  source "$conda_sh"
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
