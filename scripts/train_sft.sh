#!/usr/bin/env bash
# Stage 0: SmolVLA SFT on PegInsertion (forwards to scripts/train_sft.py).
#
# AutoDL web terminal (official 20k-step, detached):
#   bash scripts/train_sft.sh
#   bash scripts/train_sft.sh --resume
# Do not start the official run from a Cursor/Codex agent shell.
#
# Foreground helpers:
#   bash scripts/train_sft.sh --check
#   bash scripts/train_sft.sh --smoke
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${SFT_LOG:-$ROOT/outputs/sft/peg_insertion_run.log}"
PIDFILE="${SFT_PIDFILE:-$ROOT/outputs/sft/peg_insertion.pid}"
OUTDIR="$ROOT/outputs/sft/peg_insertion"
CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"

export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:18082}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:18082}"

CHECK=0
SMOKE=0
RESUME=0
WORKER=0
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --check) CHECK=1 ;;
    --smoke) SMOKE=1 ;;
    --resume) RESUME=1 ;;
    --worker) WORKER=1 ;;
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
  if [[ "$RESUME" -eq 1 ]]; then
    PY_ARGS+=(--resume)
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

  # Keep a process-ancestry fallback for agent versions that do not export a
  # stable marker.  Deliberately do not reject an ordinary VS Code terminal.
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

if [[ "$CHECK" -eq 1 || "$SMOKE" -eq 1 ]]; then
  _activate
  cd "$ROOT"
  _fill_python_args
  exec python "$ROOT/scripts/train_sft.py" "${PY_ARGS[@]}"
fi

if agent_reason="$(_agent_shell_reason)"; then
  echo "Refusing to start official SFT from a Cursor/Codex agent shell ($agent_reason)." >&2
  echo "Agent-owned processes are externally SIGKILLed after roughly 20 minutes; nohup/setsid cannot detach that ownership." >&2
  echo "Use the AutoDL web terminal:" >&2
  echo "  cd $ROOT && bash scripts/train_sft.sh" >&2
  exit 1
fi

if [[ "$WORKER" -eq 0 ]]; then
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "SFT already running pid=$(cat "$PIDFILE")" >&2
    exit 1
  fi
  if pgrep -f '[l]erobot-train' >/dev/null; then
    echo "lerobot-train already running:" >&2
    pgrep -af '[l]erobot-train' || true
    exit 1
  fi

  mkdir -p "$ROOT/outputs/sft"

  if [[ "$RESUME" -eq 0 && -d "$OUTDIR" ]]; then
    if [[ ! -f "$OUTDIR/checkpoints/last/pretrained_model/train_config.json" ]]; then
      ts="$(date +%Y%m%d_%H%M%S)"
      echo "archiving incomplete $OUTDIR -> ${OUTDIR}_killed_${ts}"
      mv "$OUTDIR" "${OUTDIR}_killed_${ts}"
      if [[ -f "$LOG" ]]; then
        mv "$LOG" "${LOG}.killed_${ts}"
      fi
    else
      echo "output dir exists with a checkpoint; pass --resume or move it aside" >&2
      exit 1
    fi
  fi

  if [[ "$RESUME" -eq 0 ]]; then
    : >"$LOG"
  fi

  worker_args=(--worker)
  if [[ "$RESUME" -eq 1 ]]; then
    worker_args+=(--resume)
  fi
  if [[ "${#EXTRA[@]}" -gt 0 ]]; then
    worker_args+=("${EXTRA[@]}")
  fi

  nohup setsid bash --noprofile --norc "$ROOT/scripts/train_sft.sh" \
    "${worker_args[@]}" </dev/null >/dev/null 2>&1 &

  sleep 2
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "started pid=$(cat "$PIDFILE") log=$LOG"
    exit 0
  fi
  echo "worker pidfile missing; last log:" >&2
  tail -n 20 "$LOG" 2>/dev/null || true
  exit 1
fi

# Detached worker (setsid). Do not launch this flag by hand.
set +e
# Process ownership is external metadata, not an environment variable.  Do not
# pretend that unsetting IDE variables can make an agent-owned process safe.
python3 -c 'import ctypes; ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, 0)' || true

mkdir -p "$(dirname "$LOG")" "$(dirname "$PIDFILE")"
echo $$ >"$PIDFILE"

_log() { echo "[detached] $*" >>"$LOG"; }

trap '_log "signal SIGTERM $(date -Iseconds) pid=$$ ppid=$PPID"; rm -f "$PIDFILE"; exit 143' TERM
trap '_log "signal SIGHUP $(date -Iseconds) pid=$$ ppid=$PPID"; rm -f "$PIDFILE"; exit 129' HUP
trap '_log "signal SIGINT $(date -Iseconds) pid=$$ ppid=$PPID"; rm -f "$PIDFILE"; exit 130' INT

_activate
cd "$ROOT"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# sshd/PAM leaves a 1024 soft NOFILE. The failed 16-worker configuration
# opened enough 3-camera videos/tensor DupFds to hit OSError 24 and deadlock:
# GPU memory stayed allocated while clocks dropped to P8 and ~25W.
if ! ulimit -n 1048576 2>/dev/null; then
  ulimit -n 65536 2>/dev/null || ulimit -n 16384 2>/dev/null || true
fi

_log "start $(date -Iseconds) pid=$$ ppid=$PPID nofile=$(ulimit -n)"
_log "ancestors=$(ps -o pid=,ppid=,cmd= -p $$ --no-headers) / ppid=$(ps -o pid=,cmd= -p $PPID --no-headers)"
if [[ -r /sys/fs/cgroup/memory.max ]]; then
  _log "cgroup_memory_max=$(< /sys/fs/cgroup/memory.max) cgroup_memory_current=$(< /sys/fs/cgroup/memory.current)"
elif [[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
  _log "cgroup_memory_max=$(< /sys/fs/cgroup/memory/memory.limit_in_bytes) cgroup_memory_current=$(< /sys/fs/cgroup/memory/memory.usage_in_bytes)"
fi

_fill_python_args
python "$ROOT/scripts/train_sft.py" "${PY_ARGS[@]}" >>"$LOG" 2>&1
ec=$?
_log "TRAIN_EXIT=$ec $(date -Iseconds)"
if [[ "$ec" -eq 137 ]]; then
  _log "TRAIN_SIGNAL=SIGKILL; inspect launcher ownership or host/cgroup OOM records (Python cannot catch SIGKILL)"
fi
rm -f "$PIDFILE"
exit "$ec"
