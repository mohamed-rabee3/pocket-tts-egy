#!/usr/bin/env bash
# Runs an Egyptian training (CONFIG: the teacher by default, or the distillation) until max_steps, resuming from the latest
# checkpoint after any crash. `touch $RUN/STOP` ends it cleanly.
#
# A watchdog kills a run that is stuck: after an exception the data-loader
# subprocesses can keep the Python process alive forever (seen after a CUDA OOM),
# so a run is restarted when its log shows a traceback, or when neither its log nor
# progress.jsonl has changed for STALL_MIN minutes. After an out-of-memory crash on
# the batch-64 config it switches to LOW_MEM_CONFIG, batch 32 x accum 2 (same effective batch).
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
RUN=${RUN:-/data/egy/runs/teacher_v2}
LOG=${LOG:-/data/egy/logs}
PREPARE_LOG=${PREPARE_LOG:-/data/egy/prepare_v2.log}
PREPARE_SCRIPT=${PREPARE_SCRIPT:-egyptian/scripts/prepare_v2.sh}
STALL_MIN=${STALL_MIN:-20}
CONFIG=${CONFIG:-$REPO/egyptian/configs/egy_teacher.yaml}
LOW_MEM_CONFIG=${LOW_MEM_CONFIG:-$REPO/egyptian/configs/egy_teacher_b32.yaml}
HF_TOKEN=$(sed -n 's/^huggingface: //p' "${HF_CREDENTIALS:-/workspace/apis.txt}")
export HF_TOKEN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOG" "$RUN"
log() { echo "[$(date '+%F %T')] $*" >> "$LOG/supervisor.log"; }

log "waiting for data preparation"
until grep -q "data ready" "$PREPARE_LOG" 2>/dev/null; do
  # Only a bash running the script: a shell that merely launched it also has the
  # name on its command line and would keep this check passing after prep died.
  if ! pgrep -f "^(/usr/bin/env )?(/bin/)?bash [^ ]*$PREPARE_SCRIPT" > /dev/null; then
    log "$PREPARE_SCRIPT stopped without 'data ready'"; exit 1
  fi
  sleep 60
done

# Minutes since the run last wrote anything (log or progress.jsonl).
idle_minutes() {
  local newest
  newest=$(stat -c %Y "$1" "$RUN/progress.jsonl" 2>/dev/null | sort -n | tail -1)
  echo $(( ($(date +%s) - newest) / 60 ))
}

restarts=0
cd "$REPO"
while [ ! -f "$RUN/STOP" ]; do
  attempt_log="$LOG/train_$(date +%Y%m%d_%H%M%S).log"
  log "starting training ($CONFIG), log $attempt_log"
  # Own process group, so the watchdog can kill the loaders along with the trainer.
  setsid uv run python training/train.py "$CONFIG" > "$attempt_log" 2>&1 &
  pid=$!
  reason=""
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    if grep -q "^Traceback" "$attempt_log"; then
      sleep 120  # let a clean exit happen first
      kill -0 "$pid" 2>/dev/null && reason="exception, process did not exit"
    elif [ "$(idle_minutes "$attempt_log")" -ge "$STALL_MIN" ]; then
      reason="no output for $STALL_MIN min"
    fi
    if [ -n "$reason" ]; then
      log "watchdog: $reason; killing process group $pid"
      kill -TERM -- "-$pid" 2>/dev/null; sleep 30; kill -KILL -- "-$pid" 2>/dev/null
      break
    fi
  done
  wait "$pid"
  code=$?
  if [ $code -eq 0 ] && [ -z "$reason" ]; then log "training finished"; break; fi
  restarts=$((restarts + 1))
  if grep -q "OutOfMemoryError\|CUDA out of memory" "$attempt_log"; then
    log "OOM crash"
    if [ "$CONFIG" != "$LOW_MEM_CONFIG" ]; then
      CONFIG=$LOW_MEM_CONFIG
      log "switching to the low-memory config"
    fi
  else
    log "crash (exit $code${reason:+, $reason}), see $attempt_log"
  fi
  if [ $restarts -ge 30 ]; then log "too many restarts, giving up"; exit 1; fi
  sleep 60
done
