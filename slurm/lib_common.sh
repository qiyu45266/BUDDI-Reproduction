#!/bin/bash
# Shared library for BUDDI training pool.
# Sourced by train.sbatch. Do NOT execute directly.

LOCK_TTL=600       # 10 min: stale lockfile reclaim threshold
PREEMPT_POLL=30    # 30s: watchdog poll interval

# ---------- Lockfile primitives ----------

claim_lock() {
  local LOCKFILE=$1
  mkdir -p "$(dirname $LOCKFILE)"
  if (set -o noclobber; echo "$SLURM_JOB_ID@$(hostname)@$(date +%s)" > $LOCKFILE) 2>/dev/null; then
    return 0
  fi
  return 1
}

# Try to claim the lock. If held by a live job, exit 0 cleanly.
# If the holder is dead and lock is stale (>LOCK_TTL), reclaim.
acquire_or_yield() {
  local LOCKFILE=$1
  if claim_lock $LOCKFILE; then
    echo "[lock] CLAIMED by $SLURM_JOB_ID on $(hostname)"
    return 0
  fi

  local OWNER=$(cat $LOCKFILE 2>/dev/null || echo "?@?@0")
  local OWNER_JID=${OWNER%%@*}
  local OWNER_TS=${OWNER##*@}
  local NOW=$(date +%s)
  local AGE=$((NOW - OWNER_TS))

  if squeue -j $OWNER_JID -h 2>/dev/null | grep -q .; then
    echo "[lock] held by LIVE job $OWNER  ->  graceful exit"
    exit 0
  fi
  if [ $AGE -lt $LOCK_TTL ]; then
    echo "[lock] held by recent ghost $OWNER (age ${AGE}s)  ->  graceful exit"
    exit 0
  fi
  echo "[lock] STALE (${AGE}s old, owner $OWNER_JID dead)  ->  reclaiming"
  rm -f $LOCKFILE
  claim_lock $LOCKFILE || { echo "[lock] re-claim race lost  ->  exit"; exit 0; }
}

# ---------- Watchdog: safe ckpt backup + preemption listener ----------

start_watchdog() {
  local CKPT_DIR=$1
  local PREEMPT_FILE=$2
  local TRAINER_PID=$3
  local MY_GPU=${4:-unknown}
  (
    while kill -0 $TRAINER_PID 2>/dev/null; do
      sleep $PREEMPT_POLL
      # safe ckpt copy (atomic rename)
      LATEST=$(ls -t $CKPT_DIR/*.pt 2>/dev/null | grep -v _safe | head -1)
      if [ -n "$LATEST" ]; then
        cp $LATEST $CKPT_DIR/_safe.pt.tmp 2>/dev/null && \
          mv $CKPT_DIR/_safe.pt.tmp $CKPT_DIR/_safe.pt
      fi
      # A100 preempts non-A100. PREEMPT sentinel is set by an incoming A100 job.
      if [ "$MY_GPU" != "A100" ] && [ -f $PREEMPT_FILE ]; then
        echo "[watchdog] PREEMPT received  ->  SIGTERM trainer $TRAINER_PID"
        kill -TERM $TRAINER_PID
        sleep 60   # let trainer save and exit
        break
      fi
    done
  ) &
  echo $!
}

# ---------- Self-perpetuating chain ----------

submit_heir() {
  local HEIR_SCRIPT=$1
  local HEIR_EXPORT=$2
  local PARTITION=$3
  local GRES=$4

  # Respect kill switch
  if [ -f $BUDDI_ROOT/exp_logs/personA/KILL_SWITCH ]; then
    echo "[heir] KILL_SWITCH active  ->  not queueing heir"
    return
  fi

  local NEW_JID=$(sbatch --parsable \
    --dependency=afterany:$SLURM_JOB_ID \
    --partition=$PARTITION \
    --gres=$GRES \
    --export=$HEIR_EXPORT \
    $HEIR_SCRIPT 2>/dev/null || true)
  if [ -n "$NEW_JID" ]; then
    echo "[heir] queued $NEW_JID as heir of $SLURM_JOB_ID  (partition=$PARTITION)"
  else
    echo "[heir] FAILED to queue heir"
  fi
}
