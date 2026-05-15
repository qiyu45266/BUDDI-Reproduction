#!/bin/bash
# Idempotent pool-refiller. Safe to run as a cron tick (every 5 min).
# If any (model, gpu) combination has fewer pending+running jobs than TARGET,
# top it up. The self-perpetuating chain in train.sbatch should keep the pool
# alive on its own; this script is a safety net for edge cases (sbatch failures,
# manual scancel storms, etc.).

set -euo pipefail
cd $(dirname $0)

LOG=$HOME/buddi/slurm_logs/controller.log
mkdir -p $(dirname $LOG)
exec >>$LOG 2>&1

# Stop refilling if kill switch is set
if [ -f $HOME/buddi/exp_logs/personA/KILL_SWITCH ]; then
  echo "$(date) KILL_SWITCH set, not refilling"
  exit 0
fi

# Desired minimum (pending OR running) count per (model, gpu)
declare -A TARGET=(
  [cond_A100]=2    [uncond_A100]=2
  [cond_2080Ti]=2  [uncond_2080Ti]=2
  [cond_1080Ti]=1  [uncond_1080Ti]=1
)

count_existing() {
  # MODEL + GPU are tagged in the --comment field, readable via %k.
  local MODEL=$1 GPU=$2
  squeue -u $USER -h -o "%k" 2>/dev/null | \
    grep -E "MODEL=$MODEL GPU=$GPU\$" | wc -l
}

resolve_resource() {
  case $1 in
    A100)   echo "gpu       gpu:NVIDIAA100-PCIE-40GB:1   256" ;;
    2080Ti) echo "multi_gpu gpu:NVIDIAGeForceRTX2080Ti:1 128" ;;
    1080Ti) echo "gpu       gpu:NVIDIAGeForceGTX1080Ti:1  64" ;;
  esac
}

for KEY in "${!TARGET[@]}"; do
  MODEL=${KEY%_*}
  GPU=${KEY##*_}
  WANT=${TARGET[$KEY]}
  HAVE=$(count_existing $MODEL $GPU)
  NEED=$((WANT - HAVE))
  if [ $NEED -gt 0 ]; then
    read PART GRES BS <<<"$(resolve_resource $GPU)"
    for i in $(seq 1 $NEED); do
      JID=$(sbatch --parsable \
        --partition=$PART --gres=$GRES \
        --export=MODEL=$MODEL,BATCH=$BS,GPU_TYPE=$GPU \
        --comment="MODEL=$MODEL GPU=$GPU" \
        $PWD/train.sbatch 2>/dev/null || true)
      echo "$(date) [refill] $KEY  have=$HAVE  want=$WANT  submitted=$JID"
    done
  fi
done
