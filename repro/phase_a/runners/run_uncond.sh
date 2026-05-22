#!/usr/bin/env bash
# Phase A: unconditional sampling at 4 DDIM schedules (paper §4 baseline + 3 faster).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUDDI_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SEED="${SEED:-42}"   # override:  SEED=7 bash run_uncond.sh

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hhcenv39

cd "$BUDDI_ROOT"
export PYTHONPATH="$BUDDI_ROOT"
export PYOPENGL_PLATFORM=egl   # headless GL for pyrender on WSL/server

OUT_BASE="$BUDDI_ROOT/repro/phase_a/outputs/uncond"
CKPT="$BUDDI_ROOT/essentials/buddi/buddi_unconditional.pt"
CFG="$BUDDI_ROOT/essentials/buddi/buddi_unconditional.yaml"
[ -f "$CKPT" ] || { echo "missing $CKPT — run repro/shared/env/2_fetch_essentials.sh first"; exit 1; }
[ -d "$BUDDI_ROOT/essentials/body_models/smplx" ] || { echo "missing SMPL-X — run repro/shared/env/3_fetch_bodymodels.sh first"; exit 1; }
mkdir -p "$OUT_BASE"

run_schedule() {
  local MAX_T=$1 SKIP=$2 N=$3 RENDER=$4
  echo "============================================================"
  echo " Schedule: max-t=$MAX_T skip=$SKIP samples=$N render=$RENDER seed=$SEED"
  echo " $(date)"
  echo "============================================================"
  local EXTRA=""
  [ "$RENDER" = "1" ] && EXTRA="--save-vis --max-images-render=8"
  python -u llib/methods/hhc_diffusion/evaluation/sample.py \
    --exp-cfg "$CFG" --checkpoint-name "$CKPT" --output-folder "$OUT_BASE" \
    --num-samples "$N" --max-t "$MAX_T" --skip-steps "$SKIP" \
    --batch_size 8 --log-steps 100 --seed "$SEED" \
    $EXTRA
}

# max-t  skip   N    render?
run_schedule   1000   10    512   1     # paper baseline = 100 DDIM steps
run_schedule   1000   25    512   0     #  40 steps
run_schedule   1000   50    512   0     #  20 steps
run_schedule   1000   100   512   0     #  10 steps

echo "==> All schedules done at $(date)"
ls -la "$OUT_BASE/"
