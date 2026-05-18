#!/bin/bash
#SBATCH --job-name=buddi_sample
#SBATCH --partition=multi_gpu
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=/home/gaor/buddi/slurm_logs/sample_%j.out
#SBATCH --error=/home/gaor/buddi/slurm_logs/sample_%j.err

# ============================================================
# Generate BUDDI unconditional samples at 4 different sampling
# schedules. Each takes ~10-40 min depending on schedule.
# Outputs end up in: outputs/person_a/uncond/generate_<T>_<S>_v0/
# ============================================================

set -euo pipefail

export BUDDI_ROOT=$HOME/buddi
cd $BUDDI_ROOT

# Auto-detect conda
CONDA_SH=""
for C in $HOME/miniconda3/etc/profile.d/conda.sh \
         $HOME/anaconda3/etc/profile.d/conda.sh \
         /opt/conda/etc/profile.d/conda.sh; do
  [ -f "$C" ] && { CONDA_SH="$C"; break; }
done
source $CONDA_SH
conda activate hhcenv39

OUT_BASE=$BUDDI_ROOT/outputs/person_a/uncond
CKPT=$BUDDI_ROOT/essentials/buddi/buddi_unconditional.pt
CFG=$BUDDI_ROOT/essentials/buddi/buddi_unconditional.yaml

mkdir -p $OUT_BASE

# Run a schedule: max-t skip-steps num-samples render
run_schedule() {
  local MAX_T=$1
  local SKIP=$2
  local N=$3
  local RENDER=$4

  echo "============================================================"
  echo " Schedule: max-t=$MAX_T  skip=$SKIP  samples=$N  render=$RENDER"
  echo " $(date)"
  echo "============================================================"

  EXTRA=""
  [ "$RENDER" = "1" ] && EXTRA="--save-vis --max-images-render=8"

  python -u llib/methods/hhc_diffusion/evaluation/sample.py \
    --exp-cfg $CFG \
    --checkpoint-name $CKPT \
    --output-folder $OUT_BASE \
    --num-samples $N \
    --max-t $MAX_T \
    --skip-steps $SKIP \
    --batch_size 8 \
    --log-steps 100 \
    $EXTRA
}

# === Schedules (paper baseline first, then progressively faster) ===
#                  max-t  skip   N    render?
run_schedule       1000   10    512   1     # paper baseline = 100 DDIM steps
run_schedule       1000   25    512   0     #  40 steps
run_schedule       1000   50    512   0     #  20 steps
run_schedule       1000   100   512   0     #  10 steps  (aggressive)

echo "==> All schedules done at $(date)"
ls -la $OUT_BASE/
