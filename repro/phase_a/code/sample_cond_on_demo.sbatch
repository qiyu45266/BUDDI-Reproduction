#!/bin/bash
#SBATCH --job-name=buddi_cond_demo
#SBATCH --partition=multi_gpu
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/home/gaor/buddi/slurm_logs/cond_demo_%j.out
#SBATCH --error=/home/gaor/buddi/slurm_logs/cond_demo_%j.err

# Use BUDDI conditional ckpt + 6 demo images (with precomputed BEV)
# to demonstrate conditional generation. Outputs samples + 360-view GIFs.

set -euo pipefail
export BUDDI_ROOT=$HOME/buddi
cd $BUDDI_ROOT

CONDA_SH=""
for C in $HOME/miniconda3/etc/profile.d/conda.sh \
         $HOME/anaconda3/etc/profile.d/conda.sh \
         /opt/conda/etc/profile.d/conda.sh; do
  [ -f "$C" ] && { CONDA_SH="$C"; break; }
done
source $CONDA_SH
conda activate hhcenv39

OUT_BASE=$BUDDI_ROOT/outputs/person_a/cond
CKPT=$BUDDI_ROOT/essentials/buddi/buddi_cond_bev.pt
CFG=$BUDDI_ROOT/essentials/buddi/buddi_cond_bev.yaml

mkdir -p $OUT_BASE

# Generate conditioned samples with the standard schedule + render
python -u llib/methods/hhc_diffusion/evaluation/sample.py \
  --exp-cfg $CFG \
  --checkpoint-name $CKPT \
  --output-folder $OUT_BASE \
  --num-samples 128 \
  --max-t 1000 \
  --skip-steps 10 \
  --batch_size 8 \
  --log-steps 100 \
  --save-vis \
  --max-images-render=16 \
  --condition

echo "==> Conditional sampling done at $(date)"
ls -la $OUT_BASE/
