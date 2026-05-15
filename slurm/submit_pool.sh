#!/bin/bash
# One-shot initial pool submission for BUDDI training.
# Run ONCE after smoke test passes. The self-perpetuating chain in train.sbatch
# will keep refilling. controller.sh provides a safety net.

set -euo pipefail

cd $(dirname $0)
SBATCH=$PWD/train.sbatch
mkdir -p $HOME/buddi/slurm_logs

# Remove kill switch in case it was left over from a previous teardown
rm -f $HOME/buddi/exp_logs/personA/KILL_SWITCH

submit() {
  local MODEL=$1 BATCH=$2 GPU=$3 PART=$4 GRES=$5
  local JID=$(sbatch --parsable \
    --partition=$PART --gres=$GRES \
    --export=MODEL=$MODEL,BATCH=$BATCH,GPU_TYPE=$GPU \
    --comment="MODEL=$MODEL GPU=$GPU" \
    $SBATCH)
  echo "  [submitted] job=$JID  model=$MODEL  gpu=$GPU  partition=$PART"
}

echo "==> A100 pool (highest priority)"
for i in 1 2; do
  submit cond   256 A100  gpu        gpu:NVIDIAA100-PCIE-40GB:1
  submit uncond 256 A100  gpu        gpu:NVIDIAA100-PCIE-40GB:1
done

echo "==> 2080Ti pool (fallback, will yield to A100)"
for i in 1 2; do
  submit cond   128 2080Ti multi_gpu gpu:NVIDIAGeForceRTX2080Ti:1
  submit uncond 128 2080Ti multi_gpu gpu:NVIDIAGeForceRTX2080Ti:1
done

echo "==> 1080Ti pool (last resort)"
submit cond   64 1080Ti gpu gpu:NVIDIAGeForceGTX1080Ti:1
submit uncond 64 1080Ti gpu gpu:NVIDIAGeForceGTX1080Ti:1

echo ""
echo "==> Pool snapshot:"
squeue -u $USER -o "%.10i %.9P %.10j %.8T %.10M %.6D %R"
