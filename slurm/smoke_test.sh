#!/bin/bash
# 1-hour validation job. Run BEFORE submit_pool.sh.
# Confirms: no OOM, loss decreasing, ckpt write works, env active.
# Uses an isolated run dir (smoke_<TIMESTAMP>) so it never touches the real pool.
#
# Usage:
#   ./smoke_test.sh              # default: A100
#   ./smoke_test.sh A100
#   ./smoke_test.sh 2080Ti       # idle 2080Ti, faster to schedule
#   ./smoke_test.sh 1080Ti       # last-resort, slowest

set -euo pipefail

GPU_TYPE=${1:-A100}

case $GPU_TYPE in
  A100)
    PARTITION=gpu
    GRES=gpu:NVIDIAA100-PCIE-40GB:1
    BATCH=256
    MEM=60G
    ;;
  2080Ti)
    PARTITION=multi_gpu
    GRES=gpu:NVIDIAGeForceRTX2080Ti:1
    BATCH=128
    MEM=40G
    ;;
  1080Ti)
    PARTITION=gpu
    GRES=gpu:NVIDIAGeForceGTX1080Ti:1
    BATCH=64
    MEM=40G
    ;;
  1080)
    PARTITION=gpu
    GRES=gpu:NVIDIAGeForceGTX1080:1
    BATCH=64
    MEM=40G
    ;;
  *)
    echo "Usage: $0 [A100|2080Ti|1080Ti|1080]"
    exit 1
    ;;
esac

cd $(dirname $0)
RUN_TAG=smoke_${GPU_TYPE}_$(date +%Y%m%d_%H%M)

echo "==> Smoke test on $GPU_TYPE  (partition=$PARTITION, batch=$BATCH)"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=buddi_smoke
#SBATCH --partition=$PARTITION
#SBATCH --gres=$GRES
#SBATCH --cpus-per-task=8
#SBATCH --mem=$MEM
#SBATCH --time=01:00:00
#SBATCH --output=$HOME/buddi/slurm_logs/${RUN_TAG}_%j.out
#SBATCH --error=$HOME/buddi/slurm_logs/${RUN_TAG}_%j.err

set -euo pipefail

export BUDDI_ROOT=\$HOME/buddi
cd \$BUDDI_ROOT

CONDA_SH=""
for C in \\
    \$HOME/miniconda3/etc/profile.d/conda.sh \\
    \$HOME/anaconda3/etc/profile.d/conda.sh \\
    \$HOME/conda/etc/profile.d/conda.sh \\
    /apps/anaconda3/etc/profile.d/conda.sh \\
    /apps/miniconda3/etc/profile.d/conda.sh \\
    /opt/conda/etc/profile.d/conda.sh \\
    /usr/local/anaconda3/etc/profile.d/conda.sh \\
    /usr/local/miniconda3/etc/profile.d/conda.sh; do
  [ -f "\$C" ] && { CONDA_SH="\$C"; break; }
done
if [ -z "\$CONDA_SH" ]; then
  echo "ERROR: conda.sh not found. Edit smoke_test.sh and add your path."
  exit 1
fi
echo "[env] using \$CONDA_SH"
source \$CONDA_SH
conda activate buddi

echo "==> Smoke test on \$(hostname), GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u llib/methods/hhc_diffusion/main.py \\
  --exp-cfg llib/methods/hhc_diffusion/configs/config_buddi_v02_cond_bev.yaml \\
  --exp-opts \\
    logging.base_folder=\$BUDDI_ROOT/exp_logs/personA/_smoke \\
    logging.run=$RUN_TAG \\
    logging.logger=tensorboard \\
    logging.ckpt_freq=1 \\
    datasets.train_names=['flickrci3ds','chi3d'] \\
    datasets.train_composition=[0.75,0.25] \\
    batch_size=$BATCH \\
    training.max_steps=500 \\
    model.regressor.losses.pseudogt_v2v.weight=[1000.0]

echo "==> Smoke test finished at \$(date)"
ls -la \$BUDDI_ROOT/exp_logs/personA/_smoke/$RUN_TAG/checkpoints/ || true
EOF

echo ""
echo "Submitted smoke job on $GPU_TYPE. Monitor with:"
echo "  squeue -u \$USER -n buddi_smoke"
echo "  tail -f \$HOME/buddi/slurm_logs/${RUN_TAG}_*.out"
