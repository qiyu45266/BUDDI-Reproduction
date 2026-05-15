#!/bin/bash
# 1-hour validation job. Run BEFORE submit_pool.sh.
# Confirms: no OOM, loss decreasing, ckpt write works, env active.
# Uses an isolated run dir (smoke_<TIMESTAMP>) so it never touches the real pool.

set -euo pipefail

cd $(dirname $0)
RUN_TAG=smoke_$(date +%Y%m%d_%H%M)

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=buddi_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:NVIDIAA100-PCIE-40GB:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=$HOME/buddi/slurm_logs/${RUN_TAG}_%j.out
#SBATCH --error=$HOME/buddi/slurm_logs/${RUN_TAG}_%j.err

set -euo pipefail

export BUDDI_ROOT=\$HOME/buddi
cd \$BUDDI_ROOT
source \$HOME/miniconda3/etc/profile.d/conda.sh
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
    batch_size=256 \\
    training.max_steps=500 \\
    model.regressor.losses.pseudogt_v2v.weight=[1000.0]

echo "==> Smoke test finished at \$(date)"
ls -la \$BUDDI_ROOT/exp_logs/personA/_smoke/$RUN_TAG/checkpoints/ || true
EOF

echo "Submitted smoke job. Watch with:"
echo "  squeue -u \$USER -n buddi_smoke"
echo "  tail -f \$HOME/buddi/slurm_logs/${RUN_TAG}_*.out"
