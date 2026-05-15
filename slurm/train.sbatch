#!/bin/bash
#SBATCH --job-name=buddi
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=47:55:00
#SBATCH --output=/home/gaor/buddi/slurm_logs/%x_%j.out
#SBATCH --error=/home/gaor/buddi/slurm_logs/%x_%j.err
#SBATCH --requeue
#SBATCH --signal=B:TERM@300

# ============================================================
# Universal BUDDI training script.
# Required env vars (set via sbatch --export):
#   MODEL=cond|uncond
#   BATCH=256|128|64
#   GPU_TYPE=A100|2080Ti|1080Ti
# Partition + --gres are set on the sbatch command line.
# ============================================================

set -euo pipefail

export BUDDI_ROOT=$HOME/buddi
cd $BUDDI_ROOT

# --- Adjust conda path to your install ---
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate buddi

source $BUDDI_ROOT/slurm/lib_common.sh

# Resolve model-specific config
case ${MODEL:-} in
  cond)
    CFG=llib/methods/hhc_diffusion/configs/config_buddi_v02_cond_bev.yaml
    RUN_NAME=buddi_FC_cond
    EXTRA_OPTS="model.regressor.losses.pseudogt_v2v.weight=[1000.0]"
    ;;
  uncond)
    CFG=llib/methods/hhc_diffusion/configs/config_buddi_v02.yaml
    RUN_NAME=buddi_FC_uncond
    EXTRA_OPTS="datasets.chi3d.load_unit_glob_and_transl=True \
                model.regressor.losses.pseudogt_v2v.weight=[100.0]"
    ;;
  *)
    echo "ERROR: MODEL must be cond|uncond (got '${MODEL:-}')"
    exit 1
    ;;
esac

BASE=$BUDDI_ROOT/exp_logs/personA/$MODEL
RUN_DIR=$BASE/$RUN_NAME
CKPT_DIR=$RUN_DIR/checkpoints
LOCKFILE=$RUN_DIR/TRAIN.lock
PREEMPT_FILE=$RUN_DIR/PREEMPT

# Resolve partition + GRES for heir submission (must match sbatch CLI)
case ${GPU_TYPE:-} in
  A100)
    HEIR_PARTITION=gpu
    HEIR_GRES=gpu:NVIDIAA100-PCIE-40GB:1
    ;;
  2080Ti)
    HEIR_PARTITION=multi_gpu
    HEIR_GRES=gpu:NVIDIAGeForceRTX2080Ti:1
    ;;
  1080Ti)
    HEIR_PARTITION=gpu
    HEIR_GRES=gpu:NVIDIAGeForceGTX1080Ti:1
    ;;
  *)
    echo "ERROR: GPU_TYPE must be A100|2080Ti|1080Ti (got '${GPU_TYPE:-}')"
    exit 1
    ;;
esac

echo "============================================================"
echo " $(date)  Job=$SLURM_JOB_ID  Node=$(hostname)"
echo "  Model=$MODEL  GPU=$GPU_TYPE  Batch=$BATCH"
echo "  Run=$RUN_DIR"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

mkdir -p $CKPT_DIR $HOME/buddi/slurm_logs

# Lockfile arbitration -- if another job is training, exit cleanly.
acquire_or_yield $LOCKFILE

# A100 preempts non-A100 holders. We set the sentinel; the existing 2080Ti
# trainer's watchdog will catch it within 30s. We already hold the lock,
# so this is just a cleanup signal.
if [ "$GPU_TYPE" = "A100" ]; then
  touch $PREEMPT_FILE
  echo "[priority] A100 set PREEMPT sentinel"
  sleep 5
  rm -f $PREEMPT_FILE
fi

# Trap: on any exit (success / SIGTERM / crash / timeout warning),
# release lock and queue an heir on the same partition.
on_exit() {
  local EC=$?
  echo "[exit] $(date) job $SLURM_JOB_ID exiting (code=$EC)"
  [ -n "${WD_PID:-}" ] && kill $WD_PID 2>/dev/null || true
  rm -f $LOCKFILE
  submit_heir $BUDDI_ROOT/slurm/train.sbatch \
              "MODEL=$MODEL,BATCH=$BATCH,GPU_TYPE=$GPU_TYPE" \
              $HEIR_PARTITION $HEIR_GRES
}
trap on_exit EXIT INT TERM

# Launch trainer in background so watchdog can SIGTERM it on preemption.
python -u llib/methods/hhc_diffusion/main.py \
  --exp-cfg $CFG \
  --exp-opts \
    logging.base_folder=$BASE \
    logging.run=$RUN_NAME \
    logging.logger=tensorboard \
    logging.ckpt_freq=1 \
    datasets.train_names=['flickrci3ds','chi3d'] \
    datasets.train_composition=[0.75,0.25] \
    datasets.augmentation.use=True \
    batch_size=$BATCH \
    training.max_steps=200000 \
    $EXTRA_OPTS &
TRAINER_PID=$!

WD_PID=$(start_watchdog $CKPT_DIR $PREEMPT_FILE $TRAINER_PID $GPU_TYPE)
echo "[main] trainer pid=$TRAINER_PID  watchdog pid=$WD_PID"

wait $TRAINER_PID
RC=$?
echo "[main] trainer exited with code $RC at $(date)"
exit $RC
