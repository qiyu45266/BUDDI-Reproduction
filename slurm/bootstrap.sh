#!/bin/bash
# End-to-end environment bootstrap & diagnostic for BUDDI Person A.
# Idempotent: safe to re-run. Each step is independently skippable.
# Exits on first failure with a clear "next step" message.
#
# Usage:
#   bash bootstrap.sh              # full run
#   bash bootstrap.sh --skip-env   # skip conda env creation (just verify)
#   bash bootstrap.sh --skip-gpu   # skip GPU-side check (login-node only)

set -uo pipefail   # NOTE: no -e; we handle errors explicitly per step

# ---------- formatting ----------
RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; BLU=$'\033[34m'; RST=$'\033[0m'
LOG=$HOME/buddi/bootstrap.log
mkdir -p $(dirname $LOG)

pass() { echo "${GRN}[PASS]${RST} $*"        | tee -a $LOG; }
fail() { echo "${RED}[FAIL]${RST} $*"        | tee -a $LOG; }
info() { echo "${BLU}[INFO]${RST} $*"        | tee -a $LOG; }
warn() { echo "${YLW}[WARN]${RST} $*"        | tee -a $LOG; }
step() { echo "" | tee -a $LOG
         echo "${BLU}==================== $* ====================${RST}" | tee -a $LOG; }

die() { fail "$1"; echo "" ; echo "Next step: $2" ; exit 1; }

SKIP_ENV=0; SKIP_GPU=0
for arg in "$@"; do
  case $arg in
    --skip-env) SKIP_ENV=1 ;;
    --skip-gpu) SKIP_GPU=1 ;;
  esac
done

echo "================================================================" | tee $LOG
echo " BUDDI bootstrap  $(date)" | tee -a $LOG
echo " host: $(hostname)" | tee -a $LOG
echo " log : $LOG" | tee -a $LOG
echo "================================================================" | tee -a $LOG

# ============================================================
step "0/8  Sanity: basic shell + paths"
# ============================================================
[ -n "${HOME:-}" ]    && pass "HOME=$HOME"                   || die "HOME not set" "fix shell"
[ -d "$HOME/buddi" ]  && pass "found $HOME/buddi"            || die "no $HOME/buddi" "git clone the repo first"
BUDDI_ROOT=$HOME/buddi
cd $BUDDI_ROOT

# ============================================================
step "1/8  Conda installation"
# ============================================================
if [ ! -f $HOME/miniconda3/etc/profile.d/conda.sh ]; then
  warn "miniconda3 not found at $HOME/miniconda3"
  if [ -t 0 ]; then
    read -p "Install miniconda now? [y/N] " ans
    if [ "$ans" = "y" ] || [ "$ans" = "Y" ]; then
      cd $HOME
      wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
      bash /tmp/mc.sh -b -p $HOME/miniconda3
      rm -f /tmp/mc.sh
      cd $BUDDI_ROOT
    else
      die "miniconda missing" "install miniconda or run with --skip-env after fixing"
    fi
  else
    die "miniconda missing (no tty)" "run interactively or install manually"
  fi
fi
source $HOME/miniconda3/etc/profile.d/conda.sh
pass "conda sourced ($(conda --version))"

# ============================================================
step "2/8  Conda ToS acceptance"
# ============================================================
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1 || true
pass "conda ToS accepted (main + r)"

# ============================================================
step "3/8  Conda environment hhcenv39"
# ============================================================
if [ $SKIP_ENV -eq 1 ]; then
  info "skipping env creation (--skip-env)"
else
  if conda env list | grep -qE "^hhcenv39\s"; then
    pass "env hhcenv39 already exists"
  else
    info "creating env hhcenv39 (this takes 5-15 min)"
    conda create -n hhcenv39 python=3.9 -y >>$LOG 2>&1 \
      && pass "env created" \
      || die "env create failed" "see $LOG for details"
  fi
fi

conda activate hhcenv39 2>/dev/null \
  && pass "env activated" \
  || die "cannot activate hhcenv39" "check 'conda env list'"

# ============================================================
step "4/8  Core packages (PyTorch + pytorch3d, CUDA 11.3)"
# ============================================================
need_install() {
  python -c "import $1" 2>/dev/null && return 1 || return 0
}

if need_install torch; then
  info "installing torch=1.12.1 + cu113 (A100-compatible)"
  conda install -n hhcenv39 -c pytorch -c nvidia \
    pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.3 -y >>$LOG 2>&1 \
    && pass "torch installed" \
    || die "torch install failed" "see $LOG, try mamba or check network"
else
  TV=$(python -c "import torch; print(torch.__version__)")
  pass "torch already installed ($TV)"
fi

if need_install pytorch3d; then
  info "installing pytorch3d deps"
  conda install -n hhcenv39 -c fvcore -c iopath -c conda-forge fvcore iopath -y >>$LOG 2>&1 \
    && pass "fvcore/iopath installed" \
    || warn "fvcore/iopath install issue (continuing)"
  conda install -n hhcenv39 -c bottler nvidiacub -y >>$LOG 2>&1 || warn "nvidiacub install issue"
  info "installing pytorch3d=0.7.2"
  conda install -n hhcenv39 -c pytorch3d pytorch3d=0.7.2 -y >>$LOG 2>&1 \
    && pass "pytorch3d installed" \
    || die "pytorch3d install failed" "see $LOG"
else
  pass "pytorch3d already installed"
fi

# ============================================================
step "5/8  Python deps (pip, smplx, tensorboard, etc.)"
# ============================================================
PIP_LOG=$LOG
need_install tensorboard && \
  conda install -n hhcenv39 -c conda-forge tensorboard -y >>$PIP_LOG 2>&1

pip install --quiet "setuptools==58.2.0" wheel 2>>$PIP_LOG \
  && pass "setuptools pinned" || warn "setuptools pin issue"

# Pin numpy<2: torch 1.12 was compiled against numpy 1.x and crashes with numpy 2.x
NP_MAJOR=$(python -c "import numpy; print(numpy.__version__.split('.')[0])" 2>/dev/null || echo "0")
if [ "$NP_MAJOR" != "1" ]; then
  info "downgrading numpy to <2 (torch 1.12 ABI requirement)"
  pip install --quiet "numpy<2" 2>>$PIP_LOG \
    && pass "numpy pinned to 1.x" \
    || die "numpy downgrade failed" "see $LOG"
else
  pass "numpy version OK ($(python -c 'import numpy; print(numpy.__version__)'))"
fi

PYDEPS=(opencv-python smplx scipy scikit-image loguru omegaconf ipdb einops chumpy trimesh)
MISSING=()
for p in "${PYDEPS[@]}"; do
  mod=${p//-/_}                    # opencv-python -> opencv_python
  case $p in
    opencv-python)   mod=cv2 ;;
    scikit-image)    mod=skimage ;;
  esac
  python -c "import $mod" 2>/dev/null || MISSING+=($p)
done
if [ ${#MISSING[@]} -gt 0 ]; then
  info "installing missing pip pkgs: ${MISSING[*]}"
  pip install --quiet "${MISSING[@]}" 2>>$PIP_LOG \
    && pass "pip deps installed" \
    || die "pip install failed" "see $LOG"
else
  pass "all pip deps already installed"
fi

# ============================================================
step "6/8  BUDDI import sanity (login node, CPU)"
# ============================================================
cd $BUDDI_ROOT
python - <<'PY' 2>&1 | tee -a $LOG
import sys, traceback
print("python:", sys.version.split()[0])
def safe_ver(mod):
    return getattr(mod, "__version__", "(no __version__)")
try:
    import numpy; print("numpy :", numpy.__version__)
    assert numpy.__version__.startswith("1."), "numpy must be 1.x for torch 1.12"
    import torch
    print("torch :", torch.__version__, "cuda-build:", torch.version.cuda)
    import torchvision; print("tv    :", torchvision.__version__)
    import pytorch3d;   print("p3d   :", pytorch3d.__version__)
    import smplx;       print("smplx :", safe_ver(smplx))
    import omegaconf;   print("oc    :", omegaconf.__version__)
    import loguru, einops, chumpy, trimesh, cv2, skimage, scipy
    print("all training imports OK")
except Exception:
    traceback.print_exc(); sys.exit(2)
PY
[ $? -eq 0 ] && pass "BUDDI training imports OK" \
              || die "import test failed" "see error above; likely a version mismatch"

# ============================================================
step "7/8  Repo structure & essentials"
# ============================================================
check_file() {
  if [ -e "$1" ]; then pass "found $1"
  else fail "MISSING $1  --  $2"; MISSING_ESSENTIALS=1
  fi
}
MISSING_ESSENTIALS=0
check_file "essentials/buddi/buddi_unconditional.pt" "run ./fetch_data.sh"
check_file "essentials/buddi/buddi_cond_bev.pt"      "run ./fetch_data.sh"
check_file "essentials/body_models/smplx"            "run ./fetch_bodymodels.sh"
check_file "llib/methods/hhc_diffusion/main.py"      "repo incomplete -- re-clone"
check_file "llib/methods/hhc_diffusion/configs/config_buddi_v02_cond_bev.yaml" "repo incomplete"

# Data symlinks (warn-only, not fatal: A may not need data on login node)
if [ ! -e "datasets/processed" ]; then
  warn "datasets/processed not linked yet. On compute node before training, run:"
  warn "  ln -s \$PROCESSED_DATASETS_FOLDER \$HOME/buddi/datasets/processed"
fi
if [ ! -e "datasets/original" ]; then
  warn "datasets/original not linked yet."
fi

[ $MISSING_ESSENTIALS -eq 0 ] \
  && pass "essentials present" \
  || die "essentials missing" "follow each MISSING hint above, then re-run"

# ============================================================
step "8/8  GPU verification (CUDA on A100/2080Ti)"
# ============================================================
if [ $SKIP_GPU -eq 1 ]; then
  info "skipping GPU check (--skip-gpu)"
  pass "bootstrap complete (login node only)"
  exit 0
fi

# Pick a GPU type that's most likely idle.
# IMPORTANT: must use -N (per-node) + custom format to expose GRES + state.
GPU_PROBE=""
SINFO_MG=$(sinfo -p multi_gpu -N -h -o "%T %G" 2>/dev/null)
SINFO_GPU=$(sinfo -p gpu       -N -h -o "%T %G" 2>/dev/null)

if echo "$SINFO_MG"  | grep -E "^idle " | grep -qi rtx; then
  GPU_PROBE="--partition=multi_gpu --gres=gpu:NVIDIAGeForceRTX2080Ti:1"
  info "found idle 2080Ti on multi_gpu"
elif echo "$SINFO_MG" | grep -E "^idle " | grep -qi 1080ti; then
  GPU_PROBE="--partition=multi_gpu --gres=gpu:NVIDIAGeForceGTX1080Ti:1"
  info "found idle 1080Ti on multi_gpu"
elif echo "$SINFO_GPU" | grep -E "^idle " | grep -qi 1080ti; then
  GPU_PROBE="--partition=gpu --gres=gpu:NVIDIAGeForceGTX1080Ti:1"
  info "found idle 1080Ti on gpu"
elif echo "$SINFO_GPU" | grep -E "^idle " | grep -qi "1080[^t]"; then
  GPU_PROBE="--partition=gpu --gres=gpu:NVIDIAGeForceGTX1080:1"
  info "found idle 1080 on gpu"
elif echo "$SINFO_GPU" | grep -E "^idle " | grep -qi a100; then
  GPU_PROBE="--partition=gpu --gres=gpu:NVIDIAA100-PCIE-40GB:1"
  info "found idle A100"
else
  GPU_PROBE="--partition=multi_gpu --gres=gpu:NVIDIAGeForceRTX2080Ti:1"
  warn "no idle node detected, defaulting to multi_gpu/2080Ti (least likely to queue)"
fi

info "submitting 5-min GPU sanity job  ($GPU_PROBE)"
GPU_LOG=$HOME/buddi/bootstrap_gpu_$(date +%H%M).log

cat <<EOF | sbatch --parsable $GPU_PROBE --time=00:05:00 --mem=8G --cpus-per-task=4 \
                   --output=$GPU_LOG --error=$GPU_LOG --job-name=buddi_check \
                   > /tmp/buddi_check_jid.txt
#!/bin/bash
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate hhcenv39
cd $BUDDI_ROOT
echo "==> node=\$(hostname)  date=\$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - <<'PY'
import torch
print("cuda available :", torch.cuda.is_available())
print("device name    :", torch.cuda.get_device_name(0))
print("device count   :", torch.cuda.device_count())
x = torch.randn(1024,1024, device="cuda")
y = (x @ x).sum().item()
print("matmul ok, sum :", y)
import pytorch3d
print("pytorch3d ver  :", pytorch3d.__version__)
PY
echo "==> done at \$(date)"
EOF

JID=$(cat /tmp/buddi_check_jid.txt)
info "job $JID submitted, waiting for completion (max 5 min)..."

# Poll until job leaves queue
ELAPSED=0
while squeue -j $JID -h 2>/dev/null | grep -q .; do
  sleep 10
  ELAPSED=$((ELAPSED+10))
  if [ $ELAPSED -gt 600 ]; then
    warn "job still queued after 10 min, you can check manually:"
    warn "  tail $GPU_LOG"
    warn "  squeue -j $JID"
    exit 0
  fi
done

echo ""
info "--- GPU job output ---"
cat $GPU_LOG 2>/dev/null | tee -a $LOG
echo ""

if grep -q "matmul ok" $GPU_LOG 2>/dev/null; then
  pass "GPU sanity check PASSED"
  echo ""
  echo "${GRN}================================================================${RST}"
  echo "${GRN}  BOOTSTRAP COMPLETE.  Next: ./smoke_test.sh 2080Ti${RST}"
  echo "${GRN}================================================================${RST}"
else
  die "GPU sanity check FAILED" "inspect $GPU_LOG; common causes: CUDA mismatch, OOM, missing driver"
fi
