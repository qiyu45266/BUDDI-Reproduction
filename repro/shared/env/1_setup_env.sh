#!/usr/bin/env bash
# One-shot env build for the BUDDI reproduction, Ada-class GPU (sm_89) friendly.
# Works on WSL2 Ubuntu 22.04/24.04 or native Linux with CUDA 11.8 driver (>=520.61).
# Idempotent — safe to re-run.
set -euo pipefail

# 1. Auto-detect repo root: this script lives at repro/shared/env/1_setup_env.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUDDI_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export BUDDI_ROOT
echo "BUDDI_ROOT=$BUDDI_ROOT"

# 2. Verify miniconda is installed
if [ ! -d "$HOME/miniconda3" ]; then
  echo "miniconda not found at \$HOME/miniconda3. Run repro/shared/env/0_install_miniconda.sh first."
  exit 1
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# 3. Accept conda TOS (required by conda >= 25 for default channels)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1 || true

# 4. Create env if missing
if ! conda env list | grep -q '^hhcenv39 '; then
  conda create -n hhcenv39 python=3.9 -y
fi
conda activate hhcenv39

PIP="$HOME/miniconda3/envs/hhcenv39/bin/pip"
PY="$HOME/miniconda3/envs/hhcenv39/bin/python"

# 5. Pin numpy<1.24 FIRST so subsequent installs do not pull numpy>=2 (PyTorch 2.0.1
#    was built against numpy 1.x; mixing causes Bus error on import).
$PIP install --quiet "numpy<1.24"

# 6. PyTorch 2.0.1 + CUDA 11.8 (native sm_89 for RTX 4070, sm_86 for 3060/3070/3080/3090)
$PY -c "import torch" 2>/dev/null \
  || $PIP install --quiet --index-url https://download.pytorch.org/whl/cu118 \
       torch==2.0.1+cu118 torchvision==0.15.2+cu118

# 7. Sanity-check torch + GPU before pulling more
$PY <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available — check NVIDIA driver"
cap = torch.cuda.get_device_capability(0)
print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)} cap=sm_{cap[0]}{cap[1]}")
x = torch.randn(64, 64, device="cuda")
_ = (x @ x).sum().item()
print("CUDA matmul OK")
PY

# 8. pytorch3d 0.7.4 prebuilt wheel (py39 + cu118 + pyt201). No nvcc compile needed.
$PY -c "import pytorch3d" 2>/dev/null || {
  $PIP install --quiet fvcore iopath
  $PIP install --quiet --no-index --no-cache-dir pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt201/download.html
}

# 9. Runtime deps for sample.py / analyze.py / make_gallery.py
$PIP install --quiet --upgrade-strategy only-if-needed \
  opencv-python smplx scipy scikit-image loguru omegaconf ipdb einops trimesh \
  matplotlib tqdm imageio imageio-ffmpeg pyrender 'pyglet<2' \
  tensorboard pandas wandb \
  setuptools==58.2.0

# 10. chumpy from mattloper GitHub master (PyPI chumpy 0.70 imports `from numpy
#     import int, bool, float, complex` which numpy>=1.20 removed). The GitHub
#     master fixes this. Use --no-build-isolation so chumpy sees env's setuptools.
$PY -c "import chumpy" 2>/dev/null \
  || $PIP install --quiet --no-build-isolation "git+https://github.com/mattloper/chumpy.git"

# 11. libGLU for pyrender (no sudo needed — install from conda-forge)
$PY -c "import pyrender" 2>/dev/null \
  || conda install -n hhcenv39 -c conda-forge -y libglu 2>&1 | tail -3

# 12. Re-pin numpy<1.24 in case any conda/pip step bumped it
$PIP install --quiet --force-reinstall --no-deps "numpy==1.23.5"

# 13. Apply our patches to upstream llib/ (currently: --seed support in sample.py).
#     Each patch is idempotent — git apply --check first, skip if already applied.
PATCH_DIR="$BUDDI_ROOT/repro/shared/patches"
if [ -d "$PATCH_DIR" ]; then
  for p in "$PATCH_DIR"/*.patch; do
    [ -f "$p" ] || continue
    name=$(basename "$p")
    cd "$BUDDI_ROOT"
    if git apply --check "$p" 2>/dev/null; then
      git apply "$p" && echo "  applied $name"
    elif git apply --reverse --check "$p" 2>/dev/null; then
      echo "  $name already applied — skip"
    else
      echo "  WARN: cannot apply $name (does not apply cleanly and is not already applied)"
    fi
  done
fi

# 14. Final import verification
$PY <<'PY'
import importlib
mods = ["torch","torchvision","pytorch3d","numpy","cv2","smplx","scipy","skimage",
        "loguru","omegaconf","einops","trimesh","matplotlib","tqdm","imageio",
        "pyrender","tensorboard","pandas","chumpy","wandb"]
bad = []
for m in mods:
    try: importlib.import_module(m)
    except Exception as e: bad.append((m, repr(e)[:120]))
if bad:
    print("MISSING:", bad)
    raise SystemExit(2)
print("All imports OK.")
import torch
print(f"  torch={torch.__version__} cuda_ok={torch.cuda.is_available()}")
PY

echo
echo "==================================================="
echo " env hhcenv39 ready. activate with: conda activate hhcenv39"
echo " patches applied from repro/shared/patches/"
echo "==================================================="
