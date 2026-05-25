#!/usr/bin/env bash
# B1 — demo sanity fit: run the BUDDI conditional optimisation loop on
# one demo image (college_232334). Reduced to 50 iters/stage for speed.
#
# Wall-clock: ~20 min on RTX 4070 Laptop, sm_89.
# Output: per-image fits + visualisations under repro/phase_b/outputs/demo_fit/

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hhcenv39
cd "$ROOT"
export PYTHONPATH="$ROOT"

OUT_BASE="$ROOT/repro/phase_b/outputs"
mkdir -p "$OUT_BASE"

python llib/methods/hhcs_optimization/main.py \
  --exp-cfg llib/methods/hhcs_optimization/configs/buddi_cond_bev_demo.yaml \
  --exp-opts \
    logging.base_folder="$OUT_BASE" \
    logging.run=demo_fit \
    datasets.train_names=['demo'] \
    datasets.train_composition=[1.0] \
    datasets.demo.original_data_folder=demo/data/FlickrCI3D_Signatures/demo \
    datasets.demo.image_folder=images \
    datasets.demo.image_name_select=college_232334 \
    model.optimization.pretrained_diffusion_model_ckpt=essentials/buddi/buddi_cond_bev.pt \
    model.optimization.pretrained_diffusion_model_cfg=essentials/buddi/buddi_cond_bev.yaml \
    model.optimization.hhcs.max_iters='[50,50]' \
  2>&1 | tee "$OUT_BASE/b1_demo_fit.log"

echo
echo "==== B1 demo fit done ===="
echo "Outputs at: $OUT_BASE/demo_fit/"
ls -lh "$OUT_BASE/demo_fit/" 2>/dev/null || true
