#!/usr/bin/env bash
# Build CHI3D contact-frame distribution from raw SMPL-X JSONs, then
# compute FID between each BUDDI schedule and that distribution.
# Output: chi3d_distribution.pkl, fid_vs_chi3d.csv, pareto_v2.png, and
# results.csv gets a new fid_vs_chi3d column.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUDDI_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hhcenv39
cd "$BUDDI_ROOT"
export PYTHONPATH="$BUDDI_ROOT"

[ -d datasets/original/CHI3D/train ] || {
  echo "missing datasets/original/CHI3D/train — extract chi3d_train.tar.gz first"
  exit 1
}

OUT_DIR="$BUDDI_ROOT/repro/phase_a/outputs"

echo "==== 1/3) build CHI3D contact-frame distribution ===="
python repro/phase_a/code/chi3d_distribution.py \
  --root datasets/original/CHI3D/train \
  --subjects s02 s03 s04 \
  --out "$OUT_DIR/chi3d_distribution.pkl"

echo
echo "==== 2/3) compute FID per schedule ===="
python repro/phase_a/code/eval_fid_chi3d.py \
  --ref "$OUT_DIR/chi3d_distribution.pkl" \
  --runs-dir "$OUT_DIR/uncond" \
  --out-csv "$OUT_DIR/uncond/fid_vs_chi3d.csv" \
  --merge-into-results

NOISE_FLOOR=$(awk -F',' 'NR==2 {print $6}' "$OUT_DIR/uncond/fid_vs_chi3d.csv")
echo
echo "==== 3/3) render 2-axis Pareto plot ===="
python repro/phase_a/code/plot_pareto_v2.py \
  --results "$OUT_DIR/uncond/results.csv" \
  --out     "$OUT_DIR/uncond/pareto_v2.png" \
  --noise-floor "$NOISE_FLOOR"

echo
echo "==== results.csv ===="
cat "$OUT_DIR/uncond/results.csv"
