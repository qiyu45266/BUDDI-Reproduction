#!/usr/bin/env bash
# RQ-7: counterfactual contact restoration probe.
# Single-step diffuse-denoise at t=10 (paper §3.2 Eq.6) on perturbed CHI3D anchors.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUDDI_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hhcenv39
cd "$BUDDI_ROOT"
export PYTHONPATH="$BUDDI_ROOT"
export PYOPENGL_PLATFORM=egl

REF="$BUDDI_ROOT/repro/phase_a/outputs/chi3d_distribution.pkl"
[ -f "$REF" ] || { echo "missing $REF — run repro/phase_a/runners/run_chi3d_fid.sh first"; exit 1; }

OUT="$BUDDI_ROOT/repro/phase_c/outputs/rq7"

python repro/phase_c/code/counterfactual_contact.py \
  --chi3d-ref "$REF" \
  --n-anchors 100 \
  --batch-size 25 \
  --out-dir   "$OUT" \
  "$@"

echo
echo "==deliverables=="
ls -la "$OUT/"
