#!/usr/bin/env bash
# RQ-5: contact-vs-proximity probe on BUDDI samples vs CHI3D contact-frame reference.
# Pre-req: Phase A's chi3d_distribution.pkl + uncond/generate_*_v0/x_starts_smplx.pkl
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUDDI_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hhcenv39
cd "$BUDDI_ROOT"
export PYTHONPATH="$BUDDI_ROOT"

REF="$BUDDI_ROOT/repro/phase_a/outputs/chi3d_distribution.pkl"
RUNS="$BUDDI_ROOT/repro/phase_a/outputs/uncond"
OUT="$BUDDI_ROOT/repro/phase_c/outputs/rq5"

[ -f "$REF" ] || { echo "missing $REF — run repro/phase_a/runners/run_chi3d_fid.sh first"; exit 1; }
[ -d "$RUNS" ] || { echo "missing $RUNS — run repro/phase_a/runners/run_uncond.sh first"; exit 1; }

python repro/phase_c/code/contact_vs_proximity.py \
  --buddi-runs "$RUNS" \
  --chi3d-ref  "$REF" \
  --out-dir    "$OUT" \
  --stride 20

echo
echo "==deliverables=="
ls -la "$OUT/"
echo
echo "==ks_results.json (head) =="
python -c "import json; d=json.load(open('$OUT/ks_results.json')); [print(k, v) for k, v in d['tests'].items()]" | head -8
