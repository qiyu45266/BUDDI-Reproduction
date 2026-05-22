#!/usr/bin/env bash
# Compute Pareto metrics + render 8x8 gallery from run_uncond.sh outputs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUDDI_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hhcenv39
cd "$BUDDI_ROOT"
export PYTHONPATH="$BUDDI_ROOT"

OUT_BASE="$BUDDI_ROOT/repro/phase_a/outputs/uncond"
CODE="$BUDDI_ROOT/repro/phase_a/code"

python "$CODE/analyze.py" --runs-dir "$OUT_BASE"

python "$CODE/make_gallery.py" \
  --renders-dir "$OUT_BASE/generate_1000_10_v0/renders" \
  --out         "$BUDDI_ROOT/repro/phase_a/outputs/gallery_uncond_baseline.png" \
  --cols 8 --max-samples 64

echo "==deliverables=="
ls -la "$OUT_BASE/results.csv" "$OUT_BASE/pareto.png" \
       "$OUT_BASE/diversity.png" "$BUDDI_ROOT/repro/phase_a/outputs/gallery_uncond_baseline.png" 2>&1
echo
echo "==results.csv=="
cat "$OUT_BASE/results.csv"
