#!/usr/bin/env bash
# RQ-5 stride sensitivity: run contact_vs_proximity.py at stride={10, 50}
# in addition to the existing stride=20 result, to check that the
# direction of the BUDDI-vs-CHI3D finding is robust to vertex-subsample choice.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hhcenv39
cd "$ROOT"
export PYTHONPATH="$ROOT"

REF="$ROOT/repro/phase_a/outputs/chi3d_distribution.pkl"
RUNS="$ROOT/repro/phase_a/outputs/uncond"

for S in 10 50; do
  OUT="$ROOT/repro/phase_c/outputs/rq5_stride${S}"
  echo
  echo "============================================================"
  echo " stride = $S → $OUT"
  echo "============================================================"
  python repro/phase_c/code/contact_vs_proximity.py \
    --buddi-runs "$RUNS" --chi3d-ref "$REF" --out-dir "$OUT" --stride "$S" 2>&1 | tail -25
done

echo
echo "==== aggregate: KS at 1cm and 2cm thresholds across strides ===="
python - <<'PY'
import json
from pathlib import Path
root = Path("repro/phase_c/outputs")
strides = []
for d in ["rq5", "rq5_stride10", "rq5_stride50"]:
    p = root / d / "ks_results.json"
    if not p.exists():
        print(f"  miss: {p}")
        continue
    ks = json.load(open(p))
    s = 20 if d == "rq5" else int(d.replace("rq5_stride", ""))
    strides.append((s, ks))

print(f"\n{'stride':>6} | {'n_steps':>7} | {'min_v2v p':>10} | {'@1cm p':>10} | {'@1cm mean(BUDDI/CHI3D)':>26} | {'@2cm p':>10} | {'@2cm mean(BUDDI/CHI3D)':>26}")
print("-" * 130)
for s, ks in sorted(strides):
    for n_steps_str, tests in sorted(ks["tests"].items(), key=lambda kv: int(kv[0])):
        n_steps = int(n_steps_str)
        mv = tests["min_v2v_m"]
        c1 = tests["contact_dens_10mm"]
        c2 = tests["contact_dens_20mm"]
        c1_ratio = f"{c1['mean_a']:.2e}/{c1['mean_b']:.2e}"
        c2_ratio = f"{c2['mean_a']:.2e}/{c2['mean_b']:.2e}"
        print(f"{s:>6} | {n_steps:>7} | {mv['p_value']:>10.3g} | {c1['p_value']:>10.3g} | {c1_ratio:>26} | {c2['p_value']:>10.3g} | {c2_ratio:>26}")
PY
