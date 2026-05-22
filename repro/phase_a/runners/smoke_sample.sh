#!/usr/bin/env bash
# Tiny 8-sample smoke test — run before the full run_uncond.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUDDI_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SEED="${SEED:-42}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hhcenv39
cd "$BUDDI_ROOT"
export PYTHONPATH="$BUDDI_ROOT"
export PYOPENGL_PLATFORM=egl

OUT="$BUDDI_ROOT/repro/phase_a/outputs/smoke"
rm -rf "$OUT"; mkdir -p "$OUT"

python -u llib/methods/hhc_diffusion/evaluation/sample.py \
  --exp-cfg essentials/buddi/buddi_unconditional.yaml \
  --checkpoint-name essentials/buddi/buddi_unconditional.pt \
  --output-folder "$OUT" \
  --num-samples 8 --max-t 1000 --skip-steps 100 \
  --batch_size 8 --log-steps 100 --seed "$SEED" \
  --save-vis --max-images-render=4 2>&1 | tail -40

echo "==result=="
find "$OUT" -maxdepth 3 -type f | head
