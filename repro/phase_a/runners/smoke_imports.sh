#!/usr/bin/env bash
# Verify llib + all third-party imports work in hhcenv39.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUDDI_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hhcenv39
export PYTHONPATH="$BUDDI_ROOT"
cd "$BUDDI_ROOT"
python - <<'PY'
mods = [
    "llib.defaults.main",
    "llib.visualization.diffusion_eval",
    "llib.visualization.scripts.tools",
    "llib.methods.hhc_diffusion.evaluation.utils",
    "llib.models.build",
    "llib.bodymodels.build",
    "llib.cameras.build",
    "llib.visualization.renderer",
    "llib.models.diffusion.build",
    "llib.methods.hhc_diffusion.train_module",
]
bad = []
for m in mods:
    try: __import__(m); print("OK ", m)
    except Exception as e:
        bad.append((m, repr(e)[:120])); print("BAD", m, "->", type(e).__name__, str(e)[:120])
raise SystemExit(0 if not bad else 2)
PY
