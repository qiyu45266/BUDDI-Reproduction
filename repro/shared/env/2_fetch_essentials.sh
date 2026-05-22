#!/usr/bin/env bash
# Download author-distributed essentials.zip (~60 MB) → essentials/{buddi,contact,priors,body_model_utils}
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUDDI_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$BUDDI_ROOT"

if [ -d essentials/buddi ] && [ -f essentials/buddi/buddi_unconditional.pt ]; then
  echo "essentials/buddi already populated — nothing to do"
  ls -la essentials/buddi/ | head
  exit 0
fi

# Use miniconda base python for gdown (does not require hhcenv39)
CONDA_PY="$HOME/miniconda3/bin/python"
CONDA_PIP="$HOME/miniconda3/bin/pip"
[ -x "$CONDA_PY" ] || { echo "miniconda not found — run repro/shared/env/0_install_miniconda.sh first"; exit 1; }
$CONDA_PY -c "import gdown" 2>/dev/null || $CONDA_PIP install --quiet gdown

if [ ! -f essentials.zip ]; then
  echo "downloading essentials.zip via gdown ..."
  "$HOME/miniconda3/bin/gdown" 1MsYaHuX2w7GQ7e3OzPckyJ9sxBnf6wE8 -O essentials.zip
fi

echo "unzipping (python zipfile) ..."
$CONDA_PY - <<'PY'
import zipfile
with zipfile.ZipFile("essentials.zip") as z:
    z.extractall(".")
print("extracted")
PY
rm -f essentials.zip
[ -d __MACOSX ] && rm -rf __MACOSX

echo "===================="
echo "essentials/ tree:"
find essentials -maxdepth 2 -type d | sort
echo "----"
ls -la essentials/buddi/
