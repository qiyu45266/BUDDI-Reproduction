#!/usr/bin/env bash
# Download SMPL-X / SMPL / SMIL body models.
#
# Required env vars (you MUST register on the websites and accept license terms first):
#   SMPLX_USER, SMPLX_PASS  — https://smpl-x.is.tue.mpg.de
#   SMPL_USER,  SMPL_PASS   — https://smpl.is.tue.mpg.de
#
# Usage:
#   SMPLX_USER=... SMPLX_PASS=... SMPL_USER=... SMPL_PASS=... \
#     bash repro/shared/env/3_fetch_bodymodels.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUDDI_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$BUDDI_ROOT"

if [ -d essentials/body_models/smplx ] && [ -f essentials/body_models/smpl/SMPL_NEUTRAL.pkl ]; then
  echo "body models already present — nothing to do"
  ls essentials/body_models/
  exit 0
fi

: "${SMPLX_USER:?need SMPLX_USER env var — register at https://smpl-x.is.tue.mpg.de}"
: "${SMPLX_PASS:?need SMPLX_PASS env var}"
: "${SMPL_USER:?need SMPL_USER env var — register at https://smpl.is.tue.mpg.de}"
: "${SMPL_PASS:?need SMPL_PASS env var}"

mkdir -p essentials/body_models
cd essentials/body_models

# URL-encode reserved chars
urle () { local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }
PY="$HOME/miniconda3/envs/hhcenv39/bin/python"
[ -x "$PY" ] || PY="$HOME/miniconda3/bin/python"

unzip_py () {
  local zip="$1"
  $PY - "$zip" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z: z.extractall(".")
PY
}

# ---------- SMPL-X (~870 MB) ----------
echo "[1/3] SMPL-X ..."
USR_ENC=$(urle "$SMPLX_USER"); PASS_ENC=$(urle "$SMPLX_PASS")
wget --post-data "username=$USR_ENC&password=$PASS_ENC" \
  'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip&resume=1' \
  -O models_smplx_v1_1.zip --no-check-certificate --continue 2>&1 | tail -3
[ -s models_smplx_v1_1.zip ] || { echo "SMPL-X download empty/failed — check credentials"; exit 1; }
unzip_py models_smplx_v1_1.zip
mv -f models/smplx ./smplx
rm -rf models models_smplx_v1_1.zip

# ---------- SMPL (~40 MB) ----------
echo "[2/3] SMPL ..."
USR_ENC=$(urle "$SMPL_USER"); PASS_ENC=$(urle "$SMPL_PASS")
wget --post-data "username=$USR_ENC&password=$PASS_ENC" \
  'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip&resume=1' \
  -O models_smpl.zip --no-check-certificate --continue 2>&1 | tail -3
[ -s models_smpl.zip ] || { echo "SMPL download empty/failed — check credentials"; exit 1; }
unzip_py models_smpl.zip
mkdir -p smpl
cp 'SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl' smpl/SMPL_NEUTRAL.pkl
rm -rf SMPL_python_v.1.1.0 models_smpl.zip

# ---------- SMIL (~40 MB, public — no credentials) ----------
echo "[3/3] SMIL ..."
mkdir -p smil_temp && cd smil_temp
wget -q 'https://obj-web.iosb.fraunhofer.de/content/sensornetze/bewegungsanalyse/smil.zip' -O smil_web.zip --no-check-certificate --continue
unzip_py smil_web.zip
mkdir -p ../smil
mv -f smil/smil_web.pkl ../smil/
cd ..
rm -rf smil_temp

echo
echo "==result=="
ls -la essentials/body_models/ 2>/dev/null || ls -la
echo
ls smplx | head 2>/dev/null
ls smpl 2>/dev/null
ls smil 2>/dev/null
