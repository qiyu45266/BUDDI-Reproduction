#!/usr/bin/env bash
set -euo pipefail
cd "$HOME"
if [ ! -f miniconda.sh ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  echo "downloaded"
fi
if [ ! -d "$HOME/miniconda3" ]; then
  bash miniconda.sh -b -p "$HOME/miniconda3"
  echo "installed"
else
  echo "miniconda3 already exists"
fi
"$HOME/miniconda3/bin/conda" init bash >/dev/null
echo "init done"
"$HOME/miniconda3/bin/conda" --version
