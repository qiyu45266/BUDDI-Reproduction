#!/usr/bin/env bash
# OPTIONAL — only needed for Phase A v2 (FID vs CHI3D real-GT distribution).
#
# Extracts the 4 ci3d.imar.ro tar archives into the standard layout that
# upstream DATA.md (§FlickrCI3D and §CHI3D) expects under datasets/original/.
#
# Pre-req: you must register at https://ci3d.imar.ro/download, download these
# 4 files yourself, and place them in this repo's datasets/ folder:
#   - FlickrCI3D_signature_train-003.tar.gz   (~3.2 GB)
#   - FlickrCI3D_signature_test.tar.gz        (~344 MB)
#   - chi3d_train.tar.gz                      (~4.2 GB)
#   - chi3d_test.tar.gz                       (~678 MB)
#
# We do not redistribute the tarballs — they are CC-BY-NC academic data.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SRC="$ROOT/datasets"
DST="$ROOT/datasets/original"

mkdir -p "$DST/FlickrCI3D_Signatures" "$DST/CHI3D"

extract_once() {
  local tar="$1" out="$2" sentinel="$3"
  if [ -d "$sentinel" ]; then
    echo "[skip] $sentinel already exists"
    return
  fi
  if [ ! -f "$tar" ]; then
    echo "[skip] missing $tar"
    return
  fi
  echo "[extract] $tar -> $out"
  tar -xzf "$tar" -C "$out"
}

extract_once "$SRC/FlickrCI3D_signature_train-003.tar.gz" "$DST/FlickrCI3D_Signatures" "$DST/FlickrCI3D_Signatures/train"
extract_once "$SRC/FlickrCI3D_signature_test.tar.gz"      "$DST/FlickrCI3D_Signatures" "$DST/FlickrCI3D_Signatures/test"
extract_once "$SRC/chi3d_train.tar.gz" "$DST/CHI3D" "$DST/CHI3D/train"
extract_once "$SRC/chi3d_test.tar.gz"  "$DST/CHI3D" "$DST/CHI3D/test"

echo
echo '==result=='
echo "FlickrCI3D_Signatures/:"; ls "$DST/FlickrCI3D_Signatures/" 2>/dev/null
for s in train test; do
  d="$DST/FlickrCI3D_Signatures/$s"
  [ -d "$d" ] && echo "  $s: $(ls "$d/images/" 2>/dev/null | wc -l) images, json=$(ls "$d/"*.json 2>/dev/null | head -1)"
done
echo
echo "CHI3D/:"; ls "$DST/CHI3D/" 2>/dev/null
for s in train test; do
  d="$DST/CHI3D/$s"
  if [ -d "$d" ]; then
    echo "  $s subjects: $(ls "$d" 2>/dev/null | tr '\n' ' ')"
    for ss in "$d"/*/; do
      [ -d "$ss" ] && echo "    $(basename "$ss"): $(ls "$ss" | tr '\n' ' ')"
    done
  fi
done
echo
du -sh "$DST"/* 2>/dev/null
