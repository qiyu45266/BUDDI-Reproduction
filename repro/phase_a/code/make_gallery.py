#!/usr/bin/env python
"""
Stitch generated 360-view GIFs (or single-frame PNGs) into a wall-friendly
gallery image for the report / slides.

Assumes sample.py was run with --save-vis, producing N gif files inside
<run_dir>/renders/00000_gen.gif, 00001_gen.gif, ...

Usage:
  python make_gallery.py --renders-dir outputs/person_a/uncond/generate_1000_10_v0/renders \
                        --out gallery_uncond.png --cols 8 --frame-idx 0
"""

import argparse
import glob
import os
from pathlib import Path

import imageio.v3 as iio
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=64)
    ap.add_argument("--frame-idx", type=int, default=0,
                    help="Which frame of the 360 gif to use (0=front)")
    args = ap.parse_args()

    gifs = sorted(glob.glob(os.path.join(args.renders_dir, "*_gen.gif")))[: args.max_samples]
    if not gifs:
        print("No *_gen.gif files found.")
        return

    print(f"Found {len(gifs)} samples; reading frame {args.frame_idx} from each")
    frames = []
    for g in gifs:
        try:
            f = iio.imread(g, index=args.frame_idx)
        except Exception:
            f = iio.imread(g, index=0)
        if f.ndim == 4:
            f = f[args.frame_idx % len(f)]
        if f.shape[-1] == 4:
            f = f[..., :3]
        frames.append(f)
    h0, w0 = frames[0].shape[:2]
    fixed = []
    for f in frames:
        h, w = f.shape[:2]
        if (h, w) != (h0, w0):
            pad = np.full((h0, w0, 3), 255, dtype=np.uint8)
            pad[:min(h, h0), :min(w, w0)] = f[:min(h, h0), :min(w, w0)]
            f = pad
        fixed.append(f)
    frames = fixed

    h, w = frames[0].shape[:2]
    rows = (len(frames) + args.cols - 1) // args.cols
    canvas = np.full((rows * h, args.cols * w, 3), 255, dtype=np.uint8)
    for i, f in enumerate(frames):
        r, c = i // args.cols, i % args.cols
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = f

    iio.imwrite(args.out, canvas)
    print(f"Saved {args.out}  ({rows}x{args.cols} grid)")


if __name__ == "__main__":
    main()
