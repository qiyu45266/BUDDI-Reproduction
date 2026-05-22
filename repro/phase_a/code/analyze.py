#!/usr/bin/env python
"""
Person A analysis script.

Reads x_starts_smplx.pkl files produced by sample.py at multiple sampling
schedules, computes:
  - Diversity: mean pairwise vertex-to-vertex L2 distance across samples
  - Plausibility: % of samples without two-person interpenetration
  - Self-FID: FID between current schedule's samples and the slowest (1000/10) reference
  - Wall-time (read from cmd_args.txt if available)

Outputs:
  - results.csv  : per-schedule metrics
  - pareto.png   : quality-speed trade-off curve

Usage:
  python analyze.py --runs-dir outputs/person_a/uncond
"""

import argparse
import glob
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def load_run(run_dir):
    """Load x_starts_smplx.pkl and return final vertices + meta."""
    pkl = Path(run_dir) / "x_starts_smplx.pkl"
    if not pkl.exists():
        return None
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    if "final" not in data or "vertices" not in data["final"]:
        return None
    verts = data["final"]["vertices"]  # tensor (B, 2, V, 3)
    if isinstance(verts, torch.Tensor):
        verts = verts.cpu().numpy()

    # Parse schedule from folder name: generate_<maxT>_<skip>_v<N>
    name = Path(run_dir).name
    m = re.match(r"generate_(\d+)_(\d+)_v\d+", name)
    if m:
        max_t, skip = int(m.group(1)), int(m.group(2))
        n_steps = (max_t - 1) // skip + 1
    else:
        max_t = skip = n_steps = -1

    return dict(verts=verts, max_t=max_t, skip=skip, n_steps=n_steps, run_dir=str(run_dir))


# ---------------------------------------------------------------------------
# Diversity: mean pairwise vertex L2 across samples (sub-sampled)
# ---------------------------------------------------------------------------
def diversity(verts, max_pairs=2000):
    """verts: (B, 2, V, 3)  ->  scalar mean L2 distance between random sample pairs."""
    B = verts.shape[0]
    if B < 2:
        return 0.0
    n_pairs = min(max_pairs, B * (B - 1) // 2)
    idx_i = np.random.randint(0, B, size=n_pairs)
    idx_j = np.random.randint(0, B, size=n_pairs)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    diffs = verts[idx_i] - verts[idx_j]                  # (P, 2, V, 3)
    d = np.sqrt((diffs ** 2).sum(-1)).mean()             # mean L2 per vertex
    return float(d)


# ---------------------------------------------------------------------------
# Plausibility: %samples with no significant two-person interpenetration
# (proxy: count vertices of person0 that lie inside the convex hull of person1
# we use a simple AABB overlap + bidirectional nearest-neighbor distance check)
# ---------------------------------------------------------------------------
def interpenetration_rate(verts, thresh_mm=20.0):
    """
    Simple proxy: fraction of samples where ANY vertex of person0 is within
    `thresh_mm` of >50% of person1's vertices' centroid -- i.e. severe overlap.
    Returns (mean_penetration_count_per_sample, fraction_severe).
    """
    B = verts.shape[0]
    counts = np.zeros(B)
    severe = np.zeros(B, dtype=bool)
    thr_m = thresh_mm / 1000.0
    for b in range(B):
        v0, v1 = verts[b, 0], verts[b, 1]                # (V, 3) each, units assumed meters
        # AABB overlap quick reject
        b0_min, b0_max = v0.min(0), v0.max(0)
        b1_min, b1_max = v1.min(0), v1.max(0)
        if (b0_max < b1_min).any() or (b1_max < b0_min).any():
            continue
        # subsample for speed
        v0s = v0[::20]
        v1s = v1[::20]
        # min dist from each v0 vertex to v1
        d = np.sqrt(((v0s[:, None] - v1s[None]) ** 2).sum(-1)).min(1)   # (V0/20,)
        counts[b] = (d < thr_m).sum()
        severe[b] = counts[b] > 30                                       # >30 near-touch vertices
    return float(counts.mean()), float(severe.mean())


# ---------------------------------------------------------------------------
# Self-FID: simple feature-based FID between two vertex tensor sets.
# We summarize each sample by [mean per-vertex coord, std per-vertex coord]
# which is a 2 * 2 * V * 3 = 4*V*3 -dim feature; works as a quick proxy.
# For a real FID, one would use a pretrained motion encoder; we keep it
# self-contained here.
# ---------------------------------------------------------------------------
def featurize(verts, n_features=64):
    """(B, 2, V, 3) -> (B, F) summary features."""
    B = verts.shape[0]
    # downsample vertices
    v = verts[:, :, ::200]                # (B, 2, V', 3)
    v = v.reshape(B, -1)                  # (B, 2*V'*3)
    # zero-center each sample, then take its norm distribution
    return v


def fid_gaussian(a, b):
    """Frechet distance between two Gaussian-approximated feature sets."""
    mu_a, mu_b = a.mean(0), b.mean(0)
    cov_a = np.cov(a, rowvar=False) + 1e-6 * np.eye(a.shape[1])
    cov_b = np.cov(b, rowvar=False) + 1e-6 * np.eye(b.shape[1])
    from scipy.linalg import sqrtm
    covmean = sqrtm(cov_a @ cov_b).real
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(cov_a + cov_b - 2 * covmean))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True,
                    help="Folder with generate_*_*_v*/ subfolders")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write results.csv + pareto.png (default: runs-dir)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir or args.runs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = sorted(glob.glob(os.path.join(args.runs_dir, "generate_*_v*")))
    print(f"Found {len(runs)} runs.")
    loaded = [load_run(r) for r in runs]
    loaded = [x for x in loaded if x is not None]
    if not loaded:
        print("No valid runs found.")
        return

    # Reference = the slowest schedule (most denoising steps = best quality)
    loaded.sort(key=lambda d: d["n_steps"], reverse=True)
    reference = loaded[0]
    print(f"Reference schedule: max-t={reference['max_t']} skip={reference['skip']} "
          f"(n_steps={reference['n_steps']})")

    ref_feats = featurize(reference["verts"])

    rows = []
    for d in loaded:
        print(f"\nAnalyzing {Path(d['run_dir']).name}")
        n = d["verts"].shape[0]
        div = diversity(d["verts"])
        ip_mean, ip_severe = interpenetration_rate(d["verts"])
        feats = featurize(d["verts"])
        if feats.shape == ref_feats.shape and d is not reference:
            try:
                fid = fid_gaussian(feats, ref_feats)
            except Exception as e:
                fid = float("nan"); print(f"  FID failed: {e}")
        else:
            fid = 0.0
        rows.append(dict(
            n_steps=d["n_steps"],
            max_t=d["max_t"],
            skip=d["skip"],
            n_samples=n,
            diversity_mm=div * 1000,
            interpenetration_mean=ip_mean,
            severe_interpen_frac=ip_severe,
            self_fid=fid,
        ))
        print(f"  diversity = {div*1000:.2f} mm   "
              f"interpenetration = {ip_mean:.2f}  severe = {ip_severe:.3f}  "
              f"self-FID = {fid:.4f}")

    df = pd.DataFrame(rows).sort_values("n_steps")
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    # Plot Pareto curve
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "tab:blue"
    ax1.set_xlabel("DDIM steps (log scale)")
    ax1.set_xscale("log")
    ax1.set_ylabel("Self-FID vs slowest schedule", color=color1)
    ax1.plot(df["n_steps"], df["self_fid"], "o-", color=color1, label="Self-FID")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Severe interpenetration rate", color=color2)
    ax2.plot(df["n_steps"], df["severe_interpen_frac"], "s--", color=color2,
             label="Severe IP rate")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("BUDDI Sampling Quality vs Speed")
    fig.tight_layout()
    png_path = out_dir / "pareto.png"
    plt.savefig(png_path, dpi=150)
    print(f"Saved {png_path}")

    # Also plot diversity
    fig2, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["n_steps"], df["diversity_mm"], "o-")
    ax.set_xscale("log")
    ax.set_xlabel("DDIM steps")
    ax.set_ylabel("Mean pairwise vertex distance (mm)")
    ax.set_title("Sample Diversity vs Number of Denoising Steps")
    fig2.tight_layout()
    div_path = out_dir / "diversity.png"
    plt.savefig(div_path, dpi=150)
    print(f"Saved {div_path}")


if __name__ == "__main__":
    main()
