#!/usr/bin/env python
"""
Compute FID between BUDDI samples (from each DDIM schedule) and a CHI3D
reference distribution, using the same featurization the paper uses
(llib/methods/hhc_diffusion/evaluation/eval.py:fid_on_params).

Output: extends repro/phase_a/outputs/uncond/results.csv with a `fid_vs_chi3d`
column and writes a small `fid_vs_chi3d.csv` summary.

Caveat (per REPRODUCIBILITY_REPORT.md §2.3): reference distribution is CHI3D
only (~350 contact-frame pairs), not the paper's 60/20/20 Flickr/CHI3D/Hi4D
mix over 8K samples. Absolute number is therefore not directly comparable to
the paper's 1.6, but the cross-schedule trend is informative.
"""

import argparse
import glob
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from llib.methods.hhc_diffusion.evaluation.eval import fid_featurize, params_for_fid
from llib.utils.metrics.diffusion import GenFID


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def to_samples_dict(buddi_pkl_or_chi3d_pkl):
    """Either format ships as {'final': {...}}; unwrap."""
    if "final" in buddi_pkl_or_chi3d_pkl and isinstance(buddi_pkl_or_chi3d_pkl["final"], dict):
        return buddi_pkl_or_chi3d_pkl["final"]
    return buddi_pkl_or_chi3d_pkl


def fid_against(ref_dict, gen_dict, device="cuda", num_samples=None):
    """Featurize both, stack into single tensor each, compute Frechet distance."""
    if num_samples is not None:
        n_ref = min(num_samples, ref_dict["global_orient"].shape[0])
        n_gen = min(num_samples, gen_dict["global_orient"].shape[0])
    else:
        n_ref = ref_dict["global_orient"].shape[0]
        n_gen = gen_dict["global_orient"].shape[0]

    ref_sub = {k: v[:n_ref] for k, v in ref_dict.items() if k != "vertices"}
    gen_sub = {k: v[:n_gen] for k, v in gen_dict.items() if k != "vertices"}

    # Ensure betas is 10d (CHI3D loader emits 10) and shape h is built same way
    # fid_featurize expects betas:(B,2,10), scale:(B,2,1) — both already present
    ref_feat = fid_featurize(ref_sub, n_ref, device=device)
    gen_feat = fid_featurize(gen_sub, n_gen, device=device)

    fid_metric = GenFID()  # no autoencoder ckpt — only uses calculate_frechet_distance
    mu_a, cov_a, mu_b, cov_b = _stats_pair(ref_feat, gen_feat)
    return fid_metric.calculate_frechet_distance(mu_a, cov_a, mu_b, cov_b)


def _stats_pair(a_feat, b_feat):
    a = params_for_fid(a_feat).cpu().numpy()
    b = params_for_fid(b_feat).cpu().numpy()
    mu_a, mu_b = np.mean(a, axis=0), np.mean(b, axis=0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    return mu_a, cov_a, mu_b, cov_b


def bootstrap_noise_floor(ref_dict, n_iters=20, device="cuda"):
    """Estimate the FID's statistical floor by random-half-split of the reference."""
    n = ref_dict["global_orient"].shape[0]
    half = n // 2
    fids = []
    for i in range(n_iters):
        rng = np.random.RandomState(1000 + i)
        idx = rng.permutation(n)
        a_idx, b_idx = idx[:half], idx[half: 2 * half]
        a = {k: v[a_idx] for k, v in ref_dict.items() if k != "vertices"}
        b = {k: v[b_idx] for k, v in ref_dict.items() if k != "vertices"}
        fid = fid_against(a, b, device=device)
        fids.append(fid)
    return float(np.mean(fids)), float(np.std(fids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="repro/phase_a/outputs/chi3d_distribution.pkl",
                    help="Reference distribution pkl from chi3d_distribution.py")
    ap.add_argument("--runs-dir", default="repro/phase_a/outputs/uncond",
                    help="Folder with generate_<maxT>_<skip>_v*/x_starts_smplx.pkl")
    ap.add_argument("--out-csv", default="repro/phase_a/outputs/uncond/fid_vs_chi3d.csv")
    ap.add_argument("--merge-into-results", action="store_true",
                    help="Also add fid_vs_chi3d column to results.csv in --runs-dir")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"loading CHI3D reference: {args.ref}", file=sys.stderr)
    ref = to_samples_dict(load_pkl(args.ref))
    print(f"  reference N = {ref['global_orient'].shape[0]} contact-frame mesh pairs", file=sys.stderr)

    print(f"computing bootstrap noise floor (20 random half-splits) ...", file=sys.stderr)
    floor_mean, floor_std = bootstrap_noise_floor(ref, n_iters=20, device=args.device)
    print(f"  noise floor: {floor_mean:.4f} ± {floor_std:.4f}", file=sys.stderr)

    runs = sorted(glob.glob(os.path.join(args.runs_dir, "generate_*_v*", "x_starts_smplx.pkl")))
    rows = []
    for r in runs:
        run_name = Path(r).parent.name
        m = re.match(r"generate_(\d+)_(\d+)_v\d+", run_name)
        n_steps = (int(m.group(1)) - 1) // int(m.group(2)) + 1 if m else -1
        print(f"\nfid for {run_name}  (n_steps={n_steps})", file=sys.stderr)
        gen = to_samples_dict(load_pkl(r))
        fid = fid_against(ref, gen, device=args.device)
        print(f"  FID vs CHI3D = {fid:.4f}", file=sys.stderr)
        rows.append({
            "run": run_name,
            "n_steps": n_steps,
            "n_buddi_samples": int(gen["global_orient"].shape[0]),
            "n_chi3d_pairs": int(ref["global_orient"].shape[0]),
            "fid_vs_chi3d": float(fid),
            "noise_floor_mean": floor_mean,
            "noise_floor_std": floor_std,
        })

    df = pd.DataFrame(rows).sort_values("n_steps")
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nsaved {args.out_csv}\n", file=sys.stderr)
    print(df.to_string(index=False))

    if args.merge_into_results:
        results_path = Path(args.runs_dir) / "results.csv"
        if results_path.exists():
            base = pd.read_csv(results_path)
            merged = base.merge(df[["n_steps", "fid_vs_chi3d"]], on="n_steps", how="left")
            merged.to_csv(results_path, index=False)
            print(f"\nmerged fid_vs_chi3d column into {results_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
