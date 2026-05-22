#!/usr/bin/env python
"""
RQ-5 — Did BUDDI learn "contact" or just "proximity"?

For every mesh-pair sample, compute two metrics:
  - min_v2v            : minimum vertex-to-vertex distance between the two people
                         (a "proximity" signal — close-by people score low even with no contact)
  - contact_density_T  : fraction of (v_a, v_b) pairs with dist < T (T in {5mm, 1cm, 2cm})
                         (a "contact" signal — measures how much surface area is in contact)

We then KS-test BUDDI's distribution vs CHI3D contact-frame ground-truth distribution
on each metric. Hypothesis: if BUDDI matches CHI3D on min_v2v but NOT on contact_density,
the model learned proximity but not the (unlabeled) contact structure.

Inputs:
  --buddi-runs   repro/phase_a/outputs/uncond  (auto-discovers generate_*_v0/x_starts_smplx.pkl)
  --chi3d-ref    repro/phase_a/outputs/chi3d_distribution.pkl

Outputs (in --out-dir):
  stats.csv               per-sample metrics for each schedule + chi3d
  ks_results.json         KS test statistics + p-values for each (schedule, metric)
  kde_min_v2v.png         KDE plot: CHI3D vs each BUDDI schedule on min_v2v
  kde_contact_density.png same on contact_density at 1cm threshold
  density_comparison.png  combined 2-panel figure for the report
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats


THRESHOLDS_M = [0.005, 0.01, 0.02]  # 5mm, 1cm, 2cm in metres


def load_verts(pkl_path):
    """Returns (B, 2, V, 3) vertex tensor from either BUDDI x_starts_smplx.pkl or chi3d_distribution.pkl."""
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    if "final" in d and isinstance(d["final"], dict) and "vertices" in d["final"]:
        v = d["final"]["vertices"]
    elif "vertices" in d:
        v = d["vertices"]
    else:
        raise ValueError(f"no vertices in {pkl_path}")
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu()
    else:
        v = torch.from_numpy(np.asarray(v))
    return v.float()


def contact_stats(verts, stride=20, thresholds_m=THRESHOLDS_M, device="cuda", batch=64):
    """
    verts: (N, 2, V, 3) torch.Tensor on cpu.
    Returns (N, 1+len(thresholds_m)) numpy array per sample:
        col 0: min_v2v (m)
        cols 1..: contact_density at each threshold (fraction of subsampled pairs below T)
    """
    N, _, V, _ = verts.shape
    Vs = (V + stride - 1) // stride
    out = np.zeros((N, 1 + len(thresholds_m)), dtype=np.float32)
    v_sub = verts[:, :, ::stride, :].to(device)  # (N, 2, Vs, 3)
    for s in range(0, N, batch):
        e = min(s + batch, N)
        chunk = v_sub[s:e]              # (b, 2, Vs, 3)
        a = chunk[:, 0]                 # (b, Vs, 3)
        b = chunk[:, 1]                 # (b, Vs, 3)
        d = torch.cdist(a, b)           # (b, Vs, Vs)
        out[s:e, 0] = d.amin(dim=(1, 2)).cpu().numpy()
        for ti, T in enumerate(thresholds_m):
            out[s:e, 1 + ti] = (d < T).float().mean(dim=(1, 2)).cpu().numpy()
    return out  # (N, 1+T) array


def discover_buddi_runs(runs_dir):
    paths = sorted(glob.glob(os.path.join(runs_dir, "generate_*_v*", "x_starts_smplx.pkl")))
    runs = []
    for p in paths:
        name = Path(p).parent.name
        m = re.match(r"generate_(\d+)_(\d+)_v\d+", name)
        if not m:
            continue
        max_t, skip = int(m.group(1)), int(m.group(2))
        n_steps = (max_t - 1) // skip + 1
        runs.append({"path": p, "name": name, "n_steps": n_steps, "max_t": max_t, "skip": skip})
    runs.sort(key=lambda x: x["n_steps"])
    return runs


def ks_pair(a, b):
    """Two-sample KS test; returns dict with statistic, p_value, n_a, n_b."""
    a = np.asarray(a)
    b = np.asarray(b)
    s = stats.ks_2samp(a, b)
    return {
        "statistic": float(s.statistic),
        "p_value": float(s.pvalue),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
    }


def plot_kde(ax, data_dict, xlabel, title, xlim=None, clip_negative=True):
    """data_dict: {label: 1d_array}. CHI3D label first → black; BUDDI variants colored."""
    from scipy.stats import gaussian_kde
    for label, arr in data_dict.items():
        arr = np.asarray(arr)
        if clip_negative:
            arr = arr[arr >= 0]
        if len(arr) < 5:
            continue
        # cap extreme outliers for KDE stability
        lo, hi = np.percentile(arr, [1, 99])
        arr_kde = arr[(arr >= lo) & (arr <= hi)]
        if len(arr_kde) < 5:
            continue
        kde = gaussian_kde(arr_kde)
        x_grid = np.linspace(arr_kde.min(), arr_kde.max(), 300)
        color = "k" if "chi3d" in label.lower() else None
        lw = 2.2 if "chi3d" in label.lower() else 1.4
        ax.plot(x_grid, kde(x_grid), label=label, color=color, linewidth=lw)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.legend(fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buddi-runs", default="repro/phase_a/outputs/uncond")
    ap.add_argument("--chi3d-ref", default="repro/phase_a/outputs/chi3d_distribution.pkl")
    ap.add_argument("--out-dir", default="repro/phase_c/outputs/rq5")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    threshold_labels = [f"contact_dens_{int(T*1000)}mm" for T in THRESHOLDS_M]
    metric_cols = ["min_v2v_m"] + threshold_labels

    print("=== loading CHI3D reference ===", file=sys.stderr)
    chi3d_verts = load_verts(args.chi3d_ref)
    print(f"  CHI3D verts shape {tuple(chi3d_verts.shape)}", file=sys.stderr)
    chi3d_stats = contact_stats(chi3d_verts, stride=args.stride, device=args.device)
    print(f"  CHI3D min_v2v median: {np.median(chi3d_stats[:, 0])*1000:.2f} mm", file=sys.stderr)
    for ti, T in enumerate(THRESHOLDS_M):
        print(f"  CHI3D contact_density @ {int(T*1000)}mm median: {np.median(chi3d_stats[:, 1+ti]):.4f}", file=sys.stderr)

    print("\n=== loading BUDDI runs ===", file=sys.stderr)
    runs = discover_buddi_runs(args.buddi_runs)
    if not runs:
        print(f"no BUDDI runs under {args.buddi_runs}", file=sys.stderr)
        sys.exit(2)

    all_rows = []
    # chi3d rows
    for i in range(chi3d_stats.shape[0]):
        row = {"source": "chi3d", "n_steps": -1}
        for ci, col in enumerate(metric_cols):
            row[col] = float(chi3d_stats[i, ci])
        all_rows.append(row)

    buddi_stats_per_run = {}
    for r in runs:
        print(f"  loading {r['name']} (n_steps={r['n_steps']}) ...", file=sys.stderr)
        v = load_verts(r["path"])
        print(f"    verts shape {tuple(v.shape)}", file=sys.stderr)
        s = contact_stats(v, stride=args.stride, device=args.device)
        buddi_stats_per_run[r["n_steps"]] = s
        for i in range(s.shape[0]):
            row = {"source": f"buddi_{r['n_steps']}", "n_steps": r["n_steps"]}
            for ci, col in enumerate(metric_cols):
                row[col] = float(s[i, ci])
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df_path = Path(args.out_dir) / "stats.csv"
    df.to_csv(df_path, index=False)
    print(f"\nsaved {df_path}  ({len(df)} rows)", file=sys.stderr)

    # ---------- KS tests ----------
    print("\n=== KS tests (BUDDI schedule vs CHI3D) ===", file=sys.stderr)
    ks_results = {"thresholds_m": THRESHOLDS_M, "n_chi3d": int(chi3d_stats.shape[0]), "tests": {}}
    for n_steps, s in buddi_stats_per_run.items():
        ks_results["tests"][int(n_steps)] = {}
        for ci, col in enumerate(metric_cols):
            res = ks_pair(s[:, ci], chi3d_stats[:, ci])
            ks_results["tests"][int(n_steps)][col] = res
            verdict = "SAME" if res["p_value"] > 0.05 else "DIFFERENT"
            print(f"  n_steps={n_steps} {col}:  KS={res['statistic']:.3f}  p={res['p_value']:.4g}  "
                  f"  median_buddi={res['median_a']*(1000 if 'min_v2v' in col else 1):.4f}"
                  f"  median_chi3d={res['median_b']*(1000 if 'min_v2v' in col else 1):.4f}"
                  f"  -> {verdict}", file=sys.stderr)
    json_path = Path(args.out_dir) / "ks_results.json"
    with open(json_path, "w") as f:
        json.dump(ks_results, f, indent=2)
    print(f"\nsaved {json_path}", file=sys.stderr)

    # ---------- KDE plots ----------
    # min_v2v in mm
    fig, ax = plt.subplots(figsize=(8, 5))
    data = {"chi3d_ref (n=373)": chi3d_stats[:, 0] * 1000}
    for n_steps, s in sorted(buddi_stats_per_run.items()):
        data[f"buddi @ {n_steps} steps"] = s[:, 0] * 1000
    plot_kde(ax, data, xlabel="min v2v distance between the two people (mm)",
             title="RQ-5 (proximity probe): is BUDDI as close-contact as CHI3D?",
             xlim=(0, 50))
    fig.tight_layout()
    fig.savefig(Path(args.out_dir) / "kde_min_v2v.png", dpi=150)
    print(f"saved kde_min_v2v.png", file=sys.stderr)
    plt.close(fig)

    # contact_density at 1cm
    fig, ax = plt.subplots(figsize=(8, 5))
    data = {"chi3d_ref (n=373)": chi3d_stats[:, 2]}  # index 2 = 1cm threshold (5mm, 1cm, 2cm)
    for n_steps, s in sorted(buddi_stats_per_run.items()):
        data[f"buddi @ {n_steps} steps"] = s[:, 2]
    plot_kde(ax, data,
             xlabel="contact density @ 1cm threshold  (fraction of subsampled vertex pairs in contact)",
             title="RQ-5 (contact probe): does BUDDI match CHI3D's contact density?",
             xlim=(0, 0.005))
    fig.tight_layout()
    fig.savefig(Path(args.out_dir) / "kde_contact_density.png", dpi=150)
    print(f"saved kde_contact_density.png", file=sys.stderr)
    plt.close(fig)

    # Combined 2-panel for the report
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5))
    data_l = {"chi3d_ref (n=373)": chi3d_stats[:, 0] * 1000}
    data_r = {"chi3d_ref (n=373)": chi3d_stats[:, 2]}
    for n_steps, s in sorted(buddi_stats_per_run.items()):
        data_l[f"buddi @ {n_steps} steps"] = s[:, 0] * 1000
        data_r[f"buddi @ {n_steps} steps"] = s[:, 2]
    plot_kde(ax_l, data_l, "min v2v (mm)", "(a) Proximity — min vertex-to-vertex distance", xlim=(0, 50))
    plot_kde(ax_r, data_r, "contact density @ 1cm",
             "(b) Contact — fraction of (v_a, v_b) pairs in contact",
             xlim=(0, 0.005))
    fig.suptitle("RQ-5: BUDDI learned proximity vs contact (CHI3D contact-frame reference in black)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(Path(args.out_dir) / "density_comparison.png", dpi=150)
    print(f"saved density_comparison.png", file=sys.stderr)
    plt.close(fig)

    # Summary table
    print("\n=== SUMMARY ===", file=sys.stderr)
    print(f"  metric               | n_steps |     KS |     p   | verdict", file=sys.stderr)
    for n_steps in sorted(buddi_stats_per_run.keys()):
        for col in ["min_v2v_m", "contact_dens_10mm"]:
            r = ks_results["tests"][int(n_steps)][col]
            verdict = "SAME" if r["p_value"] > 0.05 else "DIFFERENT"
            print(f"  {col:<20} |   {n_steps:>3}   | {r['statistic']:.3f} | {r['p_value']:.3g} | {verdict}",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
