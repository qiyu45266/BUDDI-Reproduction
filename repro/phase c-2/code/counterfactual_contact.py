#!/usr/bin/env python
"""
RQ-7 — Counterfactual contact restoration.

For each CHI3D contact-frame anchor, perturb person-A's translation in 6 directions × 6
magnitudes (1-50 cm), then run BUDDI's diffuse-denoise single-step at t=10 (the exact
operation used in paper §3.2 / Eq. 6 during optimisation). Measure:

  - contact restoration : fraction of (perturbation × anchor) where min_v2v < 1 cm
  - A correction       : ||denoised_A_verts - anchor_A_verts||  (how much BUDDI moved A back)
  - B innocent motion  : ||denoised_B_verts - anchor_B_verts||  (collateral movement on B)

Compares two models:
  - buddi_unconditional.pt   (no cH)
  - buddi_cond_bev.pt        (cH = perturbed anchor, mimicking a noisy BEV)

Outputs:
  restoration_table.csv     per-(model, anchor, direction, magnitude) raw rows
  restoration_curve.png     contact-recovery rate vs perturbation magnitude
  correction_curve.png      A correction & B motion vs magnitude
  summary.json              aggregate statistics
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from llib.utils.threed.conversion import axis_angle_to_rotation6d


# ---------------------------------------------------------------------------
# Diffusion module setup (via argv-hack on sample.py:parse_args)
# ---------------------------------------------------------------------------
def load_diffusion_module(ckpt_path, cfg_yaml, batch_size, device="cuda"):
    """Reuse llib's setup_diffusion_module via the sample.py CLI parser."""
    import os as _os, sys as _sys
    orig_argv = _sys.argv[:]
    tmp_out = "/tmp/rq7_dummy_logs"
    _os.makedirs(tmp_out, exist_ok=True)
    _sys.argv = [
        "rq7",
        "--exp-cfg", cfg_yaml,
        "--checkpoint-name", ckpt_path,
        "--output-folder", tmp_out,
        "--batch_size", str(batch_size),
    ]
    try:
        from llib.methods.hhc_diffusion.evaluation.sample import parse_args
        from llib.methods.hhc_diffusion.evaluation.utils import setup_diffusion_module

        cfg, cmd_args = parse_args()
        cfg.device = device
        mod = setup_diffusion_module(cfg, cmd_args)
        return mod, cfg, cmd_args
    finally:
        _sys.argv = orig_argv


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def min_v2v_per_pair(verts, stride=20):
    """verts: (B, 2, V, 3) -> (B,) min vertex-to-vertex distance using stride-subsample."""
    v_sub = verts[:, :, ::stride, :]  # (B, 2, Vs, 3)
    a, b = v_sub[:, 0], v_sub[:, 1]
    d = torch.cdist(a, b)             # (B, Vs, Vs)
    return d.amin(dim=(1, 2))


def mean_per_vertex_dist(v_a, v_b):
    """v_a, v_b: (B, V, 3) -> (B,) mean over verts of L2(v_a - v_b)."""
    return (v_a - v_b).norm(dim=-1).mean(dim=-1)


def rename_to_module_keys(anchor_aa):
    """
    Rename chi3d_distribution.pkl keys to the dict keys cast_smpl/diffuse_denoise expect:
      global_orient → orient   (axis-angle still)
      body_pose     → pose     (axis-angle still)
      betas+scale   → shape
      transl        → transl
    cast_smpl will later convert orient/pose AA→rot6d and apply relative_transl.
    """
    return {
        "orient": anchor_aa["global_orient"].clone(),    # (N, 2, 3) AA
        "pose":   anchor_aa["body_pose"].clone(),        # (N, 2, 63) AA
        "shape":  torch.cat([anchor_aa["betas"], anchor_aa["scale"]], dim=-1).clone(),  # (N, 2, 11)
        "transl": anchor_aa["transl"].clone(),           # (N, 2, 3)
    }


def batched_cast_smpl(mod, aa_params_cpu, batch_size, device):
    """
    Apply mod.cast_smpl in chunks of `batch_size`. After this:
      orient/pose are rotation-6d
      transl[0] = 0, transl[1] = offset from A   (because relative_transl=True)
    Returns dict of cpu tensors.
    """
    N = aa_params_cpu["orient"].shape[0]
    out = {k: [] for k in ["orient", "pose", "shape", "transl"]}
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        chunk_cpu = {k: v[s:e] for k, v in aa_params_cpu.items()}
        if (e - s) < batch_size:
            pad = batch_size - (e - s)
            chunk_cpu = {k: torch.cat([v, v[-1:].repeat([pad] + [1] * (v.ndim - 1))], dim=0)
                         for k, v in chunk_cpu.items()}
        else:
            pad = 0
        chunk_dev = {k: v.to(device) for k, v in chunk_cpu.items()}
        with torch.no_grad():
            chunk_cast = mod.cast_smpl(chunk_dev)
        for k in out:
            v = chunk_cast[k].detach().cpu()
            out[k].append(v[: v.shape[0] - pad] if pad else v)
    return {k: torch.cat(vs, dim=0) for k, vs in out.items()}


def batched_get_smpl(mod, params, batch_size, device):
    """
    Run mod.get_smpl in chunks of `batch_size`. Returns (N, 2, V, 3) on CPU.
    Always moves chunk to `device` before calling the body model.
    """
    N = params["orient"].shape[0]
    out = []
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        chunk = {k: v[s:e] for k, v in params.items()}
        if (e - s) < batch_size:
            chunk = {k: torch.cat([v, v[-1:].repeat([batch_size - (e - s)] + [1] * (v.ndim - 1))], dim=0)
                     for k, v in chunk.items()}
            pad = batch_size - (e - s)
        else:
            pad = 0
        chunk = {k: v.to(device) for k, v in chunk.items()}
        with torch.no_grad():
            sm0, sm1 = mod.get_smpl(chunk)
            v = torch.stack([sm0.vertices, sm1.vertices], dim=1).detach().cpu()
        out.append(v[: v.shape[0] - pad] if pad else v)
    return torch.cat(out, dim=0)


def _chunk_dict(d, s, e, pad, like_chunk):
    """Slice d[s:e] and pad to like_chunk batch dim by repeating last row."""
    if not d:
        return {}
    out = {}
    bsz_target = next(iter(like_chunk.values())).shape[0]
    for k, v in d.items():
        if not isinstance(v, torch.Tensor):
            out[k] = v
            continue
        vv = v[s:e]
        if pad > 0:
            vv = torch.cat([vv, vv[-1:].repeat([pad] + [1] * (vv.ndim - 1))], dim=0)
        out[k] = vv
    return out


def batched_diffuse_denoise(mod, params, y, t_val, batch_size, device, seed=42):
    """Run mod.diffuse_denoise in chunks; returns (N, 2, V, 3) denoised verts on CPU.
    `y` (guidance dict) is sliced into the same chunks as `params`."""
    N = params["orient"].shape[0]
    out_verts = []
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        chunk = {k: v[s:e] for k, v in params.items()}
        if (e - s) < batch_size:
            pad = batch_size - (e - s)
            chunk = {k: torch.cat([v, v[-1:].repeat([pad] + [1] * (v.ndim - 1))], dim=0)
                     for k, v in chunk.items()}
        else:
            pad = 0
        chunk = {k: v.to(device) for k, v in chunk.items()}
        # Slice y to same range; tolerate non-tensor entries (skip).
        y_chunk = _chunk_dict(y, s, e, pad, chunk)
        y_chunk = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in y_chunk.items()}
        bs = batch_size
        torch.manual_seed(seed + s)
        noise = {k: torch.randn_like(v) for k, v in chunk.items()}
        t = torch.full([bs], t_val, dtype=torch.long, device=device)
        with torch.no_grad():
            res = mod.diffuse_denoise(chunk, y_chunk, t, noise=noise)
        sm0, sm1 = res["denoised_smpls"]
        v = torch.stack([sm0.vertices, sm1.vertices], dim=1).detach().cpu()
        out_verts.append(v[: v.shape[0] - pad] if pad else v)
    return torch.cat(out_verts, dim=0)


# ---------------------------------------------------------------------------
# Perturbation grid
# ---------------------------------------------------------------------------
def build_perturbation_grid():
    grid = [{"direction": "0", "mag_cm": 0, "delta": (0.0, 0.0, 0.0)}]   # control
    for axis_i, axis_name in enumerate("xyz"):
        for sign, sign_name in [(1, "+"), (-1, "-")]:
            for mag_cm in [1, 2, 5, 10, 20, 50]:
                d = [0.0, 0.0, 0.0]
                d[axis_i] = sign * mag_cm / 100.0
                grid.append({
                    "direction": f"{sign_name}{axis_name}",
                    "mag_cm": mag_cm,
                    "delta": tuple(d),
                })
    return grid  # 1 control + 36 perturbations


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_experiment(mod, cfg, anchor_params_cpu, anchor_verts_cpu, anchor_min_v2v_cpu,
                   grid, model_name, contact_threshold=0.01, device="cuda",
                   conditional=False, n_anchors=None):
    bs = cfg.batch_size
    N = anchor_params_cpu["orient"].shape[0]
    if n_anchors is not None and n_anchors < N:
        N = n_anchors

    anchor_params_dev = {k: v[:N].to(device) for k, v in anchor_params_cpu.items()}
    anchor_verts = anchor_verts_cpu[:N]                          # cpu
    anchor_min_v2v = anchor_min_v2v_cpu[:N]                      # cpu

    rows = []
    for gi, g in enumerate(grid):
        delta = torch.tensor(g["delta"], device=device).view(1, 3)             # (1, 3)
        delta_norm = float(delta.norm().item())
        delta_dir = delta / max(delta_norm, 1e-9)                              # (1, 3)
        perturbed = {k: v.clone() for k, v in anchor_params_dev.items()}
        # canonical frame: transl[0]=0; perturb B's offset from A
        perturbed["transl"][:, 1, :] = perturbed["transl"][:, 1, :] + delta

        # Pre-compute body verts for the *perturbed* anchor (no diffusion).
        # We'll compare denoised vs both anchor (was-clean) and perturbed (input-to-model).
        perturbed_verts = batched_get_smpl(mod, perturbed, bs, device)         # (N,2,V,3) cpu

        # guidance
        if conditional:
            # cond_bev expects per-human split tokens (see
            # train_module.get_guidance_params: `guidance = self.split_humans(guidance)`).
            y = {}
            for pp, v in perturbed.items():
                for ii in range(2):
                    y[f"{pp}_h{ii}"] = v[:, ii].clone()
        else:
            y = {}

        denoised_verts = batched_diffuse_denoise(
            mod, perturbed, y, t_val=10, batch_size=bs, device=device, seed=42 + gi
        )

        denoised_min_v2v = min_v2v_per_pair(denoised_verts)
        a_corr = mean_per_vertex_dist(denoised_verts[:, 0], anchor_verts[:, 0])
        b_corr = mean_per_vertex_dist(denoised_verts[:, 1], anchor_verts[:, 1])
        # "from perturbed" = how much the model changed B from its input
        b_delta_from_perturbed = mean_per_vertex_dist(denoised_verts[:, 1], perturbed_verts[:, 1])
        # pullback: project (perturbed_B_centroid - denoised_B_centroid) onto delta_hat.
        # Positive = denoised B moved back toward the anchor side (away from perturbation direction).
        delta_dir_cpu = delta_dir.cpu()
        perturbed_B_centroid = perturbed_verts[:, 1].mean(dim=1)               # (N, 3)
        denoised_B_centroid = denoised_verts[:, 1].mean(dim=1)                 # (N, 3)
        pullback_m = ((perturbed_B_centroid - denoised_B_centroid) * delta_dir_cpu).sum(dim=-1)
        # For Δ=0, delta_dir is unit-anywhere; pullback meaning is degenerate — store as nan.
        if delta_norm < 1e-9:
            pullback_m = torch.full_like(pullback_m, float("nan"))

        for i in range(N):
            row = {
                "model": model_name,
                "anchor_idx": i,
                "direction": g["direction"],
                "mag_cm": g["mag_cm"],
                "anchor_min_v2v_m": float(anchor_min_v2v[i]),
                "denoised_min_v2v_m": float(denoised_min_v2v[i]),
                "a_correction_m": float(a_corr[i]),
                "b_motion_from_anchor_m": float(b_corr[i]),
                "b_motion_from_perturbed_m": float(b_delta_from_perturbed[i]),
                "pullback_m": float(pullback_m[i]),
            }
            for thr_cm, thr_m in [(0.5, 0.005), (1, 0.01), (2, 0.02), (5, 0.05)]:
                row[f"recovery_{thr_cm}cm"] = bool(denoised_min_v2v[i].item() < thr_m)
            rows.append(row)
        if gi % 6 == 0 or gi == len(grid) - 1:
            rec1 = float((denoised_min_v2v < contact_threshold).float().mean())
            rec5 = float((denoised_min_v2v < 0.05).float().mean())
            pb_med = float(pullback_m[~torch.isnan(pullback_m)].median()) if delta_norm > 0 else float("nan")
            print(f"  {model_name}  {gi+1:>3}/{len(grid)}  dir={g['direction']:>2}  mag={g['mag_cm']:>2}cm  "
                  f"rec@1cm={rec1:.2f}  rec@5cm={rec5:.2f}  pullback_median={pb_med*100:+.2f}cm",
                  file=sys.stderr)
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_curves(df, out_dir):
    out_dir = Path(out_dir)

    # Aggregate by (model, mag_cm) across directions
    df_pert = df[df["mag_cm"] > 0]   # exclude Δ=0 baseline
    df_zero = df[df["mag_cm"] == 0]
    agg = (df_pert.groupby(["model", "mag_cm"])
             .agg(rec_5mm=("recovery_0.5cm", "mean"),
                  rec_1cm=("recovery_1cm", "mean"),
                  rec_2cm=("recovery_2cm", "mean"),
                  rec_5cm=("recovery_5cm", "mean"),
                  a_corr=("a_correction_m", "mean"),
                  b_motion=("b_motion_from_anchor_m", "mean"),
                  b_from_perturbed=("b_motion_from_perturbed_m", "mean"),
                  pullback_cm=("pullback_m", lambda s: float(s.mean() * 100)),
                  n=("recovery_1cm", "size"))
             .reset_index())

    # Recovery at multiple thresholds
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, g in agg.groupby("model"):
        g = g.sort_values("mag_cm")
        for col, marker, label_suffix in [("rec_5mm", "x", "5mm"), ("rec_1cm", "o", "1cm"),
                                          ("rec_2cm", "s", "2cm"), ("rec_5cm", "^", "5cm")]:
            ax.plot(g["mag_cm"], g[col], marker=marker, linestyle="-",
                    label=f"{m}  contact@{label_suffix}")
    # Plot Δ=0 baselines as horizontal lines
    for m in df_zero["model"].unique():
        z = df_zero[df_zero["model"] == m]
        for col, ls, lab in [("recovery_0.5cm", ":", "5mm baseline"),
                             ("recovery_1cm", ":", "1cm baseline"),
                             ("recovery_5cm", ":", "5cm baseline")]:
            base = z[col].mean()
            ax.axhline(base, linestyle=ls, alpha=0.3,
                       label=f"{m} Δ=0 {lab}: {base:.2f}")
    ax.set_xlabel("perturbation magnitude on person B's relative translation (cm)")
    ax.set_ylabel("contact recovery rate")
    ax.set_xscale("log")
    ax.set_title("RQ-7: contact recovery after single-step denoise at t=10")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "restoration_curve.png", dpi=150)
    plt.close(fig)

    # Pullback curve: signed B-pullback toward anchor (cm), vs magnitude
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, g in agg.groupby("model"):
        g = g.sort_values("mag_cm")
        ax.plot(g["mag_cm"], g["pullback_cm"], "o-", label=f"{m}  pullback")
        # Plot the magnitude itself as the upper bound (full restoration)
    mags = sorted(df_pert["mag_cm"].unique())
    ax.plot(mags, mags, "k--", alpha=0.4, label="full pullback (= Δ)")
    ax.axhline(0, color="grey", alpha=0.3, label="no pullback")
    ax.set_xlabel("perturbation magnitude (cm)")
    ax.set_ylabel("signed pullback of B centroid toward anchor (cm)")
    ax.set_xscale("log")
    ax.set_title("RQ-7: BUDDI's restoring force on perturbed B")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "pullback_curve.png", dpi=150)
    plt.close(fig)

    # A correction & B motion (from anchor and from perturbed) vs magnitude
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5))
    for m, g in agg.groupby("model"):
        g = g.sort_values("mag_cm")
        ax_a.plot(g["mag_cm"], g["a_corr"] * 100, "o-", label=m)
        ax_b.plot(g["mag_cm"], g["b_motion"] * 100, "o-", label=f"{m}  B vs anchor")
        ax_b.plot(g["mag_cm"], g["b_from_perturbed"] * 100, "s--", label=f"{m}  B vs perturbed")
    # Plot baseline noise floor (Δ=0)
    for m in df_zero["model"].unique():
        z = df_zero[df_zero["model"] == m]
        ax_a.axhline(z["a_correction_m"].mean() * 100, linestyle=":", alpha=0.4,
                     label=f"{m} Δ=0 noise floor")
        ax_b.axhline(z["b_motion_from_anchor_m"].mean() * 100, linestyle=":", alpha=0.4,
                     label=f"{m} Δ=0 noise floor")
    ax_a.set_xlabel("perturbation magnitude (cm)"); ax_a.set_ylabel("(cm)")
    ax_a.set_title("Person A correction  (unperturbed)  ← noise floor")
    ax_a.set_xscale("log"); ax_a.grid(alpha=0.3); ax_a.legend(fontsize=8)
    ax_b.set_xlabel("perturbation magnitude (cm)")
    ax_b.set_title("Person B motion  (vs anchor: where B landed;  vs perturbed: how much model changed B)")
    ax_b.set_xscale("log"); ax_b.grid(alpha=0.3); ax_b.legend(fontsize=8)
    fig.suptitle("RQ-7: where does the denoising correction land?", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "correction_curve.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chi3d-ref", default="repro/phase_a/outputs/chi3d_distribution.pkl")
    ap.add_argument("--uncond-ckpt", default="essentials/buddi/buddi_unconditional.pt")
    ap.add_argument("--uncond-cfg", default="essentials/buddi/buddi_unconditional.yaml")
    ap.add_argument("--cond-ckpt", default="essentials/buddi/buddi_cond_bev.pt")
    ap.add_argument("--cond-cfg", default="essentials/buddi/buddi_cond_bev.yaml")
    ap.add_argument("--n-anchors", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--out-dir", default="repro/phase_c/outputs/rq7")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-cond", action="store_true",
                    help="Run unconditional only (use if conditional path is tricky)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"=== loading CHI3D anchors from {args.chi3d_ref}", file=sys.stderr)
    chi3d = pickle.load(open(args.chi3d_ref, "rb"))
    aa = {k: v for k, v in chi3d["final"].items() if isinstance(v, torch.Tensor)}
    N = min(args.n_anchors, aa["global_orient"].shape[0])
    print(f"  using {N} anchors out of {aa['global_orient'].shape[0]}", file=sys.stderr)

    grid = build_perturbation_grid()
    print(f"  perturbation grid: {len(grid)} entries (6 dirs × 6 magnitudes)", file=sys.stderr)

    all_rows = []

    # ------------- UNCONDITIONAL -------------
    print(f"\n=== loading buddi_unconditional ===", file=sys.stderr)
    mod_u, cfg_u, _ = load_diffusion_module(
        args.uncond_ckpt, args.uncond_cfg, args.batch_size, args.device
    )
    print(f"  batch_size={cfg_u.batch_size}", file=sys.stderr)
    # Step 1: build AA-format anchor dict with the keys cast_smpl expects.
    aa_renamed = rename_to_module_keys({k: v[:N] for k, v in aa.items()})
    # Step 2: cast_smpl → canonical params (orient/pose rot6d, transl[0]=0, transl[1]=offset)
    anchor_params_canonical = batched_cast_smpl(mod_u, aa_renamed, cfg_u.batch_size, args.device)
    # Sanity: confirm transl[0] is zeroed
    assert anchor_params_canonical["transl"][:, 0, :].abs().max() < 1e-6, \
        f"relative_transl didn't zero transl[0]; max={anchor_params_canonical['transl'][:,0,:].abs().max()}"
    print(f"  canonical-frame transl[0] zeroed (sanity ok)", file=sys.stderr)
    # Step 3: get anchor verts in canonical frame
    anchor_verts = batched_get_smpl(mod_u, anchor_params_canonical, cfg_u.batch_size, args.device)
    anchor_min_v2v = min_v2v_per_pair(anchor_verts)
    print(f"  anchor min_v2v median (canonical frame): "
          f"{float(anchor_min_v2v.median()) * 1000:.2f} mm", file=sys.stderr)

    print(f"\n=== running unconditional grid ===", file=sys.stderr)
    rows_u = run_experiment(
        mod_u, cfg_u,
        anchor_params_cpu=anchor_params_canonical,
        anchor_verts_cpu=anchor_verts,
        anchor_min_v2v_cpu=anchor_min_v2v,
        grid=grid, model_name="uncond",
        device=args.device, conditional=False, n_anchors=N,
    )
    all_rows.extend(rows_u)
    del mod_u
    torch.cuda.empty_cache()

    # ------------- CONDITIONAL -------------
    if not args.skip_cond:
        print(f"\n=== loading buddi_cond_bev ===", file=sys.stderr)
        try:
            mod_c, cfg_c, _ = load_diffusion_module(
                args.cond_ckpt, args.cond_cfg, args.batch_size, args.device
            )
            # rebuild anchor params via cast_smpl on cond model
            aa_renamed_c = rename_to_module_keys({k: v[:N] for k, v in aa.items()})
            anchor_params_c = batched_cast_smpl(mod_c, aa_renamed_c, cfg_c.batch_size, args.device)
            print(f"\n=== running conditional grid ===", file=sys.stderr)
            rows_c = run_experiment(
                mod_c, cfg_c,
                anchor_params_cpu=anchor_params_c,
                anchor_verts_cpu=anchor_verts,
                anchor_min_v2v_cpu=anchor_min_v2v,
                grid=grid, model_name="cond_bev",
                device=args.device, conditional=True, n_anchors=N,
            )
            all_rows.extend(rows_c)
        except Exception as e:
            print(f"  conditional path failed: {type(e).__name__}: {e}", file=sys.stderr)
            print(f"  continuing with unconditional only", file=sys.stderr)

    # ------------- save + plot -------------
    df = pd.DataFrame(all_rows)
    df.to_csv(Path(args.out_dir) / "restoration_table.csv", index=False)
    print(f"\nsaved restoration_table.csv  ({len(df)} rows)", file=sys.stderr)

    summary = (df.groupby(["model", "mag_cm"])
                 .agg(rec_5mm=("recovery_0.5cm", "mean"),
                      rec_1cm=("recovery_1cm", "mean"),
                      rec_5cm=("recovery_5cm", "mean"),
                      a_corr_cm=("a_correction_m", lambda s: float(s.mean() * 100)),
                      b_anchor_cm=("b_motion_from_anchor_m", lambda s: float(s.mean() * 100)),
                      b_perturbed_cm=("b_motion_from_perturbed_m", lambda s: float(s.mean() * 100)),
                      pullback_cm=("pullback_m", lambda s: float(s.mean() * 100) if not s.isna().all() else float("nan")),
                      n=("recovery_1cm", "size"))
                 .reset_index())
    summary_path = Path(args.out_dir) / "summary.json"
    summary.to_json(summary_path, orient="records", indent=2)
    print(f"saved summary.json", file=sys.stderr)
    print("\n=== SUMMARY (recovery rate at each magnitude) ===", file=sys.stderr)
    print(summary.to_string(index=False))

    plot_curves(df, args.out_dir)
    print(f"\nsaved restoration_curve.png + correction_curve.png", file=sys.stderr)


if __name__ == "__main__":
    main()
