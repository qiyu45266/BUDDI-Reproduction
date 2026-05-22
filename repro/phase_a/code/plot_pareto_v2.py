#!/usr/bin/env python
"""
Render a 2-panel Pareto figure showing both:
  - self-FID (left axis): BUDDI samples vs slowest-schedule reference
  - FID vs CHI3D (right axis): BUDDI samples vs raw CHI3D MoCap distribution

Both should agree on the cross-schedule trend (sweet spot ~40 steps); having
two independent measurements strengthens the headline finding.
"""
import argparse, os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="repro/phase_a/outputs/uncond/results.csv")
    ap.add_argument("--out", default="repro/phase_a/outputs/uncond/pareto_v2.png")
    ap.add_argument("--noise-floor", type=float, default=None,
                    help="If given, drawn as horizontal dashed line on the CHI3D FID axis")
    args = ap.parse_args()

    df = pd.read_csv(args.results).sort_values("n_steps")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "tab:blue"
    ax1.set_xlabel("DDIM steps (log scale)")
    ax1.set_xscale("log")
    ax1.set_ylabel("self-FID (slowest schedule = reference)", color=color1)
    ax1.plot(df["n_steps"], df["self_fid"], "o-", color=color1, label="self-FID")
    ax1.tick_params(axis="y", labelcolor=color1)

    if "fid_vs_chi3d" in df.columns:
        ax2 = ax1.twinx()
        color2 = "tab:red"
        ax2.set_ylabel("FID vs CHI3D contact-frame distribution", color=color2)
        ax2.plot(df["n_steps"], df["fid_vs_chi3d"], "s--", color=color2, label="FID vs CHI3D")
        ax2.tick_params(axis="y", labelcolor=color2)
        if args.noise_floor is not None:
            ax2.axhline(args.noise_floor, color=color2, alpha=0.25, linestyle=":",
                        label=f"CHI3D noise floor ({args.noise_floor:.2f})")

    plt.title("BUDDI Sampling Pareto — self-FID vs real-GT FID")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
