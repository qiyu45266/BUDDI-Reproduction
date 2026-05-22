# Phase A — analysis code

Python + sbatch sources for the Phase A pipeline. Run from the **repo root**
via the wrappers under `repro/phase_a/runners/`, not directly — they handle
env activation and path resolution.

## Files

| File | Purpose |
|---|---|
| `analyze.py` | Reads `x_starts_smplx.pkl` from each schedule's run dir, computes diversity / interpenetration / self-FID, writes `results.csv`, `pareto.png`, `diversity.png` |
| `make_gallery.py` | Stitches 360°-render `*_gen.gif` files into an N×M PNG grid |
| `chi3d_distribution.py` | Walks `datasets/original/CHI3D/train/{s02,s03,s04}/`, reads each action's contact-frame SMPL-X params, applies the same coordinate transform `llib/data/preprocess/chi3d.py` uses, computes SMPL-X vertices, emits a BUDDI-format dict → `chi3d_distribution.pkl` (94 MB, gitignored). |
| `eval_fid_chi3d.py` | Loads BUDDI samples + CHI3D reference, computes FID with paper's `fid_on_params` machinery (`llib.methods.hhc_diffusion.evaluation.eval`). Emits `fid_vs_chi3d.csv` + merges a `fid_vs_chi3d` column into `results.csv`. Also runs a bootstrap noise floor (20 random half-splits). |
| `plot_pareto_v2.py` | Renders the 2-axis `pareto_v2.png` showing self-FID and FID-vs-CHI3D together with the noise floor. |
| `sample_schedules.sbatch` | Cluster-side launcher for the same 4-schedule sweep that `runners/run_uncond.sh` runs locally. Kept for reference / cluster re-use. |
| `sample_cond_on_demo.sbatch` | Cluster-side launcher for cond sampling on bundled demo BEVs. **Not used** in the current pipeline — cond sampling is out of Phase A's pivoted scope (see `repro/phase_a/REPRODUCIBILITY_REPORT.md §3.3`). |

## Why the cluster sbatch files are still here

They are the original cluster-execution form of the same logic. The local
runners under `repro/phase_a/runners/` are the canonical entry points; the
sbatch files exist so that a future teammate with cluster access can `sbatch
sample_schedules.sbatch` and get the same outputs without re-deriving the
flags. Both forms pass identical `python ... sample.py` invocations.

## Coordination with shared infra

- These scripts assume the env built by `repro/shared/env/1_setup_env.sh` is
  active and that `repro/shared/patches/sample_add_seed.patch` has been
  applied (so `--seed` is recognised).
- They read inputs from / write outputs to `repro/phase_a/outputs/`; the
  runners pass these paths in explicitly so the code itself is path-agnostic.
