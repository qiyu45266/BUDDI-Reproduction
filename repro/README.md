# BUDDI Reproduction — Team Workspace

This directory contains everything our team adds on top of the upstream
[`muelea/buddi`](https://github.com/muelea/buddi) repository for our USI AdvTML
reproduction project (paper: Müller et al., "Generative Proxemics", CVPR 2024).

## How the work is structured

We split the reproduction into three **phases**, not three siloed person-deliverables.
Each phase is one team member's primary responsibility but everyone reviews / runs
the others' code; the env, patches, and conventions in `shared/` are common to all
three.

| Phase | Scope | Status |
|---|---|---|
| **A — Generation analysis** | Unconditional sampling + DDIM-step Pareto study + FID vs real CHI3D ground-truth distribution. Maps to paper §5.1 (C1, C2, **C3 partial**). | ✅ Complete (see `phase_a/REPRODUCIBILITY_REPORT.md`) |
| **B — Optimization-based fitting** | Optimization with BUDDI as prior on FlickrCI3D / CHI3D / Hi4D test sets. Maps to paper §5.2 and Tables 1, 3. | 🟡 In progress (see `phase_b/README.md`) |
| **C — Mechanistic probes** | Beyond-paper RQs on what BUDDI actually learned and what the prior actually does in §3.2 Eq.6. RQ-5 (contact vs proximity) + RQ-7 (counterfactual restoring force). | ✅ Complete (see `phase_c/REPRODUCIBILITY_REPORT.md`) |

Phases B and C **build on Phase A's env and patches** — no re-installation needed.

## Layout

```
repro/
├── README.md                          ← this file
├── HANDOFF.md                         ← team decisions log (read this for context)
├── shared/                            ← common to all phases
│   ├── env/
│   │   ├── 0_install_miniconda.sh     ← bootstrap conda
│   │   ├── 1_setup_env.sh             ← build `hhcenv39` (Ada GPU compatible)
│   │   ├── 2_fetch_essentials.sh      ← author ckpt + aux (~60 MB)
│   │   ├── 3_fetch_bodymodels.sh      ← SMPL-X / SMPL / SMIL (~900 MB, needs creds)
│   │   └── 4_extract_datasets.sh      ← OPTIONAL: extract raw CHI3D/FlickrCI3D tars (for Phase A v2 / B)
│   ├── patches/
│   │   └── sample_add_seed.patch      ← --seed flag for llib/.../sample.py
│   └── docs/
│       └── UPSTREAM_NOTES.md          ← which upstream scripts we replace and why
├── phase_a/                           ← generation analysis
│   ├── REPRODUCE.md                   ← zero-state how-to
│   ├── REPRODUCIBILITY_REPORT.md      ← claim-by-claim audit vs paper §5.1
│   ├── code/                          ← analysis scripts (analyze, make_gallery, chi3d_distribution, eval_fid_chi3d, plot_pareto_v2, sbatch)
│   ├── runners/                       ← thin wrappers: smoke_imports, smoke_sample, run_uncond, run_analyze, run_chi3d_fid
│   └── outputs/                       ← deliverables (small artifacts tracked; .pkl/large .png gitignored)
├── phase_b/                           ← optimization
│   └── README.md                      ← starting points + known blockers
└── phase_c/                           ← robustness RQ
    └── README.md                      ← starting points + design sketch
```

## Quickstart for any phase

```bash
# Once, for everyone:
bash repro/shared/env/0_install_miniconda.sh
bash repro/shared/env/1_setup_env.sh                # ~6 min
bash repro/shared/env/2_fetch_essentials.sh         # ~2 min
SMPLX_USER=... SMPLX_PASS=... SMPL_USER=... SMPL_PASS=... \
  bash repro/shared/env/3_fetch_bodymodels.sh       # ~90 sec, needs accounts
```

Then go to your phase directory and follow its README.

## Coordination

- New upstream modifications → add as a patch under `shared/patches/`, never edit
  `llib/` in-place.
- New env/data deps → extend `shared/env/1_setup_env.sh` (idempotent), not a side script.
- Anything that affects more than one phase → document it in `HANDOFF.md`.
- Outputs from one phase are valid sanity references for the others
  (e.g. Phase A's gallery / `x_starts_smplx.pkl` files for Phase B sanity checks).

See `phase_a/REPRODUCIBILITY_REPORT.md` for the reproml.org-aligned audit of what
the paper claims, what we reproduced, and the (honest) deltas.
