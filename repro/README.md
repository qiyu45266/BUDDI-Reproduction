# BUDDI Reproduction — Team Workspace

This directory contains everything our team adds on top of the upstream
[`muelea/buddi`](https://github.com/muelea/buddi) repository for our USI AdvTML
reproduction project (paper: Müller et al., "Generative Proxemics", CVPR 2024).

**Everything outside `repro/` is upstream and untouched** — diff against
`origin/main` should be empty on those paths. Our modifications to upstream code
(currently: a `--seed` flag in `llib/.../sample.py`) are stored as patches in
`shared/patches/` and applied during environment setup.

## How the work is structured

We split the reproduction into three **phases**, not three siloed person-deliverables.
Each phase is one team member's primary responsibility but everyone reviews / runs
the others' code; the env, patches, and conventions in `shared/` are common to all
three.

| Phase | Scope | Status |
|---|---|---|
| **A — Generation analysis** | Unconditional sampling from the pretrained ckpt + DDIM-step Pareto study. Maps to paper §5.1. | ✅ Complete (see `phase_a/REPRODUCIBILITY_REPORT.md`) |
| **B — Optimization-based fitting** | Optimization with BUDDI as prior on FlickrCI3D / CHI3D / Hi4D test sets. Maps to paper §5.2 and Tables 1, 3. | 🟡 In progress (see `phase_b/README.md`) |
| **C — Robustness extension** | Beyond-paper RQ: how does the cond model behave under perturbed BEV inputs? | 🟡 In progress (see `phase_c/README.md`) |

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
│   │   └── 3_fetch_bodymodels.sh      ← SMPL-X / SMPL / SMIL (~900 MB, needs creds)
│   ├── patches/
│   │   └── sample_add_seed.patch      ← --seed flag for llib/.../sample.py
│   └── docs/
│       └── UPSTREAM_NOTES.md          ← which upstream scripts we replace and why
├── phase_a/                           ← generation analysis
│   ├── REPRODUCE.md                   ← zero-state how-to
│   ├── REPRODUCIBILITY_REPORT.md      ← claim-by-claim audit vs paper §5.1
│   ├── code/                          ← analysis scripts (analyze, make_gallery, sbatch)
│   ├── runners/                       ← thin wrappers around sample.py / analyze.py
│   └── outputs/                       ← deliverables (small artifacts tracked; .pkl gitignored)
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
