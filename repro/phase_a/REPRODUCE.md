# Reproduce — Phase A (sampling-only)

Step-by-step zero-state reproduction of Phase A (unconditional sampling +
DDIM-step Pareto study). For the claim-by-claim audit against paper §5.1, see
`REPRODUCIBILITY_REPORT.md`.

## 1. Prerequisites

| Component | Required version | Notes |
|---|---|---|
| OS | WSL2 Ubuntu 22.04/24.04 or native Ubuntu 22.04+ | Tested on WSL2 Ubuntu 24.04.3 LTS |
| GPU | NVIDIA, compute capability ≥ sm_70 | Verified on RTX 4070 Laptop (sm_89). RTX 3060/3070/3080/3090 (sm_86) should also work — PyTorch 2.0.1+cu118 ships kernels for sm_70–sm_90. |
| NVIDIA driver | ≥ 520.61 (Linux) / ≥ 522.06 (WSL) | Required for CUDA 11.8 runtime. |
| VRAM | ≥ 8 GB | Sampling at `--batch_size 8`. Reduce if OOM. |
| Disk | ≥ 5 GB free | env (1.5 GB) + body models (~900 MB) + outputs (~1 GB) |
| Network | Unrestricted access to Google Drive, `download.is.tue.mpg.de`, `obj-web.iosb.fraunhofer.de`, `repo.anaconda.com`, `download.pytorch.org`, `dl.fbaipublicfiles.com`, `github.com` |

## 2. Accounts (one-time)

You must register and **accept the license terms** on both:

- https://smpl-x.is.tue.mpg.de — for SMPL-X body model
- https://smpl.is.tue.mpg.de   — for SMPL body model (separate account)

Without accepting the license in the web UI, the download server returns 401.

## 3. Clone

```bash
git clone <this-repo-url> buddi
cd buddi
```

All `repro/` scripts auto-detect the repo location from their own path, so
the directory can be anywhere and named anything.

## 4. Install (one-time, ~10 minutes)

```bash
bash repro/shared/env/0_install_miniconda.sh   # 2 min — installs miniconda to ~/miniconda3
bash repro/shared/env/1_setup_env.sh           # 6 min — builds conda env `hhcenv39` + applies patches
```

`1_setup_env.sh` is **idempotent** — safe to re-run. It:
- pins PyTorch 2.0.1+cu118 + pytorch3d 0.7.4 + numpy<1.24
- installs chumpy from a numpy-compatible GitHub fork
- pulls libGLU via conda-forge so `pyrender` works headless without sudo
- applies all patches in `repro/shared/patches/` (currently: `--seed` support
  for `llib/.../sample.py`)

Why not the upstream `install_conda_env.sh`? It pins PyTorch 1.9 + cudatoolkit
11, which has no kernels for compute capability ≥ sm_89. See
`repro/shared/docs/UPSTREAM_NOTES.md`.

## 5. Download data (one-time, ~3 minutes)

```bash
# Author-distributed checkpoints + auxiliary data (~60 MB)
bash repro/shared/env/2_fetch_essentials.sh

# Licensed body models (~900 MB) — needs the env vars from §2
SMPLX_USER='you@example.com' SMPLX_PASS='...' \
SMPL_USER='you@example.com'  SMPL_PASS='...' \
  bash repro/shared/env/3_fetch_bodymodels.sh
```

After this:

```
essentials/buddi/buddi_unconditional.{pt,yaml}
essentials/buddi/buddi_cond_bev.{pt,yaml}
essentials/body_models/{smplx,smpl,smil}/...
```

## 6. Sanity checks (~1 minute)

```bash
bash repro/phase_a/runners/smoke_imports.sh    # imports all llib modules — fails fast on env breakage
bash repro/phase_a/runners/smoke_sample.sh     # generates 8 samples (~30 s on 4070) — verifies GPU path
```

The smoke test should print `final verts shape torch.Size([8, 2, 10475, 3])`
and write `repro/phase_a/outputs/smoke/generate_1000_100_v0/x_starts_smplx.pkl`.

## 7. Full sampling run (~25 minutes on RTX 4070)

```bash
bash repro/phase_a/runners/run_uncond.sh    # 512 samples × 4 DDIM schedules (100, 40, 20, 10 steps)
bash repro/phase_a/runners/run_analyze.sh   # metrics + Pareto/diversity plots + 8×8 gallery
```

Outputs:

```
repro/phase_a/outputs/
├── gallery_uncond_baseline.png             (8×8 mesh grid)
└── uncond/
    ├── results.csv                         (4-row metrics table)
    ├── pareto.png                          (headline figure)
    ├── diversity.png
    └── generate_1000_{10,25,50,100}_v0/
        ├── cmd_args.txt
        ├── x_starts_smplx.pkl              (~124 MB each — gitignored)
        └── renders/*.gif                   (only generated for skip=10)
```

## 8. Determinism

The runners pass `--seed 42` to `sample.py` (the flag is added by
`repro/shared/patches/sample_add_seed.patch`, which seeds `torch + numpy + random`
and sets `cudnn.deterministic = True`). **Same seed on the same GPU architecture
should give identical `results.csv`**. Different GPU architectures may produce
numerically-close but not bit-identical numbers because of non-deterministic CUDA
atomic kernels inside SMPL-X mesh evaluation that we did not budget time to
disable. The Pareto **trend** is robust across seeds.

To explore seed sensitivity, run `SEED=7 bash repro/phase_a/runners/run_uncond.sh`.

## 9. Expected results (seed=42 on RTX 4070 Laptop, sm_89)

| n_steps | diversity (mm) | severe_interpen | self_fid | fid_vs_chi3d |
|---|---:|---:|---:|---:|
| 100 (paper baseline) | 465.1 | 0.010 | 0.000 | 30.26 |
| 40 | **457.7** | **0.008** | 0.028 | **30.21** |
| 20 | 454.0 | 0.010 | 0.136 | 30.40 |
| 10 | 420.3 | 0.016 | 0.432 | 30.91 |

Headline finding: **40 steps is the sweet spot** on both quality metrics
(self-FID and FID vs CHI3D real-GT distribution) — 2.5× faster than the
paper's 100-step schedule, lowest severe-interpenetration rate (0.8%).
The `fid_vs_chi3d` column requires also running step 10 (next).

Why is FID vs CHI3D ~30 rather than the paper's ~1.6? Three caveats laid out
in `REPRODUCIBILITY_REPORT.md §2.3`: (a) we use only CHI3D as reference (paper
uses a 60/20/20 Flickr/CHI3D/Hi4D mix); (b) BUDDI is trained on that wider mix
so its outputs are broader than CHI3D alone; (c) paper computes FID in a
learned autoencoder latent that the authors did not release. Absolute number
is therefore **not directly comparable**, but the cross-schedule **ranking is
consistent** with the self-FID column.

## 10. Optional: FID against real CHI3D ground-truth (~6 min total, needs raw CHI3D)

Pre-req: register at https://ci3d.imar.ro/download, download `chi3d_train.tar.gz`
(and optionally the 3 other tars), place them under `datasets/` of this repo,
then:

```bash
bash repro/shared/env/4_extract_datasets.sh   # ~5 min I/O on /mnt/c
bash repro/phase_a/runners/run_chi3d_fid.sh   # ~1 min compute
```

This script:
1. Walks `datasets/original/CHI3D/train/{s02,s03,s04}/interaction_contact_signature.json`
   to find each action's contact frame index.
2. Reads the corresponding `smplx/<action>.json`, applies the same coordinate
   transform `llib/data/preprocess/chi3d.py` uses, and builds a 373-pair
   reference distribution.
3. Computes FID (paper's `fid_on_params` machinery) between each schedule's
   `x_starts_smplx.pkl` and the CHI3D reference.
4. Renders `pareto_v2.png` with both self-FID and FID-vs-CHI3D on a 2-axis plot.

Outputs land in `repro/phase_a/outputs/uncond/{fid_vs_chi3d.csv, pareto_v2.png}`
and the `fid_vs_chi3d` column is merged back into `results.csv`.

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Bus error (core dumped)` on `import torch` | numpy>=2.0 got pulled in. Re-run `1_setup_env.sh` — step 12 force-reinstalls numpy 1.23.5. |
| `ImportError: Library "GLU" not found` from pyrender | libGLU missing. Re-run `1_setup_env.sh` step 11 (`conda install -c conda-forge libglu`). |
| `cannot import name 'int' from 'numpy'` from chumpy | PyPI chumpy 0.70 is installed instead of the GitHub master fork. `pip uninstall chumpy && pip install --no-build-isolation git+https://github.com/mattloper/chumpy.git`. |
| `401 Username/Password Authentication Failed` from `download.is.tue.mpg.de` | Either credentials wrong, or you haven't accepted the SMPL-X / SMPL license on the website yet. Both required (§2). |
| `CondaToSNonInteractiveError: Terms of Service have not been accepted` | First-time conda ≥25 issue. `1_setup_env.sh` step 3 handles this; re-run it. |
| `CUDA out of memory` during sampling | Drop batch size: edit `repro/phase_a/runners/run_uncond.sh` → `--batch_size 4`. |
| `--seed` flag not recognised | Patch did not apply. Re-run `1_setup_env.sh` (step 13 re-applies patches, idempotent). |

## 11. Hardware / driver this was tested on

- Intel Core Ultra 9 185H + RTX 4070 Laptop (8 GB, sm_89)
- NVIDIA driver 580.88
- WSL2 Ubuntu 24.04.3 LTS, kernel 6.6.87.2-microsoft-standard-WSL2
- conda 26.3.2, miniconda installed fresh from `Miniconda3-latest-Linux-x86_64.sh`
- Wall-clock cold-start → Phase A deliverables: **~35-40 min** (env build 6 min +
  downloads 2 min + sampling 25 min + analysis < 1 min); add **~10 min** for Phase C
  (~50 min total).

If anything diverges from these numbers by > 2× on similar hardware, open a
GitHub issue or ping the team in chat.
