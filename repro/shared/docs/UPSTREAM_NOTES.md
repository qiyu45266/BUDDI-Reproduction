# Upstream notes

Catalogue of what we change vs upstream `muelea/buddi`, why, and how we keep the
upstream tree byte-identical so anyone can `git diff origin/main -- <path>` to
verify.

## Hard rule

**Nothing outside `repro/` is edited or deleted.** All our additions live under
`repro/`. The only exception is during env setup, when `1_setup_env.sh` calls
`git apply` to patch a small handful of upstream files; the patch text itself
is committed in `repro/shared/patches/`, so a reviewer can read it without running
anything.

## Patches we apply (idempotent)

| Patch file | Target | Why |
|---|---|---|
| `sample_add_seed.patch` | `llib/methods/hhc_diffusion/evaluation/sample.py` | Upstream samples non-deterministically. We add a `--seed` flag that seeds `torch + numpy + random + cudnn.deterministic` so `results.csv` numbers are reproducible across runs on the same GPU architecture. |

`1_setup_env.sh` step 13 applies all patches in `repro/shared/patches/`. The
apply step is safe to re-run: it checks `git apply --check` first, skips if the
patch is already applied (i.e. it applies cleanly in reverse), and warns if it
neither applies forward nor is already applied (indicating someone edited the
upstream file directly).

## Upstream scripts we **do not use** (replaced under `repro/shared/env/`)

| Upstream | Replacement | Reason |
|---|---|---|
| `install_conda_env.sh` | `repro/shared/env/1_setup_env.sh` | Upstream pins `pytorch=1.9.1` + `cudatoolkit=11`, which has no kernels for compute capability ≥ sm_89 (Ada / RTX 4070, also affects sm_90 / H100). Our replacement uses `pytorch 2.0.1 + cu118` (native sm_89), pins `numpy<1.24` to avoid the PyTorch 2.0 / NumPy 2 ABI break, installs `chumpy` from a GitHub fork that does not use the removed `numpy.int / numpy.bool` aliases, and pulls `libGLU` via `conda-forge` so `pyrender` works headless without `sudo apt`. |
| `fetch_data.sh` | `repro/shared/env/2_fetch_essentials.sh` | Upstream uses system `unzip`, which is not installed on default WSL2 Ubuntu. Our replacement uses Python's `zipfile`. |
| `fetch_bodymodels.sh` | `repro/shared/env/3_fetch_bodymodels.sh` | Upstream prompts interactively for credentials. Our replacement takes them via env vars (`SMPLX_USER` / `SMPLX_PASS` / `SMPL_USER` / `SMPL_PASS`), URL-encodes them correctly, validates that downloads are non-empty (catches the silent failure when credentials are wrong but the server returns an HTML error page that gets "unzipped" without complaint), and uses the same Python-`zipfile` shim. |
| (no upstream equivalent) | `repro/shared/env/4_extract_datasets.sh` | Extracts the 4 raw `ci3d.imar.ro` tars into `datasets/original/{FlickrCI3D_Signatures,CHI3D}/` exactly per upstream DATA.md layout. Only needed for Phase A v2 (FID vs CHI3D real GT) and Phase B. |

## Upstream cluster scaffold (`slurm/*.sh`, `slurm/train.sbatch`) — left **deleted**

Upstream ships a Slurm-based training pipeline:
`bootstrap.sh`, `controller.sh`, `lib_common.sh`, `smoke_test.sh`, `status.sh`,
`submit_pool.sh`, `teardown.sh`, `train.sbatch`, plus a `README.md`.

We **do not restore these** in our team workspace because:

1. The full-training path requires `processed_pseudogt_fits.pkl` for FlickrCI3D,
   whose official Google Drive link is dead (upstream issue
   [#13](https://github.com/muelea/buddi/issues/13)) and which the upstream
   maintainer has not responded to since 2024-02. See `HANDOFF.md §3.2`.
2. Our team pivoted to a pretrained-checkpoint-only reproduction on local GPU
   (RTX 4070 Laptop, no cluster). The Slurm scripts have no role in that path.
3. Keeping them would invite confusion ("which env do I use, `bootstrap.sh` or
   `1_setup_env.sh`?"). Both cannot be right; one had to win.

A reviewer who wants the Slurm scaffold can restore it with:
```bash
git checkout origin/main -- slurm/
```

This is the **only** intentional deletion of an upstream tracked file.

## Upstream files we left **untouched**

Everything not listed above. To verify:
```bash
git diff origin/main -- $(git ls-tree --name-only origin/main | grep -v -E '^(slurm)$')
```
should be empty.
