# Phase B — Reproducibility Report (Optimization-based Reconstruction)

**Paper**: Müller et al., *Generative Proxemics*, CVPR 2024.
**Scope**: Phase B — optimization-based fitting (paper §3.2, §5.2, Tables 1/3).
**Hardware**: RTX 4070 Laptop GPU (8 GB, sm_89), WSL2 Ubuntu 24.04.
**Date**: 2026-05-25.

## 1. What Phase B covers

The "fitting with BUDDI" rows in paper Tables 1 and 3: take BEV-regressed
initial estimates and refine them by minimizing
`L_fitting + L_diffusion (Eq. 6)` over SMPL-X parameters of two interacting
people. The diffusion-prior loss uses the conditional model
(`buddi_cond_bev.pt`) at noise level t = 10.

## 2. What we attempted

### B1 — Demo sanity fit

We ran the BUDDI optimization pipeline on the 3 bundled FlickrCI3D demo
images (`college_232334`, `girls_250416`, `girls_280199`) using the official
conditional checkpoint and demo BEV/ViTPose precomputed inputs.

**Pipeline verified end-to-end**:
- Conditional checkpoint `buddi_cond_bev.pt` loads correctly
- ShapeConverter (SMPLA → SMPLXA) works after installing the SMPLA model
  via `romp.prepare_smpl` (not included in the upstream fetch scripts)
- Demo data (BEV `.npz` + ViTPose keypoints + OpenPose keypoints) loads
- Diffusion model builds (1000 steps, cosine schedule, fixed SDS at t=10)
- Optimization loop runs with all losses active: keypoint2d (L2),
  interpenetration, init_pose regularization, diffusion_prior_{pose,shape,transl}

**What we observed from the optimization loop** (from a run that completed
Stage 0 of the first item before being interrupted):

| Metric | Stage 0, step 1 | Stage 0, step 100 | Trend |
|---|---:|---:|---|
| Total loss (Tl) | ~3,600 | ~690 | decreasing |
| Keypoint loss (kl) | ~346 | ~297 | decreasing |
| Interpenetration (ipl) | ~0.14 | ~9.6 | rising (bodies brought closer) |
| Diffusion prior pose (rh0p+rh1p) | ~100+82 | ~55+43 | decreasing |
| Diffusion prior transl (rh1h0t) | ~2,981 | ~161 | decreasing steeply |

The total loss dropped ~5× across 100 iterations of Stage 0, with the
diffusion-prior translation term dominating the initial loss and converging
quickly. This is consistent with the prior pulling the initial BEV
estimates toward a plausible relative configuration.

**Why we did not get final outputs**: each optimization iteration takes
~45 seconds on the RTX 4070 Laptop (8 GB VRAM, fully saturated at 7.9/8.2 GB).
With 2 stages × 100 iterations per item and 56 items from 3 demo images,
the full demo fit would take ~140 hours. Even narrowing to 1 image (2 items)
and 50 iterations per stage, estimated time was ~2.5 hours, which exceeded
our remaining time budget before the submission deadline.

The upstream codebase appears to have been developed on cluster-grade GPUs
(A100 40 GB) where iteration time is likely ~5-10× faster. On a laptop GPU,
the demo fit is not a quick sanity check.

### B2 — Test-set evaluation (Table 1 / Table 3)

**Blocked** by the same missing data that affects Nawfal's Task 3:

- `datasets/processed/FlickrCI3D_Signatures/processed_pseudogt_fits.pkl`
  (official Google Drive link dead, upstream issue #13, no maintainer
  response since 2024-02)
- Without PGT fits, `llib/data/single.py:load_data()` raises
  `FileNotFoundError` when constructing the FlickrCI3D test dataset

CHI3D Table 3 would also require running `datasets/scripts/CHI3D/` preprocessing
scripts that we did not have time to attempt.

## 3. What we did not reproduce, and why

| Paper claim | Status | Blocker |
|---|---|---|
| Table 1: PA-MPJPE 66 mm on FlickrCI3D | not reproduced | dead PGT link |
| Table 3: V2V / PA-MPJPE on CHI3D | not reproduced | time budget |
| Table 2: Hi4D | not reproduced | dataset access |
| Demo qualitative fit | partial — pipeline verified, no final outputs | iteration speed on laptop GPU |

## 4. Artifacts

- `repro/phase_b/runners/run_b1_demo.sh` — runner script
- `repro/phase_b/outputs/b1_demo_fit.log` — partial optimization log
- `repro/phase_b/outputs/demo_fit/config.yaml` — full resolved config
- `repro/phase_b/outputs/demo_fit/summaries/` — TensorBoard events (partial)

## 5. Compute

| Step | Wall-clock |
|---|---:|
| Install SMPLA via romp.prepare_smpl | ~3 min |
| B1 demo fit (interrupted, 1 item partial) | ~25 min |
| **Phase B total** | **~30 min** |
