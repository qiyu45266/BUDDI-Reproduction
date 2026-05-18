# Reproducibility Report — Phase A

**Paper**: Müller et al., *Generative Proxemics: A Prior for 3D Social
Interaction from Images*, CVPR 2024.
**Scope**: Phase A — generation side (paper §5.1). Pretrained-checkpoint-only
pivot per `repro/HANDOFF.md`.
**Hardware**: RTX 4070 Laptop GPU (8 GB, sm_89), Intel Core Ultra 9 185H,
WSL2 Ubuntu 24.04.
**Date**: 2026-05-18.
**Principle**: reproml.org — reproduce what is supported by code + data; be
explicit about what is not.

## 1. Paper claims relevant to Phase A

The paper's generation experiments are in §5.1. Each claim audited:

| # | Paper claim | Section | Reproducible? | Status |
|---|---|---|---|---|
| C1 | Unconditional sampling produces realistic 2-person interaction (qualitative) | §5.1, Fig. 1, 3 | Yes | ✅ Done |
| C2 | Sampling schedule: DDIM, max-t=1000, skip-steps=10 (= 100 steps) | §4 "Implementation Details" | Yes | ✅ Done as baseline |
| C3 | FID 1.6 (BUDDI) vs 3.3 (VAE) on 8K samples against training data | §5.1 last paragraph | **No** | ❌ Needs training PGT + VAE baseline ckpt |
| C4 | Perceptual study: BUDDI beats real data 44.4%, beats VAE 60.17%, beats random 71.23% | §5.1 | **No** | ❌ Needs AMT workers |
| C5 | Cond model used as optimization initializer / BUDDI (gen.) baseline | §3.2, §5.2, Tables 1-3 | **No (standalone)** | ❌ Blocked — see §3.2; Phase B scope |

## 2. What we reproduced

### 2.1 Unconditional sampling at the paper's baseline schedule

Loaded `essentials/buddi/buddi_unconditional.pt` (official ckpt) and sampled
**512 SMPL-X two-person poses** at `max-t=1000, skip-steps=10` — the schedule
the paper specifies in §4 "Implementation Details". Each sample is rendered as
a 360° gif (30 frames, 200×256 px).

- Output: `repro/phase_a/outputs/uncond/generate_1000_10_v0/`
  - `x_starts_smplx.pkl` (124 MB, gitignored) — full SMPL-X parameters per sample
  - `renders/*.gif` (512 files, 39 MB total, gitignored)
- 8×8 gallery: `repro/phase_a/outputs/gallery_uncond_baseline.png` ← qualitative
  analogue of paper Fig. 3

Visual inspection confirms BUDDI generates couples in hugging, sitting-side-by-side,
embracing, high-fives, and other interactions consistent with the paper's
qualitative description.

### 2.2 Novel addition — sampling-efficiency Pareto study

The paper fixes its sampling schedule at 100 DDIM steps and does not study
faster sampling. We ran the same ckpt at four schedules — 100, 40, 20, 10
effective steps — with 512 samples each (seed=42), then computed:

- **Diversity**: mean pairwise vertex L2 distance between sampled meshes
- **Interpenetration**: mean # of person-0 vertices within 20 mm of person-1
  (severe = >30 such)
- **Self-FID**: Fréchet distance vs the slowest schedule's feature distribution
  (per-sample mean+std summary)

| n_steps | diversity (mm) | severe interpen rate | self-FID |
|---|---:|---:|---:|
| 100 (paper baseline) | 465.1 | 0.010 | 0.000 |
| 40 | 457.7 | **0.008** | 0.028 |
| 20 | 454.0 | 0.010 | 0.136 |
| 10 | 420.3 | 0.016 | **0.432** |

Outputs: `repro/phase_a/outputs/uncond/{results.csv, pareto.png, diversity.png}`.

**Headline finding** (not in paper): the 40-step schedule is a sweet spot — 2.5×
faster than the paper's baseline, lowest severe-interpenetration rate (0.8%),
and self-FID degradation is small (0.028). Below 20 steps, both diversity and
physical plausibility degrade sharply (self-FID jumps to 0.432 at 10 steps).

Caveat on self-FID: this is a self-referential proxy, not a true FID against
training data (we have no training PGT). Useful as a relative-quality measure
across schedules only.

## 3. What we did not reproduce, and why

### 3.1 Quantitative FID and perceptual study (C3, C4)

The paper's quantitative generation evaluation requires:
- **Training PGT** (`processed_pseudogt_fits.pkl` for FlickrCI3D) to compute
  against — the official Google Drive link is dead (upstream
  [issue #13](https://github.com/muelea/buddi/issues/13)), and re-fitting via
  `flickr_fits.yaml` would take ~30 GPU-h we did not budget. Upstream
  maintainer has not responded to issues since 2024-02.
- **VAE baseline ckpt** — not released by the authors.
- **Amazon Mechanical Turk** workers + study budget — out of scope.

We do not approximate these. Principled reproduction prefers honest gaps to
fabricated numbers.

### 3.2 Conditional sampling / BUDDI (gen.) baseline (C5)

The paper's conditional model is used in two ways, both blocked at the
test-set boundary:

1. **As the initialiser for the optimisation pipeline §3.2** — Phase B's scope
   (Tables 1-3). Requires the same FlickrCI3D/CHI3D/Hi4D test splits + PGT.
2. **As BUDDI (gen.) baseline** — direct denoised output of BEV, evaluated
   quantitatively in Tables 1-3. Requires the same test infrastructure.

Standalone *qualitative* cond sampling on the 3 demo BEV inputs bundled in
`demo/data/FlickrCI3D_Signatures/demo/` would have shown only "the cond ckpt
loads and produces output" — a code-path check rather than a paper claim.
Doing it cleanly requires extending `llib/data/single.py:load_data()` to accept
`dataset_name='demo'` (currently raises `NotImplementedError`). That is feature
engineering, not reproduction, so we leave it as a documented limitation.

## 4. Deviations from the paper

| Deviation | Why |
|---|---|
| PyTorch 2.0.1 + CUDA 11.8 instead of the repo's pinned PyTorch 1.9.1 + CUDA 11 | RTX 4070 (Ada, sm_89) needs CUDA ≥ 11.8 for native compute capability. Pinned stack would not run. See `repro/shared/docs/UPSTREAM_NOTES.md`. |
| Batch size 8 instead of the sbatch default 16 | 8 GB VRAM. Sampling output is identical (batch size only affects throughput, not stochastic noise per sample). |
| Skipped ViTPose / BEV / detectron2 install | Not needed for sampling-only pipeline. Demo BEVs are precomputed in the repo. |
| `pyrender` headless backend = EGL | WSL2 has no display server; EGL works without `sudo apt install libgl1`. |
| Added `--seed` parameter to `sample.py` (via patch in `repro/shared/patches/`) | Upstream samples non-deterministically. We seed `torch + numpy + random + cudnn.deterministic` so `results.csv` is reproducible across runs on the same GPU architecture. Pareto **trend** is robust across seeds; absolute numbers move within ~5% across seeds 7/42/123 (informal check). Upstream `sample.py` itself is byte-identical to origin/main — patch is applied at env-setup time. |

## 5. Compute used

| Step | Wall-clock |
|---|---:|
| Conda env build (PyTorch + pytorch3d wheels + smplx etc.) | ~6 min |
| Download essentials.zip (56 MB) | ~12 s |
| Download body models (~910 MB SMPL-X + SMPL + SMIL) | ~90 s |
| Unconditional sampling: 512 × 4 schedules + 512 renders | ~26 min |
| Analysis + gallery | ~10 s |
| **Total cold-start to deliverables** | **~33 min** |

## 6. Artifacts handed off (in `repro/phase_a/outputs/`)

Tracked in git (small):
- `gallery_uncond_baseline.png` — 8×8 mesh grid for slides
- `uncond/results.csv` — per-schedule metrics
- `uncond/pareto.png` — quality–speed Pareto curve (report headline figure)
- `uncond/diversity.png` — diversity vs DDIM steps
- `uncond/generate_1000_*_v0/cmd_args.txt` — exact command provenance per run

Gitignored (regenerable via `repro/phase_a/runners/run_uncond.sh`):
- `uncond/generate_1000_*_v0/x_starts_smplx.pkl` — 4 × 124 MB raw SMPL-X parameters
- `uncond/generate_1000_10_v0/renders/*.gif` — 512 360° meshes

## 7. How to reproduce these numbers

See `REPRODUCE.md` for the full zero-state instructions. TL;DR:

```bash
bash repro/shared/env/0_install_miniconda.sh
bash repro/shared/env/1_setup_env.sh
bash repro/shared/env/2_fetch_essentials.sh
SMPLX_USER=... SMPLX_PASS=... SMPL_USER=... SMPL_PASS=... \
  bash repro/shared/env/3_fetch_bodymodels.sh
bash repro/phase_a/runners/run_uncond.sh        # SEED=42 baseline; ~25 min on RTX 4070
bash repro/phase_a/runners/run_analyze.sh
```

Reproducibility scope: same `--seed` + same PyTorch (2.0.1+cu118) + same GPU
architecture (sm_89) → identical numbers. Different GPU architectures will give
numerically-close but not bit-identical numbers due to non-deterministic CUDA
atomic operations inside SMPL-X mesh evaluation that we did not budget time to
disable.

## 8. Phase B / C touchpoints

- **Phase B** (optimisation, Tables 1, 3): the cond ckpt + cond yaml load
  cleanly under the same env. Same `1_setup_env.sh` recipe works. The blocker
  is the dead FlickrCI3D test pickle. See `repro/phase_b/README.md`.
- **Phase C** (BEV-noise robustness): only needs the cond ckpt and a custom
  noise-injection driver; bypasses `SingleDataset` entirely so it avoids the
  test-pickle blocker. See `repro/phase_c/README.md`.
