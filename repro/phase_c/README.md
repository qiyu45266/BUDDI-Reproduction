# Phase C — Robustness extension (beyond-paper RQ)

A research question the paper does not address: **how does BUDDI's conditional
denoiser degrade as the BEV regressor's input gets noisier?**

The cond model is trained on clean BEV outputs (paper §4) and used as both an
optimization initializer (§3.2) and a standalone refinement baseline (the
`BUDDI (gen.)` row in Tables 1–3). If a deployment uses a weaker / older BEV
regressor, or if BEV fails on out-of-distribution images, the conditioning is
noisy. We want to quantify how that propagates.

## Concrete RQ candidates (pick one or combine)

1. **Synthetic Gaussian perturbation on BEV outputs**: add isotropic noise of
   std `σ` to BEV's predicted SMPL-X params before they enter the conditional
   diffusion, sweep `σ ∈ {0, 1, 2, 5, 10} mm` (on translation) / `{0, 5, 15, 30}°`
   (on pose), measure (a) vertex L2 between conditioned-sample vs clean-conditioned
   sample, (b) self-FID against the clean run, (c) interpenetration rate change.
2. **Token dropout on conditioning**: zero out random subsets of the
   conditioning tokens (paper §3.1 architecture: tokens per param per person).
   This simulates partial occlusion / BEV failure on one person.
3. **Mismatched conditioning**: swap person-A's BEV with person-B's BEV.
   Stress-test whether the cond model is brittle to identity / pose mismatch.

## What's already done for you

| What | Where |
|---|---|
| Env | `repro/shared/env/1_setup_env.sh` — same recipe as Phase A |
| `buddi_cond_bev.{pt,yaml}` | `essentials/buddi/` after `2_fetch_essentials.sh` |
| SMPL-X body model | `essentials/body_models/` after `3_fetch_bodymodels.sh` |
| Reference unconditional samples (for "clean baseline" comparisons) | `repro/phase_a/outputs/uncond/generate_1000_10_v0/x_starts_smplx.pkl` |
| Reference cond-sample plumbing (function `sample_conditional_with_inpainting` in `llib/methods/hhc_diffusion/evaluation/utils.py`) | upstream, read-only |
| Diversity / interpenetration / self-FID metric implementations | `repro/phase_a/code/analyze.py` — reuse directly |

## You can skip Phase B's blocker

Phase C does **not** need the dead FlickrCI3D PGT data. You can:
- Build your own tiny dataset of 8–32 BEV-style inputs (either by running
  ROMP/BEV yourself on a handful of in-the-wild images via
  `install_thirdparty.sh` + the upstream `demo.sh` flow, or by handcrafting
  synthetic SMPL-X parameter sets to use as `cH`).
- Skip the heavyweight `llib/data/single.py` SingleDataset path entirely; just
  call `sample_conditional_with_inpainting` directly with a dict you construct.

This independence is documented in `HANDOFF.md §7`.

## Suggested layout

```
repro/phase_c/
├── README.md                          ← this file
├── code/
│   ├── perturb_bev.py                 ← noise injector
│   ├── run_robustness_sweep.py        ← σ-sweep driver
│   └── analyze_robustness.py          ← reuse repro/phase_a/code/analyze.py helpers
├── runners/
│   └── run_robustness.sh
└── outputs/
    ├── sweep_results.csv
    └── degradation_curve.png          ← headline figure
```

## Suggested deliverables for the report

1. A **degradation curve**: x-axis = perturbation magnitude, y-axis =
   metric-of-choice (vertex L2 vs clean / self-FID / severe-interpen rate).
2. A short qualitative gallery showing 4 noise levels side-by-side for the same
   seed → demonstrates "what does X% degradation look like".
3. One paragraph **discussion** tying the curve back to Phase B's optimization:
   "since BUDDI is the prior in Eq. 3, conditioning noise N translates to fit
   error ≈ k·N in the optimization output" (validate or refute with B's outputs
   if available).

## Coordination touchpoints with Phase A and B

- Reuse Phase A's metric code (`repro/phase_a/code/analyze.py:diversity`,
  `interpenetration_rate`, `featurize`, `fid_gaussian`). Don't re-implement.
- If Phase B gets a working test-set eval (B2 path), Phase C can extend it by
  running the same eval at multiple noise levels — that becomes the most
  rigorous story for the report.
