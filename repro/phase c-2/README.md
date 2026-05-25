# Phase C — Beyond-paper RQs on BUDDI's prior

Mechanistic probes of what BUDDI actually learned and what the prior actually
does during paper-style optimization. Two RQs implemented:

## RQ-5 — Did BUDDI learn contact, or just proximity?

Compares BUDDI's generated samples (4 DDIM schedules × 512) against CHI3D's
real contact-frame distribution (373) on two metrics: min vertex-to-vertex
distance (proximity) vs fraction of vertex pairs in contact at multiple
thresholds. KS tests on each pair.

**Finding**: BUDDI captures proximity but **systematically under-represents
near-contact density at 1-2 cm** (KS-DIFFERENT at all 4 schedules, BUDDI ~40%
lower). Consistent across schedules → property of the training distribution,
not of sampling speed.

→ See `REPRODUCIBILITY_REPORT.md §RQ-5` and `outputs/rq5/`.

## RQ-7 — What restoring force does the prior apply at t=10?

The single-step diffuse-denoise operation at t=10 is exactly what paper §3.2
Eq. 6 uses inside the optimization loop. We perturb person B's relative
translation by Δ in 36 directions/magnitudes (1-50 cm) and measure how far
the model pulls B back.

**Finding**: **~76% linear-proportional pullback** for the unconditional
model, stable across Δ ∈ {2-50 cm}; **~64%** for the conditional model on
CHI3D-clean inputs (but note: cond was trained on noisy BEV, so this probe
is OOD for it — see report §RQ-7 "Important caveat"). The "constant pullback
fraction" pattern means absolute restoring distance grows linearly with
perturbation magnitude, but we stop short of calling it a literal
spring-constant: the per-step diffuse-denoise dynamics aren't strictly
linear, only the empirically-observed pullback magnitude is.

→ See `REPRODUCIBILITY_REPORT.md §RQ-7` and `outputs/rq7/`.

## Why these RQs (over the original "BEV noise" RQ)

The original Phase C RQ — perturb BEV inputs and observe degradation — is
a special case of the more general probe in RQ-7. The current RQ-7
quantifies the prior's restoring-force constant, which directly explains the
behavior the BEV-perturbation RQ would have observed. RQ-5 then provides the
"what was learned" context. Together they form a tighter mechanistic story.

## Layout

```
repro/phase_c/
├── README.md                          ← this file
├── REPRODUCIBILITY_REPORT.md          ← full claim-by-claim audit
├── code/
│   ├── contact_vs_proximity.py        ← RQ-5
│   └── counterfactual_contact.py      ← RQ-7
├── runners/
│   ├── run_rq5.sh
│   └── run_rq7.sh
└── outputs/
    ├── rq5/{stats.csv, ks_results.json, kde_*.png, density_comparison.png}
    └── rq7/{restoration_table.csv, summary.json,
              pullback_curve.png, restoration_curve.png, correction_curve.png}
```

## How to reproduce

Prerequisites (from Phase A):
- `repro/phase_a/outputs/chi3d_distribution.pkl` — built by
  `repro/phase_a/runners/run_chi3d_fid.sh`
- `repro/phase_a/outputs/uncond/generate_*_v0/x_starts_smplx.pkl` — built by
  `repro/phase_a/runners/run_uncond.sh`

```bash
bash repro/phase_c/runners/run_rq5.sh    # ~1 min, CPU+GPU
bash repro/phase_c/runners/run_rq7.sh    # ~5 min, GPU
```

## Coordination with Phase A and B

- **From Phase A**: RQ-5 and RQ-7 reuse the unconditional samples and CHI3D
  reference distribution Phase A built. No new sampling.
- **For Phase B**: RQ-7's pullback measurement gives an empirical
  characterisation of the prior's restoring behaviour on a specific test
  distribution (CHI3D contact frames). It does **not** directly equal the
  restoring force at paper §3.2's actual operating point (which uses noisy
  BEV outputs as input); a follow-up probe with BEV-format noisy `c_H` would
  be needed for a paper-aligned coefficient.
- **From RQ-5 / RQ-7 jointly**: BUDDI's generated samples sit slightly
  farther apart than CHI3D's contact frames, and the prior's single-step
  pullback restores ~76% of a translation perturbation toward those generated
  samples. This is consistent with — but not by itself proof of — the
  hypothesis that BUDDI's main contribution in paper Table 1's
  Contact-Heuristic-vs-BUDDI comparison (68 mm vs 66 mm) is pose-coherence
  rather than contact placement.
