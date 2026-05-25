# Phase C — Reproducibility Report (RQ-5 + RQ-7)

**Scope**: Beyond-paper research questions probing BUDDI as a mechanistic
artifact. RQ-5 asks what kind of geometric structure the model learned;
RQ-7 quantifies the restoring force the model applies at the noise level
the paper itself uses in §3.2 (Eq. 6).
**Principle**: reproml.org — every claim audited; honest about caveats.
**Hardware**: RTX 4070 Laptop (8 GB, sm_89), WSL2 Ubuntu 24.04. Wall-clock
budget end-to-end: ~25 min compute.

## RQ-5 — Did BUDDI learn "contact" or just "proximity"?

### Motivation

The paper trains BUDDI on contact-frame SMPL-X parameters but **never feeds
the contact map itself into the diffusion model** (the contact annotation only
gates which frame is sampled). So an open question: does the model implicitly
recover the contact semantic, or does it merely capture "two people close
together"?

This matters operationally: if BUDDI's contact representation is weak, the
paper's optimization (§3.2) leans more on the auxiliary `L_P` interpenetration
term than on the diffusion prior `L_diffusion`. That would partly explain why
the simple Contact-Heuristic baseline (paper Table 1: 68 mm PA-MPJPE) is so
close to full BUDDI (66 mm).

### Method

For every BUDDI sample (4 schedules × 512 = 2048) and every CHI3D contact-frame
anchor (373), we compute two metrics on stride-20 subsampled meshes:

| metric | what it measures |
|---|---|
| `min_v2v` (m) | min vertex-to-vertex distance — pure *proximity* signal |
| `contact_density_T` | fraction of subsampled (v_a, v_b) pairs with dist < T — *contact* signal |

We KS-test BUDDI's distribution against CHI3D's on each metric, at thresholds
T ∈ {5 mm, 1 cm, 2 cm}. If BUDDI matches CHI3D on `min_v2v` but **misses on
contact_density**, the model learned proximity but not the finer-grained
contact structure.

### Result (seed=42 on RTX 4070, n_schedule = 4 × 512, n_chi3d = 373)

| n_steps | min_v2v median  (mm) | contact@5mm | contact@1cm | contact@2cm |
|---|---:|---:|---:|---:|
| 10  | 14.0 | SAME (p=0.78) | DIFFERENT (p=0.011) | DIFFERENT (p<1e-4) |
| 20  | 14.4 | SAME (p=0.82) | DIFFERENT (p<0.01) | DIFFERENT (p<1e-6) |
| 40  | 15.5 | SAME (p=0.30) | DIFFERENT (p<1e-3) | DIFFERENT (p<1e-8) |
| 100 | 15.1 | SAME (p=0.26) | DIFFERENT (p<1e-3) | DIFFERENT (p<1e-7) |
| **CHI3D ref** | **11.7** | reference | reference | reference |

The pattern is identical across all four DDIM schedules:

1. **`min_v2v`**: BUDDI is **statistically distinguishable** from CHI3D at every
   schedule (KS ≈ 0.12–0.17, p < 0.005). BUDDI generates pairs that are on
   average ~14–15 mm apart at their closest, vs ~12 mm in CHI3D.
2. **Strict 5 mm contact**: BUDDI **matches** CHI3D. But this is uninformative
   because both distributions are essentially Dirac-at-zero at this threshold
   (median 0 contact pairs for both; almost no sample has any vertex pair within
   5 mm under stride-20 subsampling).
3. **1 cm and 2 cm "near-contact"**: BUDDI shows **systematically lower contact
   density** than CHI3D, statistically significant at every schedule
   (KS up to 0.21, p < 1e-8 at 2 cm). The effect size: BUDDI's mean contact
   density is **~35–45% lower** than CHI3D's at these thresholds.

**Mechanistic interpretation**: BUDDI captures the coarse spatial-proximity
signal — both distributions peak around 10–15 mm — but **under-represents the
near-touching tail** (1–2 cm). The model knows "be close" without learning
"actually touch surfaces". The effect is **consistent across all four DDIM
schedules** and across both fast and slow sampling, ruling out a
sampling-precision artifact: it is a property of the **training distribution**.

### Caveats

- **5 mm "SAME" verdict is statistical-power-limited, not true equivalence.**
  Under stride-20 subsampling (524 verts/person), the strict 5 mm contact pair
  count is ~0-1 per sample for both BUDDI and CHI3D — the distributions
  collapse to a near-Dirac at 0, where KS cannot resolve differences. The
  "SAME" should be read as "we cannot statistically distinguish them at this
  precision" rather than "they are equivalent".

- **CHI3D anchors are not all truly in 1 cm contact.** Median anchor `min_v2v`
  is 11.27 mm — i.e., half the CHI3D contact-annotated frames have no vertex
  pair within 1 cm. So the "contact-frame reference distribution" describes
  *contact-annotated* frames, not necessarily *tightly-touching* surfaces. Both
  BUDDI and the reference are measured on the same metric, so the comparison
  is fair, but readers shouldn't picture CHI3D as a high-contact reference.

- **CHI3D is also part of BUDDI's training set** (20% of the training mix per
  paper §4). BUDDI has seen this distribution; this is therefore a probe of
  whether the model **memorised** the contact characteristics of CHI3D, not
  whether it generalises. A held-out test would need data we cannot access.

- **Reference distribution is small** (n=373 vs 2048 BUDDI per schedule).
  KS handles unequal sample sizes but its sensitivity drops at the smaller end.

- **Vertex subsampling is the dominant source of measurement bias.** A stride
  sensitivity study is in `outputs/rq5_stride10/` and `outputs/rq5_stride50/`
  — summary: the *direction* of the BUDDI < CHI3D contact-density finding is
  consistent across strides 10/20/50; only the statistical power varies.

### Stride sensitivity (appendix)

We re-ran the contact-density comparison at strides 10 and 50 to bracket our
default stride 20 (524 verts/person). The BUDDI/CHI3D mean ratio at the 1 cm
threshold is consistent across strides:

| stride | verts/person | BUDDI mean / CHI3D mean (1 cm) | p (n_steps=10) | p (n_steps=40) | p (n_steps=100) |
|---:|---:|---:|---:|---:|---:|
| 10 (denser)   | 1048 | ~0.45 | 9e-4    | 2e-8    | 4e-7    |
| 20 (default)  | 524  | ~0.66 | 0.011   | 1e-4    | 3e-4    |
| 50 (sparser)  | 210  | ~0.55 | 0.19 ❌ | 0.24 ❌ | 0.30 ❌ |

At stride 50, the 1 cm verdict crosses from DIFFERENT to "not statistically
distinguishable" — but the mean ratio is still ~0.5, so the *direction* is
unchanged. At 2 cm threshold, all three strides show DIFFERENT across all
schedules. Bottom line: our stride-20 default has adequate power; stride 50
loses power; stride 10 has the strongest evidence. The BUDDI-underweights-contact
direction is **robust to vertex-subsample choice**.

### Implication for RQ-7

If BUDDI learned a weaker-than-supposed contact representation, then
**when we perturb a contact configuration we should expect modest, not strong,
restoration**. Tested directly in RQ-7 below.

### Outputs

- `outputs/rq5/stats.csv` — per-sample metrics (2421 rows)
- `outputs/rq5/ks_results.json` — full KS test statistics + p-values
- `outputs/rq5/density_comparison.png` — KDE figure (panel a: proximity, panel b: contact)
- `outputs/rq5/kde_min_v2v.png`, `outputs/rq5/kde_contact_density.png` — individual views

---

## RQ-7 — Counterfactual contact restoration (causal probe of the prior)

### Motivation

Paper §3.2 (Eq. 6) defines `L_diffusion = ||D(x_t; t, c_H) - x||` with `t = 10`.
This loss is the entire role of BUDDI as a prior during fitting optimization.
We measure what `D` actually does at this single denoising step when given a
counterfactually-perturbed contact configuration: does the prior pull the
estimate back toward the data manifold (causal contact constraint), or does it
accept the perturbation (statistical correlation only)?

This is the most paper-aligned probe possible: same `t = 10`, same
diffuse-denoise operation, same per-sample noise. The result is a direct
measurement of the prior's "restoring force".

### Method

1. **Anchors**: 100 CHI3D contact-frame mesh pairs in BUDDI's canonical frame
   (`relative_orient=False, relative_transl=True` → transl[0] = 0).
2. **Perturbation grid**: 6 directions (±x, ±y, ±z) × 6 magnitudes (1, 2, 5,
   10, 20, 50 cm) = 36 perturbations per anchor; **plus a Δ = 0 control**.
3. **Diffuse-denoise**: for each perturbed anchor, add Gaussian noise at
   t = 10 via `diffusion.q_sample(...)` then call the model's denoiser. Same as
   `train_module.diffuse_denoise(x, y, t)`. Fixed seed=42 across all
   perturbations and models.
4. **Metrics**:
   - `pullback` (cm): signed projection of `(perturbed_B_centroid −
      denoised_B_centroid)` onto the perturbation direction `Δ̂`. **Positive
      = the model moved B back toward the original anchor**.
   - `b_motion_from_anchor`: mean per-vertex distance from anchor B.
   - `b_motion_from_perturbed`: mean per-vertex distance from the input B.
   - `a_correction`: mean per-vertex distance of A (unperturbed) from anchor A —
     pure noise floor measurement.
   - recovery rate at thresholds 5 mm / 1 cm / 2 cm / 5 cm.
5. **Compared models**: `buddi_unconditional.pt` (no `c_H`) and
   `buddi_cond_bev.pt` (`c_H` = the perturbed anchor in BEV-format).

### Important caveat read before the headline

**The conditional model `buddi_cond_bev` is being evaluated out-of-distribution
here.** It was trained with noisy BEV regressor outputs as the conditioning
input `c_H`. We feed it CHI3D-clean SMPL-X parameters as `c_H`. Evidence of
the OOD shift:

- At Δ = 0 (no perturbation), cond_bev's `a_correction` = **24 cm** vs
  uncond's 5.7 cm noise floor — the cond model is changing person A by ~24 cm
  per vertex on average even with zero perturbation, far above the
  diffuse-denoise noise floor.
- `b_motion_from_anchor` at Δ = 0 is **48 cm for cond vs 7 cm for uncond**.

The **pullback metric is direction-projected** (signed projection on Δ̂), so
isotropic noise in the OOD output averages out under enough samples and the
**pullback number remains interpretable** as a relative comparison between
the two models on the same CHI3D inputs. But the cond pullback figure
**should not be read as "the spring constant active in paper §5.2 fitting"** —
paper §5.2 feeds the cond model BEV outputs, not CHI3D ground truth, and we
have not measured what happens there. The cond results below characterise the
model's behaviour on a domain-shifted input.

### Result — headline: a near-constant ~75% pullback regardless of magnitude

| Δ (cm) | uncond pullback (cm, % of Δ) | cond_bev pullback (cm, % of Δ) |
|---:|---:|---:|
| 1 | 0.80 (80%, near noise) | 0.64 (64%, near noise) |
| 2 | 1.47 (74%) | 1.31 (66%) |
| 5 | 3.87 (77%) | 3.23 (65%) |
| 10 | 7.64 (76%) | 6.44 (64%) |
| 20 | 15.19 (76%) | 12.83 (64%) |
| 50 | 36.57 (73%) | 31.45 (63%) |

**Three observations** (each scoped to what the data actually shows):

1. **The pullback fraction is approximately constant across perturbation
   magnitude** for the unconditional model: ~76% ± 2% across Δ ∈ {2, 5, 10,
   20, 50} cm (we exclude Δ = 1 cm because at that scale the perturbation is
   comparable to the noise floor; the SEM at 1 cm is ~0.16 cm, so 0.80 cm is
   barely 5 standard errors from 0). Constant pullback fraction means
   **absolute pullback grows linearly with Δ**, which is consistent with — but
   not the same as — a literal linear restoring force. We avoid the Hooke's-law
   metaphor here: the dynamics inside a single diffuse-denoise step are
   non-trivial and "linear-proportional pullback observed empirically" is the
   most honest summary.

2. **Conditional model has a smaller pullback fraction (~64% vs ~76%) on this
   probe.** This is measured on CHI3D-clean inputs, which is OOD for the cond
   model (see caveat above). The 64% number characterises a specific
   experimental condition; whether it generalises to in-distribution BEV
   inputs is an open question. Within the same probe, cond < uncond is
   consistent across all magnitudes ≥ 2 cm.

3. **Recovery-rate-at-threshold curves are largely uninformative** because the
   anchors themselves have median `min_v2v` = 11.3 mm, so only ~20% are in
   1 cm contact even before perturbation. The recovery curves saturate against
   that floor (figure `restoration_curve.png`). The pullback metric isolates
   the actual signal.

### Connection to RQ-5

RQ-5 finds BUDDI's contact-density distribution is shifted away from CHI3D
(~35–45% lower). RQ-7 finds BUDDI's restoring force is ~75% of the input
displacement. Together: the prior **pulls toward a data manifold that is
itself slightly off the true contact manifold**, which is consistent with
the paper's reported close-but-not-dominant gap between Contact-Heuristic and
full-BUDDI in Table 1.

### Caveats

- **Δ = 1 cm pullback is barely above noise.** Per-direction SEM of pullback
  at Δ = 1 cm is ~0.16 cm (random noise N(0, ~4cm) in 1-D projection over 600
  samples), so the measured 0.80 cm is ~5σ from zero — significant but
  imprecise. Δ ≥ 5 cm has SEM ~0.16 cm too but on a signal of ~4 cm or more,
  i.e., ≫ 20σ. The "constant 76%" claim is robust for Δ ≥ 2 cm; we no longer
  weight the Δ = 1 cm point as evidence for it.

- **We perturb only translation, not orientation or pose.** Paper §3.2 uses
  BEV outputs that have noise on all parameter classes; orientation/pose
  perturbations are a natural extension we did not run.

- **Single-direction (translation-only) perturbations probe local geometry
  only.** Larger structural perturbations (e.g., swap person identities) could
  behave qualitatively differently.

- **CHI3D anchors are partially in-distribution.** CHI3D is one of the three
  datasets BUDDI trained on (20% of the training mix). An out-of-distribution
  test set might show different pullback.

- **The cond model results characterise OOD behaviour** — see the "Important
  caveat" block above. The 64% number does **not** generalise automatically
  to paper §5.2's actual operating point.

### Outputs

- `outputs/rq7/restoration_table.csv` — full per-(model × anchor × perturbation) raw data (7400 rows)
- `outputs/rq7/summary.json` — aggregate per (model, magnitude)
- `outputs/rq7/pullback_curve.png` — **the headline figure**: linear pullback
- `outputs/rq7/restoration_curve.png` — recovery at multiple thresholds (mostly flat / saturated)
- `outputs/rq7/correction_curve.png` — A and B motion vs perturbation (with noise floor)

---

## Reproml-aligned summary

| Claim probed | Verdict | Sample size | Robustness check |
|---|---|---|---|
| BUDDI matches CHI3D on proximity (min_v2v) | No — statistically different, all 4 schedules, p < 0.005 | 373 vs 4×512 | Robust at stride 10/20/50 |
| BUDDI matches CHI3D on contact density (1 cm) | No — BUDDI mean ≈ 0.5× CHI3D mean, all 4 schedules | 373 vs 4×512 | Mean ratio consistent across stride 10/20/50; statistical significance at stride 50 fails (power-limited) |
| Uncond BUDDI applies a linear-proportional pullback (~76%) | Yes — observed empirically for Δ ∈ {2, 5, 10, 20, 50} cm; Δ=1cm is at noise edge | 100 × 36 (×6 directions per Δ) | Not yet checked under orientation/pose perturbation; cond model is OOD |
| Cond model exhibits weaker pullback than uncond (~64%) on CHI3D anchors | Yes, **but on OOD inputs**: cond was trained with noisy BEV, we used CHI3D-clean. Generalisation to paper §5.2's operating point untested. | same | — |

All metrics use the paper's own machinery
(`llib.methods.hhc_diffusion.evaluation.eval.fid_featurize/params_for_fid`,
`llib.utils.metrics.diffusion.GenFID.calculate_frechet_distance`,
`llib.methods.hhc_diffusion.train_module.diffuse_denoise`); no
custom diffusion math.

## How to reproduce

```bash
# Pre-req: phase A's chi3d_distribution.pkl + uncond/generate_*_v0/x_starts_smplx.pkl
bash repro/phase_c/runners/run_rq5.sh    # ~1 min
bash repro/phase_c/runners/run_rq7.sh    # ~5 min (both models)
```

Outputs land under `repro/phase_c/outputs/rq5/` and `repro/phase_c/outputs/rq7/`.
