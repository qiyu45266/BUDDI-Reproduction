# BUDDI Reproduction — Team Handoff

**Purpose**: capture decisions taken so anyone (any teammate, any device, or a
fresh Claude session) can pick up without re-reading the chat history.

**Last updated**: 2026-05-18

---

## 1. Project context

- **Paper**: Müller et al., *Generative Proxemics: A Prior for 3D Social
  Interaction from Images*, CVPR 2024 (BUDDI).
- **Upstream code**: https://github.com/muelea/buddi (cloned at repo root).
- **Course**: USI AdvTML reproduction project, **deadline 2026-05-25 23:59**;
  deliverable = report + 20-min talk.

## 2. Work split — phases, not silos

We split the reproduction into three sequential **phases** of work. Each phase
has a primary owner but all three of us review and run each other's code; the
env, patches and conventions in `repro/shared/` are common to all three.

| Phase | Scope | Primary owner | Status |
|---|---|---|---|
| A — Generation analysis (paper §5.1) | Unconditional sampling from pretrained ckpt + DDIM-step Pareto study | (rotating) | ✅ Complete |
| B — Optimization-based fitting (paper §5.2, Tables 1, 3) | Reproduce fitting metrics where data permits | (rotating) | 🟡 In progress |
| C — Robustness extension (beyond paper) | RQ on cond-model behaviour under perturbed BEV inputs | (rotating) | 🟡 In progress |

## 3. Hard decisions already taken

1. **Drop Hi4D dataset access** — requires email approval, lead time too long.
   Train (if applicable) on Flickr + CHI3D only, mapping to the paper's
   `BUDDI (F,C)` row in Table 2.
2. **Drop full-from-scratch training** (decided 2026-05-18). The FlickrCI3D
   pseudo-GT Google Drive link is dead (upstream
   [issue #13](https://github.com/muelea/buddi/issues/13)); reproducing
   `processed_pseudogt_fits.pkl` would require ~15-30 GPU-h of `flickr_fits.yaml`
   re-fitting that we did not budget. The upstream maintainer has not responded
   to issues since 2024-02 — emails won't unblock us in time.

   **Update 2026-05-22**: we obtained the **raw originals** of both datasets
   (FlickrCI3D images + contact JSON, CHI3D videos + SMPL-X JSONs) — 8.4 GB of
   tarballs from `ci3d.imar.ro`. This **does not** unblock training (we still
   lack the processed pseudo-GT pickles for FlickrCI3D and the
   `images_contact_processed.pkl` indexes for CHI3D that `llib/data/preprocess/*`
   requires). It **does** unblock Phase A v2 (FID against real CHI3D
   distribution) by reading raw `smplx/<action>.json` directly. See
   `repro/phase_a/REPRODUCIBILITY_REPORT.md §2.3`. Raw data lives at
   `datasets/original/{FlickrCI3D_Signatures,CHI3D}/` (gitignored, ~10 GB
   extracted).
3. **Phase A pivot** to pretrained-ckpt-only inference. Headline study: DDIM
   step-count vs sample quality (which the paper itself does not analyze).
4. **Reproml.org principles applied**: claim-by-claim honest audit; nothing
   fabricated; deltas from the paper's exact protocol explicitly recorded in
   `repro/phase_a/REPRODUCIBILITY_REPORT.md`.

## 4. Environment notes

- **Local laptop (RTX 4070 Laptop, sm_89, 8 GB)** is the primary target.
  Recipe: `repro/shared/env/{0_install_miniconda, 1_setup_env, 2_fetch_essentials,
  3_fetch_bodymodels}.sh`. Cold-start to working `hhcenv39` env: ~10 min.
- **Why we don't use upstream `install_conda_env.sh`**: it pins PyTorch 1.9 +
  cudatoolkit 11, which has no kernels for compute capability ≥ sm_89. Full
  explanation in `repro/shared/docs/UPSTREAM_NOTES.md`.
- **WSL2 Ubuntu 22.04/24.04 or native Linux** both work; tested on WSL2 Ubuntu
  24.04.3 LTS + NVIDIA driver 580.88.

## 5. Phase A — what's done

See `repro/phase_a/REPRODUCIBILITY_REPORT.md` for full details.

- 512 unconditional samples × 4 DDIM schedules (100 / 40 / 20 / 10 steps) from
  `buddi_unconditional.pt` at seed=42
- Diversity / interpenetration / self-FID metrics per schedule (`results.csv`)
- Quality-vs-speed Pareto curve (headline figure, NOT in paper)
- 8×8 sample gallery (paper §5.1 qualitative analogue)
- **FID vs real CHI3D contact-frame distribution** (added 2026-05-22 after
  obtaining raw data; see §3 update). Paper claim C3 status: ❌ → 🟡 partial.
  40-step sweet spot confirmed on this independent metric too.
- All artifacts under `repro/phase_a/outputs/`; large `.pkl` / `.gif` files
  gitignored but regenerable via `repro/phase_a/runners/{run_uncond,run_chi3d_fid}.sh`

## 6. Phase B / C — what's open

See `repro/phase_b/README.md` and `repro/phase_c/README.md` for starting points
and known blockers. Brief summary:

- **B**: B1 (demo sanity fit on 3 bundled images) works out of the box.
  B2 (test-set evaluation of paper Tables 1–3) is blocked by the dead PGT link
  (same as the training blocker in §3). Four options listed in the phase README,
  including the honest "negative result" path.
- **C**: Can run **independently** of A and B. Only needs the cond ckpt; the
  proposed RQ (BEV-input perturbation sweep) sidesteps the dead-PGT blocker
  entirely by constructing its own conditioning inputs.

## 7. Where to look first when resuming

1. `repro/README.md` — team workspace overview
2. `repro/HANDOFF.md` — this file
3. Your phase: `repro/phase_<x>/README.md`
4. If env not yet built: `repro/shared/env/0_install_miniconda.sh` then `1_setup_env.sh`

## 8. Open questions / unresolved

- Whether CloseApp follow-up paper releases SMPL-X fits before deadline
  (decision: don't wait).
- Whether Phase B can salvage CHI3D-only evaluation (Table 3) by re-running
  `datasets/scripts/CHI3D/{extract_frames.py, project_joints_chi3d.py}` against
  raw CHI3D MoCap. Not yet attempted.

## 9. Citation for the report

```bibtex
@inproceedings{mueller2024buddi,
  title={Generative Proxemics: A Prior for {3D} Social Interaction from Images},
  author={M{\"u}ller, Lea and Ye, Vickie and Pavlakos, Georgios
          and Black, Michael J. and Kanazawa, Angjoo},
  booktitle={CVPR}, year={2024}
}
```
