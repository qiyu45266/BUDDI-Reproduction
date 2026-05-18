# Phase B — Optimization-based fitting

Reproduce paper §5.2 + Tables 1 (FlickrCI3D Signatures) and Table 3 (CHI3D),
optionally Table 2 (Hi4D), using BUDDI as the prior inside the
score-distillation-style optimization loop.

## What this phase covers

The "fitting with BUDDI" rows in the paper's tables — i.e. taking BEV-regressed
initial estimates and refining them by minimising
`L_fitting + L_diffusion (Eq. 3)` over SMPL-X parameters of two interacting people.
The diffusion-prior loss is computed against the conditional model
(`buddi_cond_bev.pt`) at noise level `t = 10` (paper §3.2).

## What's already done for you

| What | Where |
|---|---|
| Env `hhcenv39` (PyTorch 2.0.1+cu118, sm_89-friendly) | `repro/shared/env/1_setup_env.sh` — same recipe as Phase A; no re-install needed |
| `buddi_cond_bev.{pt,yaml}` checkpoint | `essentials/buddi/` after `2_fetch_essentials.sh` |
| SMPL-X / SMPL / SMIL body models | `essentials/body_models/` after `3_fetch_bodymodels.sh` |
| Working `llib/methods/hhcs_optimization/` code path (upstream, MIT) | `llib/methods/hhcs_optimization/main.py` — untouched |
| Demo data for sanity-fitting on 3 bundled FlickrCI3D images | `demo/data/FlickrCI3D_Signatures/demo/` (upstream, restored) |
| `--seed` patch (so your numbers are reproducible) | Applied automatically by `1_setup_env.sh` |

## Two paths

### B1 — Demo sanity run (always works, low-effort)

```bash
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate hhcenv39
cd <repo-root>
export PYTHONPATH=$(pwd)

python llib/methods/hhcs_optimization/main.py \
  --exp-cfg llib/methods/hhcs_optimization/configs/buddi_cond_bev_demo.yaml \
  --exp-opts \
    logging.base_folder=repro/phase_b/outputs/demo_fit \
    datasets.train_names=['demo'] \
    datasets.train_composition=[1.0] \
    datasets.demo.original_data_folder=demo/data/FlickrCI3D_Signatures/demo \
    datasets.demo.image_folder=images \
    model.optimization.pretrained_diffusion_model_ckpt=essentials/buddi/buddi_cond_bev.pt \
    model.optimization.pretrained_diffusion_model_cfg=essentials/buddi/buddi_cond_bev.yaml \
    logging.run=fit_buddi_cond_bev_demo
```

This runs the optimisation on the 3 demo images with precomputed BEV + ViTPose.
Output: per-image fits + visualisations under `repro/phase_b/outputs/demo_fit/`.
**Does not reproduce a paper number** — it's a "does the optimisation loop work
end-to-end" check.

### B2 — Test-set evaluation (the actual paper claim — blocked)

To reproduce Table 1 you need:
- `datasets/original/FlickrCI3D_Signatures/test/` (raw images + keypoints)
- `datasets/processed/FlickrCI3D_Signatures/` containing
  `*_diffusion.pkl` and the pseudo-GT fits (paper's "Flickr Fits")

The processed pickle's official Google Drive link is **dead** (upstream
[issue #13](https://github.com/muelea/buddi/issues/13), no maintainer response
since 2024-02). Without it, `llib/data/single.py:load_data()` raises a
`FileNotFoundError` when constructing the dataset.

Concrete options:
1. **Try contacting the author group again** — `lea.mueller@tuebingen.mpg.de`.
   HANDOFF §9 notes prior attempts went unanswered, but worth one more try.
2. **Re-fit pseudo-GT from scratch** via the `flickr_fits.yaml` config (paper
   §4 "Flickr Fits"). Cost estimate from HANDOFF: ~30 GPU-h. Probably out of budget.
3. **Drop FlickrCI3D**, report only CHI3D (Table 3) and / or Hi4D (Table 2).
   CHI3D processed data may be reconstructible from the raw MoCap by running
   `datasets/scripts/CHI3D/{extract_frames.py, project_joints_chi3d.py}` —
   we have not tried this; it might also need PGT.
4. **Honest write-up**: reproduce only what we can (B1 above + qualitative
   inspection of Phase A's gallery vs cond-fit output) and put "Table 1 not
   reproduced — PGT data unavailable, see UPSTREAM_NOTES" in the report. This
   is what reproml.org calls a "principled negative result" — perfectly fine.

## Suggested deliverables

- `repro/phase_b/outputs/demo_fit/` — qualitative renders of B1
- `repro/phase_b/REPRODUCIBILITY_REPORT.md` — same structure as Phase A's,
  claim-by-claim, calling out the PGT blocker explicitly
- (If B2 path opens up) `repro/phase_b/outputs/<dataset>_eval/` + a metrics CSV

## Coordination touchpoints with Phase A

- Phase A's `repro/phase_a/outputs/uncond/generate_1000_10_v0/x_starts_smplx.pkl`
  is sampled from the **unconditional** model. Phase B's `BUDDI (gen.)` baseline
  in Table 1 is the **conditional** model's direct denoised output — different,
  but the sample-loading + SMPL-X rendering pipeline is shared. Reuse Phase A's
  `repro/phase_a/code/make_gallery.py` if you need to visualise.
- If B builds new code (custom dataset loader, eval glue), put it in
  `repro/phase_b/code/`, mirroring Phase A's layout.
