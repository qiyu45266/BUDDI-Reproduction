# Combined report — Phase A + Phase C-1 + Phase C-2

This is the merged 6-page report covering:

- **Task 1** (Phase A, Author A): unconditional sampling + DDIM Pareto study + CHI3D-real-GT FID.
- **Task 2** (Phase B, Author C): paper Tables 1–3 reconstruction reproduction. **Deferred** — reported separately, not in this document.
- **Task 3** (Phase C-1, Nawfal): controlled BEV-noise robustness on the optimisation pipeline.
- **Beyond paper** (Phase C-2, Author A): RQ-5 (contact vs proximity) and RQ-7 (counterfactual contact restoration).

## Files

```
combined/
├── main.tex                  ← LaTeX source (TMLR-style, atml.sty)
├── main.bib                  ← bibliography
├── report.html               ← parallel HTML rendering
├── atml.sty, tmlr.sty, ...   ← style files (copied from Nawfal's draft)
├── math_commands.tex         ← math macros
├── README.md                 ← this file
└── figures/
    ├── pareto_v2.png         ← Phase A — DDIM Pareto (Fig. 1)
    ├── task3_noise_comparison.png  ← Phase C-1 (Fig. 2)
    ├── rq5_density.png       ← Phase C-2 RQ-5 (Fig. 3)
    ├── rq7_pullback.png      ← Phase C-2 RQ-7 (Fig. 4)
    ├── diversity.png         ← optional appendix
    └── gallery_uncond.png    ← optional appendix (7 MB)
```

## How to compile the LaTeX

```bash
cd repro/report/combined
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or upload the whole `combined/` folder to Overleaf as a project.

## How to view the HTML

Open `report.html` in any browser. All figures are relative to `figures/`.

## What still needs filling in

- `[Author A]` and `[Author C]` placeholders in the title block of both `main.tex` and `report.html`. Replace with actual names.
- The Task 2 follow-up document will live separately under `repro/phase_b/REPORT_PHASE_B.md` (or `.tex`); we'll cross-reference once that exists.
