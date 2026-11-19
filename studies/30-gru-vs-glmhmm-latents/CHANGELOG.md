# Changelog — 30-gru-vs-glmhmm-latents

## 2026-11-19

- r1 to report schema version 2, for going live (issue #136). No point value changed beyond the
  out-of-fold refit's reproduction tolerance (largest gap 4.3e-4, the untrained GRU at its iteration cap).
  - Uncertainty added: 95% intervals from a bootstrap over the 50 mice (`freeze_r1.py`, `bootstrap`), on
    out-of-fold predictions regenerated for every probe (`oof_r1.py`; the pilot saved them for latents
    and history only).
  - Findings regrouped (Han, 2026-11-18): K = 3 and the K = 2–10 scan are one finding (F1); four
    findings, each with a figure (Fig. 4 split out of the K-scan figure); supplementary items
    SupFig. 1–3.
  - Removed: the hurdle − reference gap sentence, the GLM-HMM likelihood column, the ridge-check
    markers, the PC slow share and Rastermap seed stability (no interval).
  - F4 no longer cites study 14's per-mouse refit as favouring K = 3: study 14's live r2 replaced that
    refit (claims check run 1).

## 2026-11-11

- PLAN.md: r1 leakage audit, the `heldout-mice` variant (steps, reproduction check, open W&B decision), the
  no-upload decision for r1's intermediates, later follow-ups.

## 2026-11-08

- Study created (issue #136).
- r1 (draft): training-mice pilot. Freeze `freeze_r1.py` of the pilot's scratch outputs (code in
  `analysis/pilot/`), offline producer `r1_pilot_decoding.py`.
