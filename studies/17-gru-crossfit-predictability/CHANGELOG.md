# Changelog — 17-gru-crossfit-predictability

## 2026-11-15

- r1 draft (cross-validated predictability): main findings F1–F4, supplementary S1–S4 (Han: scientific
  results in main figures, validation and housekeeping in supplementary).
- Claims check (claims-check/1 @ c234bff): all 4 claims supported; fixes: F2 says "stable within a mouse",
  F3 states its 760 mice, F4 adds a stratified null (`reference/pool_strata.json`, `make
  freeze-pool-strata`) and an interval for the test fold's offset, which is within random 149-mouse sets,
  so the draft's "slightly hard test split" reading was dropped; new caveats on subset rules and model origins.
- Claims check run 2 (on 586c1e8): all 4 claims supported; fixes: title no longer compares per-mouse
  SD with fold SD, "gain" vs "summed log-likelihood difference" named apart, the test-split reading
  states the check's limited power, Fig. 4 caveat names the different mouse sets, new caveat on the
  pool-fold models training on the test mice, Provenance lists `pool_strata.json`. Both runs are kept
  as records (`analysis/provenance/claims_check_r1_*.md`, studies#160).
- `heldout-rescore-<run>:v1` backfilled on all 74 scoring runs (wrapper #129/#130); records in
  `variants/scoring/launch_record/*/backfill_maps.json`.

## 2026-11-13

- Han: add 2-way fold models (variant `pool-folds-2way`, H256, full size) for comparisons with
  models that skip ignored trials.

- Han: add D=300 to the scaling curve (4 pool-fold models + 1 test-fold model); no new H.

- Han: the cross-validated scaling curve is part of study 17. Added variant `scaling-folds`,
  `reference/scaling_subsets.json` (`make freeze-scaling`) and the `scaling` question tag.

## 2026-11-12

- Han: H256 instead of H128, and the fold trainings on Cedar. Froze `reference/folds.json`
  (`make freeze-folds`) and generated the `pool-folds` sweep and Cedar session file.

- Added variant `cedar-smoke`, the first Cedar launch (dispatcher #219), at Han's request.
- Study created (#148): README design, PLAN with agreed points and open decisions D0–D6, variant `test-fold-ref`.
- Design settled with Han: 5-fold rotation over all 763 mice, 3-way GRU, test fold reuses study 02 `nxd-3way`; added CONTEXT.md and variant `pool-folds`.
