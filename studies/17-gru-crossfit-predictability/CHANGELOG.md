# Changelog — 17-gru-crossfit-predictability

## 2026-11-13

- Han: the cross-validated scaling curve is part of study 17. Added variant `scaling-folds`,
  `reference/scaling_subsets.json` (`make freeze-scaling`) and the `scaling` question tag.

## 2026-11-12

- Han: H256 instead of H128, and the fold trainings on Cedar. Froze `reference/folds.json`
  (`make freeze-folds`) and generated the `pool-folds` sweep and Cedar session file.

- Added variant `cedar-smoke`, the first Cedar launch (dispatcher #219), at Han's request.
- Study created (#148): README design, PLAN with agreed points and open decisions D0–D6, variant `test-fold-ref`.
- Design settled with Han: 5-fold rotation over all 763 mice, 3-way GRU, test fold reuses study 02 `nxd-3way`; added CONTEXT.md and variant `pool-folds`.
