# Changelog — 14-glmhmm-vs-gru-heldout

## 2026-10-27

- Study created (#48): README design, PLAN.md, and variants `lag-scan`, `hurdle` and
  `reference` with HPC sweep manifests. `glmhmm` added to the model tag vocabulary. Not
  launched: waits for the code PRs to merge.
- Beaker manifests added beside the HPC ones: one GPU per task on the GLM-HMM image
  (AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper#111), which is still to be built.
- PLAN.md's Compute section now holds timings measured on this study's cohort instead of the
  scale probe's estimate. It also gives the CPU-until-GPU backend plan
  (AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper#111) and the `max_em_iters` 500
  follow-up (AllenNeuralDynamics/aind-dynamic-foraging-models#82).
- r2 reads states only as far as study 15 (#50) shows they are recoverable.

## 2026-10-27 (evening)

- `lag-scan` launched and done: smoke `lag-scan@20261027-194921`, then all 8 cells as
  `lag-scan@20261027-200129`, from main SHAs on HPC CPU. L = 10 maximises the selection score
  for both emissions, at the scan's upper bound; gains shrink from 0.005 (L 3 → 5) to under
  0.001 (L 7 → 10). Results in `variants/lag-scan/launch_record/lag-scan/results.md`.
