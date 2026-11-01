# Changelog — 14-glmhmm-vs-gru-heldout

## 2026-11-01

- Variant `refit-prior` (round 2, #48): per-mouse refits with the weight prior centred on the
  global fit, σ ∈ {0.25, 0.5, 1, 2}, K = 1–10, both emissions, reusing round 1's global fits
  (manifest in the variant). Selection rule and checks are stated in its notes before launch.

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
