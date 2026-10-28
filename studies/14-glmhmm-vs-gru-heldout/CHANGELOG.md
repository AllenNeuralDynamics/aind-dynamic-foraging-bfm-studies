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
- HPC sweeps set `model.num_workers=10`, and the launch commands request 32 cores: restarts
  and refits run 10 at a time (AllenNeuralDynamics/aind-dynamic-foraging-models#86), about
  5× faster for a K = 10 cell. PLAN.md's Compute section records the measurement and the
  `max_em_iters` 500 cap (AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher#207).
- The Beaker templates name the built GLM-HMM image,
  `han-hou/dynamic-foraging-bfm-glmhmm-20261027`; its smoke test has not run yet.
