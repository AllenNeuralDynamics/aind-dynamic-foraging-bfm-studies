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
  and refits run 10 at a time (AllenNeuralDynamics/aind-dynamic-foraging-models#86), 3–5×
  faster for a K = 10 cell. PLAN.md's Compute section records the measurement and the
  `max_em_iters` 500 cap (AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher#207).
- The Beaker templates name the built GLM-HMM image,
  `han-hou/dynamic-foraging-bfm-glmhmm-20261027`; its smoke test has not run yet.
- 20 restarts per K > 1 (was the config's 10), as Ashwood et al.; on HPC two rounds of 10
  workers on 32 cores. 20 workers on 40 cores measured no faster.

## 2026-10-27 (evening)

- `lag-scan` launched and done: smoke `lag-scan@20261027-194921`, then all 8 cells as
  `lag-scan@20261027-200129`, from main SHAs on HPC CPU. L = 10 maximises the selection score
  for both emissions, at the scan's upper bound; gains shrink from 0.005 (L 3 → 5) to under
  0.001 (L 7 → 10). Results in `variants/lag-scan/launch_record/lag-scan/results.md`.

## 2026-10-28

- r1 review fixes: each state's mouse concentration (top 10% of mice's share) now uses each
  mouse's mean occupancy, so mice with more trials no longer count for more; the reference
  weight figure shows both the R − L contrast and the common part (L + R)/2. The freeze was
  rerun; only `top10pct_subject_share` changed.
- K grids finished: `hurdle@20261027-212905` and `reference@20261027-213006`, all 20 cells.
  Hurdle K = 5-10 were killed after scoring by the kernel's memory-map limit
  (AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper#119). Their scores are in W&B and
  `fit.json`; the fix is wrapper#120.
- r1 (draft): `freeze_round1.py` and `r1_round_1.py`. Scores under three protocols (refit,
  global fit on the same mice, held-out), the GRU comparison, the states of each emission's
  selected K, example sessions drawn with `plot_foraging_session`, timescales, and restart and
  alignment diagnostics.
- Correction before review: per-mouse refits reorder states as K grows, so state summaries now
  use posteriors under the global fit. The side-biased states at K >= 8 span whole sessions,
  not short epochs.

## 2026-11-01

- r1 also publishes the held-out **zero-shot** (global fit only) L/R-on-engaged score per K (#84):
  `scores/<emission>/<K>/LR_engaged_zero_shot` in `r1_round_1.json`, a new result-1 column, and
  a dashed line in `fig_r1_scores` panel d. The value was already in `frozen_round1.json`; no
  existing number changed. It rises to K = 10 (hurdle 0.7111 → 0.7224), while the refit score
  peaks at K = 3 (0.7207). The atlas leaderboard draws both.
