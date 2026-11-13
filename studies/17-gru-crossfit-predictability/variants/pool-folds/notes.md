# Variant `pool-folds`

**What.** Four new 3-way GRU trainings (study 02 `nxd-3way` recipe, H256, seed 0), one Cedar session each, one per pool
fold. Fold model k trains on every mouse outside pool fold k (the other 3 pool folds plus the 149
test subjects, about 610 mice), then scores fold k's mice in passes A and B. Fold membership:
`reference/folds.json`, committed before launch; `sweep.yaml` is generated from
`sweep.template.yaml` by `make freeze-folds` (grid point k = fold model k).

**Why.** Gives every training-pool mouse a held-out GRU prediction, the predictability map that
study 41 uses as its ceiling and residual map.

**Launches.** One launch, 4 tasks, one W&B group `pool-folds@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Status.**
- `folds-h256` (2026-11-13 00:00 PT, group `pool-folds@20261112-235836`, 4 A100 sessions at 256Gi):
  **failed, stopped by hand at 00:10 PT.** All 4 runs passed the pre-training evaluation, then host
  memory reached the 256 GiB limit in the first minutes of training: fold 4's container was
  killed for running out of memory (exit 137); folds 1–3 were at 88–100% of 256 GiB (fold 1 RSS
  256 GiB). H256 needs more host RAM than H128 (which peaked near 190 GB in `cedar-smoke`). GPU
  memory was not the limit (JAX's 30 GiB reservation of the 40 GB A100; no GPU OOM seen).
  Training does not need the wrapper prerequisite; scoring does.
