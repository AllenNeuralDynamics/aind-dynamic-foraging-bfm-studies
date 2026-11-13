# Variant `pool-folds`

**What.** Four new 3-way GRU trainings (study 02 `nxd-3way` recipe, H256, seed 0), one Cedar session each, one per pool
fold. Fold model k trains on every mouse outside pool fold k (the other 3 pool folds plus the 149
test subjects, about 610 mice), then scores fold k's mice in passes A and B. Fold membership:
`reference/folds.json`, committed before launch; `sweep.yaml` is generated from
`sweep.template.yaml` by `make freeze-folds` (grid point k = fold model k).

**Why.** Gives every training-pool mouse a held-out GRU prediction, the predictability map that
study 41 uses as its ceiling and residual map.

**Launches.** One launch, 4 tasks, one W&B group `pool-folds@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Status.** Not launched. Training does not need the wrapper prerequisite; scoring does.
