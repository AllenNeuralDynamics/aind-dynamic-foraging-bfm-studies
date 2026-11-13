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
- `folds-h256-320g` (2026-11-13 00:11 PT, group `pool-folds@20261113-001051`): Cedar A100 relaunch
  at 320Gi (4 x 320Gi fills the 1.3 TiB node). **Failed:** fold 1 was killed for running out of
  memory at 00:29 PT, early in training; fold 2 had reached 292 GiB. All 4 stopped at 00:31 PT.
  H256 at ~610 mice needs more than 320 GiB of host RAM, more than the A100 node can give 4 runs.
- `folds-h256-h200` (2026-11-13 00:31 PT, group `pool-folds@20261113-003134`): Cedar H200 node at
  600Gi. **Did not start:** the node had no free GPU ("largest free slot on one node is 0.687
  vCPUs, 38.4 GiB, and 0 GPUs"); the launcher stopped after the first session. Nothing ran.
- `folds-h256-beaker` (2026-11-13 00:15 PT, group `pool-folds@20261113-001528`, Beaker experiment
  [`01M79ZKT8PKHHVQH93ZN1V3450`](https://beaker.org/ex/01M79ZKT8PKHHVQH93ZN1V3450)): the same 4
  fold models on Beaker's allocated on-prem H200 slots (normal priority, not preemptible, 384GiB),
  run in parallel with the Cedar relaunch at Han's request. **Stopped at 00:26 PT, during setup:**
  384GiB exceeded one H200 bundle (1 GPU + 28 CPU + 378 GiB), so each task took 2 GPUs.
- `folds-h256-beaker-1gpu` (2026-11-13 00:28 PT, group `pool-folds@20261113-002829`, Beaker
  experiment [`01M7A0BPTN41XJ5TGJGAVNP5Z7`](https://beaker.org/ex/01M7A0BPTN41XJ5TGJGAVNP5Z7)):
  resubmitted at 370GiB / 28 CPU; verified 1 GPU per task.
  **W&B offline:** the Beaker secret `han-wandb-api-key` is the `houhan` account, which is not in
  the `NeuralDynamics` team; `wandb.init` got 403 "permission denied" three times and the wrapper
  continued with W&B offline (run data in each job's `/results/wandb/offline-run-*`). **Stopped
  at 00:58 PT** to resubmit with a `NeuralDynamics` key (Beaker secret `han-wandb-api-key-neuraldynamics`). The first launch (this or the Cedar
  relaunch) to finish all 4 folds supplies the map; the other is stopped and recorded. Outcome:
  pending.

Training does not need the wrapper prerequisite; scoring does.
