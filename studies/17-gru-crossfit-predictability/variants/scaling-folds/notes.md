# Variant `scaling-folds`

**What.** The cross-validated scaling curve (PLAN.md, "Scaling curve"). For each pool fold k,
fold model k's training mice are subsampled to D ~ 10, 30, 100 and 300 (studies 01/02 subject
ratios 0.016, 0.049, 0.163, 0.489 per curriculum, nested, seed 0; `reference/scaling_subsets.json`),
and a 3-way GRU H256 (the `pool-folds` recipe) is trained on each: 4 folds x 4 D = 16 models.
D=300 was added on 2026-11-13 (Han), with one test-fold run at D=300 (`sweep_test_d300.yaml`:
study 02's recipe with `subject_ratio` 0.489, seed 0), since study 02's grid has no D=300. The
full-size point is the `pool-folds` model of the same fold; the test fold's curve is study 02's
`nxd-3way` grid (same recipe, D = 10, 30, 100, 614, seeds 0-2).

**Why.** Studies 01/02 measured scaling on the 149 test mice only. Scoring every mouse with a
model that never saw it gives the curve over all 763 mice, with fold-to-fold error bars, and
shows whether H256 is still climbing at full size (study 02: not plateaued).

**Launches.** Two launches, one W&B group each (docs/adr/0001): `sweep_d100.yaml` +
`experiment_beaker_d100.yaml` on allocated on-prem H200 seats (Han: the 8 allocated seats take
the longer runs), and `sweep_d10_d30.yaml` + `experiment_beaker_d10_d30.yaml` preemptible on AWS
H200. Records go to `launch_record/<label>/`.

**Status.**
- `d100` (2026-11-13 00:45 PT, group `scaling-folds@20261113-004534`, Beaker experiment
  [`01M7A1AXYCBM0SBEP32KXVXR56`](https://beaker.org/ex/01M7A1AXYCBM0SBEP32KXVXR56)): 4 tasks, allocated
  on-prem H200 seats. The first two submissions returned Beaker 409 "retryable database conflict"
  and created nothing; the third went through. Outcome: pending.
- `d10-d30` (2026-11-13 00:44 PT, group `scaling-folds@20261113-004435`, Beaker experiment
  [`01M7A1963XK8VXZVKQ1762NWWF`](https://beaker.org/ex/01M7A1963XK8VXZVKQ1762NWWF)): 8 tasks, low
  priority, preemptible, AWS H200. Outcome: pending.
  **W&B offline:** the Beaker secret `han-wandb-api-key` is the `houhan` account, which is not in
  the `NeuralDynamics` team; `wandb.init` got 403 "permission denied" three times and the wrapper
  continued with W&B offline (run data in each job's `/results/wandb/offline-run-*`). **Stopped
  at 00:58 PT** to resubmit with a `NeuralDynamics` key (Beaker secret `han-wandb-api-key-neuraldynamics`).
- `d100-nd` (2026-11-13 00:56 PT, group `scaling-folds@20261113-005618`, Beaker experiment
  [`01M7A1YKMY3S4E7B03S6QM76N7`](https://beaker.org/ex/01M7A1YKMY3S4E7B03S6QM76N7)) and `d10-d30-nd`
  (00:56 PT, group `scaling-folds@20261113-005652`, experiment
  [`01M7A1ZN00710G7QA68F0AMQNA`](https://beaker.org/ex/01M7A1ZN00710G7QA68F0AMQNA)): the live runs,
  resubmitted with the `NeuralDynamics` W&B key. Several submissions before them returned Beaker
  409 "retryable database conflict" and created nothing. Outcome: pending.
- `d300` (2026-11-13 01:04 PT, group `scaling-folds@20261113-010450`, Beaker experiment
  [`01M7A2E7G9AFRRYWNRDBNAWY75`](https://beaker.org/ex/01M7A2E7G9AFRRYWNRDBNAWY75)): 4 pool-fold
  models at D~300, low priority, preemptible, on-prem H200, one 378 GiB bundle each. **Stopped at 01:33 PT, no training done:** on the oversubscribed on-prem
  cluster (our 8 allocated + another workspace's 6 low-priority jobs) each task was evicted
  (exit 143) every 2-4 min during setup. Resubmitted at normal priority (Han).
- `test-d300` (2026-11-13 01:05 PT, group `scaling-folds@20261113-010520`, Beaker experiment
  [`01M7A2F6QR9216KY1F2NM3MNF6`](https://beaker.org/ex/01M7A2F6QR9216KY1F2NM3MNF6)): the test fold's
  D=300 model (study 02 recipe, `subject_ratio` 0.489, seed 0), same resources. **Stopped at 01:33 PT, no training done:** on the oversubscribed on-prem
  cluster (our 8 allocated + another workspace's 6 low-priority jobs) each task was evicted
  (exit 143) every 2-4 min during setup. Resubmitted at normal priority (Han).
- Normal priority, not preemptible (2026-11-13 01:33-01:35 PT, `*-normal` launch records): all
  tasks queued behind the 8 allocated seats; stopped at 01:41 PT without starting. Resubmitted
  at normal priority, preemptible (the unallocated budget).
- Normal priority, preemptible on on-prem H200 (`*-np` launch records, 01:37-01:38 PT): nothing was
  scheduled in 3 min (on-prem had 0 free GPUs; normal-preemptible jobs wait for a free GPU rather
  than evicting other workspaces' low-priority jobs). Stopped at 01:44 PT; moved to AWS H200,
  normal priority, preemptible, two bundles (2 GPUs, 480 GiB) per task.
- 2026-11-13 01:56 PT: stopped the AWS 2-bundle tier-2 launches (queued; AWS filled at 01:37 PT
  with another workspace's normal-priority jobs) and, in scaling-folds, the low-priority D=10/30
  launch (all 8 evicted at 01:37 PT at 5-8k steps, before the first checkpoint). Resubmitted at
  normal priority, preemptible (unallocated tier, 64 slots), 1 GPU per task: D=10/30 and D=300 on
  either H200 cluster; 2-way on on-prem only (needs its 378 GiB bundle).
- `d10-d30-cedar` (2026-11-13 01:59 PT, group `scaling-folds@20261113-015941`): the D=10/30 runs on
  Cedar A100 in parallel with the queued Beaker `d10-d30-t2` (Han: keep whichever finishes first;
  cancel the Beaker twin if Cedar finishes before it starts). Only session 0 (D=10, pool fold 1)
  started: other sessions took 14 of the node's 16 GPUs and 89 of 96 CPUs in the same minute, and
  session 1 failed (NOT_PROGRESSED, "Insufficient cpu"); the launcher stopped. The D=300 Cedar
  launch was not attempted (needs 220 GiB; 115 GiB free).
- `d10-fold2-cedar` (2026-11-13 04:59 PT, group `scaling-folds@20261113-045923`): D=10 pool fold 2 on the one free Cedar A100
  (one-point sweep `sweep_d10_d30_p1.yaml`, 5 CPU / 48Gi), parallel to its queued Beaker twin.
  Launched from dispatcher `d1f4870` (Cedar session-log fix, #223).
- 2026-11-13 ~07:00-07:50 PT: all 13 tier-2 tasks (`d300-t2` ×4, `test-d300-t2`, `d10-d30-t2` ×8)
  started on Beaker. 08:00 PT: stopped the Beaker D=10 pool fold 1 and 2 tasks of `d10-d30-t2`
  (jobs `01M7A597NNY8GKVC2FPHAZ5W4M`, `01M7A597SS59V9KSXEK17EJS3J`, at ~1.2k and ~0.4k steps):
  their Cedar twins (`d10-d30-cedar` session 0, `d10-fold2-cedar`) were at 48k and 22k steps.
  The D=10 folds 1-2 points come from Cedar; the other 6 D=10/30 points from Beaker.
- 2026-11-13 14:50 PT: stopped Cedar session `scaling-folds-20261113-015941-0` (D=10 pool fold 1). Its
  run finished at 12:20 PT (early stop at 80k; `gru-output` artifact committed), but a Cedar
  session does not end when its run ends: it stayed `ready` and held its A100 for 2.5 h. Finished
  Cedar sessions are stopped by hand.
