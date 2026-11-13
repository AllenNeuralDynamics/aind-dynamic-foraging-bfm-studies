# Variant `pool-folds-2way`

**What.** The 4 `pool-folds` fold models trained 2-way (`data.ignore_policy=exclude`: ignored
trials are dropped from the input sequence and the loss), H256, seed 0, same training-subject
lists (`reference/folds.json`). Only `ignore_policy` differs from `pool-folds`, so each fold
gives a paired 2-way vs 3-way comparison.

**Why.** MEvolve's programs, and other models that skip ignored trials, see a sequence without
them; the 3-way GRU keeps ignored trials in its history. A 2-way GRU is the like-for-like
comparator for those models (Han, 2026-11-13). Full size only, no scaling curve. The test fold's
2-way model is reused: study 01 `nxd-grid` H256 at `subject_ratio` 1.0 (2-way, W&B
`AIND-disRNN/mice_data_scaling`, group `nxd-grid@20260623-092311`), whose held-out set is the
GRU's own 149 (146 shared with the freeze's test subjects; PLAN.md, "Mouse sets").

**Launches.** One launch, 4 tasks, one W&B group `pool-folds-2way@<launch_id>` (docs/adr/0001).
Low priority, preemptible on on-prem H200, one 378 GiB bundle each.

**Status.**
- `folds-h256-2way` (2026-11-13 01:13 PT, group `pool-folds-2way@20261113-011355`, Beaker experiment
  [`01M7A2YWB6S8T0Q8RWER74DJ6V`](https://beaker.org/ex/01M7A2YWB6S8T0Q8RWER74DJ6V)): 4 tasks, low
  priority, preemptible, on-prem H200, W&B key `han-wandb-api-key-neuraldynamics`. **Stopped at 01:33 PT, no training done:** on the oversubscribed on-prem
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
- `folds-h256-2way-hpc` (2026-11-13 08:07 PT, group `pool-folds-2way@20261113-080715`, W&B sweep
  [`x17uytie`](https://wandb.ai/NeuralDynamics/dfbfm_17-gru-crossfit-predictability/sweeps/x17uytie),
  SLURM array 27820674): the 4 tasks moved to HPC `aind` H200 (1 GPU, 16 CPU, 480G RAM, 5-day
  limit), because the tier-2 Beaker launch `folds-h256-2way-t2` was still queued after 6 h (the
  other tier-2 jobs started around 07:00-07:50 PT; these need the on-prem 378 GiB bundle). Same
  wrapper (`5affee5`), dispatcher (`7c1d686`) and foraging-models (`392c504`) as the Beaker and
  Cedar runs; launched from detached worktrees in `~/code/wt/17-hpc`. Runtime is the shared conda
  env `dynamic-foraging-bfm-gpu` (jax 0.6.1 as in the Beaker image; numpy 2.4.6, hydra-core
  1.3.2, dm-haiku 0.0.16 vs 2.5.x / 1.3.6 / 0.0.17 there); foraging-models `392c504` was installed
  into it for this launch (was `56db272`; the GRU path does not import it). W&B key: the
  NeuralDynamics key exported as `WANDB_API_KEY` (the `~/.netrc` key cannot write to
  NeuralDynamics); outputs and W&B files on scratch. The Beaker `folds-h256-2way-t2` entry is
  stopped once the HPC tasks start (whichever starts first is kept); its group then has no runs.
- `folds-h256-2way-t1` (2026-11-13 11:25 PT, group `pool-folds-2way@20261113-112505`, Beaker experiment
  [`01M7B5XY7ABMVPE8J3AYTYRHYM`](https://beaker.org/ex/01M7B5XY7ABMVPE8J3AYTYRHYM)): the 4 tasks on
  the allocated seats (normal, not preemptible, on-prem H200, 1 GPU / 28 CPU / 370GiB), after D=100
  folds 1-3 early-stopped at 90k (10:30-11:05 PT) and freed 3 of the 8 seats. Template
  `experiment_beaker_t1.yaml` pins WRAPPER/DISPATCHER/FORAGING_MODELS_REF to `5affee5` / `7c1d686` /
  `392c504`, the SHAs of every other study-17 run (the template's `main` refs had moved). Third
  copy of the same 4 runs, with `folds-h256-2way-t2` (Beaker tier 2) and `folds-h256-2way-hpc`;
  the first copy to start is kept and the others are cancelled.
- `folds-h256-2way-cedar-h200` (2026-11-13 14:51 PT, group `pool-folds-2way@20261113-145140`): the 4
  tasks on the Cedar H200 node (`session_cedar_h200.yaml`: 1 GPU, 16 CPU, 600Gi, refs pinned).
  **Did not start:** the node's 8 GPUs were all allocated to other sessions ("largest free slot on
  one node is 0.687 vCPUs, 38.4 GiB, and 0 GPUs"), although `cluster nodes list` showed only 9% of
  its memory and 23% GPU use. The launcher stopped after the first session; nothing ran.
- `folds-h256-2way-cedar-a100` (2026-11-13 14:54 PT, group `pool-folds-2way@20261113-145449`): the
  Cedar A100 node had emptied (2 of 96 CPUs, 14 GiB used), so the 4 tasks went there
  (`session_cedar_a100.yaml`: 1 GPU, 16 CPU, 400Gi, refs pinned). **Folds 1-3 started** (sessions
  `-0`, `-1`, `-2`); fold 4's session `-3` did not (NOT_PROGRESSED: node full at 3 x 400Gi).
  400Gi vs this morning's OOM at 320Gi on this node: the 3-way fold models pass every checkpoint at
  370GiB on Beaker (peak ~350 GB).
- 14:57 PT, for folds 1-3 kept on Cedar: cancelled the HPC array 27820674 (its W&B sweep agents take
  grid points in order, so a surviving task would rerun fold 1; the dependent job 27820675 then
  stops sweep `x17uytie`) and stopped the Beaker tier-2 tasks `-000`..`-002` of
  `folds-h256-2way-t2`. The allocated-seat copies of folds 1-3 (`folds-h256-2way-t1` `-000`..`-002`)
  stay queued as a backup until the Cedar runs pass their first checkpoint (10k, the RAM peak),
  then are stopped. Fold 4 waits on Beaker (`-t1` `-003` and `-t2` `-003`).
- 15:41 PT: fold 4 started on Beaker tier 2 (`folds-h256-2way-t2` task `-003`, group
  `pool-folds-2way@20261113-015357`); its allocated-seat copy (`folds-h256-2way-t1` `-003`) was
  stopped. The 4 2-way fold models therefore come from two launches: folds 1-3 Cedar A100
  (`@20261113-145449`), fold 4 Beaker on-prem H200 (`@20261113-015357`); same sweep, same pinned
  wrapper/dispatcher/foraging-models SHAs, different runtime image (Cedar `dfbfm-wrapper:20261109-wf390d4f`
  vs Beaker `dynamic-foraging-bfm-wrapper-main-20260902`).
- **15:43 PT: the 3 Cedar A100 runs (folds 1-3, `@20261113-145449`) failed** at steps 10-80 with a
  GPU out-of-memory error (`RESOURCE_EXHAUSTED: Out of memory while trying to allocate 33236989808
  bytes`; A100 has 40 GB). The full-size H256 recipe does not fit an A100's GPU memory; the
  300-step `cedar-smoke` run never reached the longest length buckets. Sessions stopped 15:47 PT;
  the W&B runs are marked failed. Full-size models need H200s (141 GB).
- 15:50 PT: folds 1-3 resubmitted at tier 2 with pinned refs (`experiment_beaker_t2.yaml`,
  launch `folds-h256-2way-t2b`, group `pool-folds-2way@20261113-154445`, Beaker experiment
  [`01M7BMSD856ZR57Z9W5CG1Z503`](https://beaker.org/ex/01M7BMSD856ZR57Z9W5CG1Z503); its fold-4 task (`-003`) was stopped at once, since fold 4 runs in
  `folds-h256-2way-t2`). The allocated-seat copies (`folds-h256-2way-t1` `-000`..`-002`) stay
  queued; the first copy of each fold to start is kept.
