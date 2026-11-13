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

**Status.** Not launched.
