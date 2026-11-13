# Variant `test-fold-ref`

**What.** The test fold: the freeze's 149 test subjects. No training. Rescores study 02's
`nxd-3way` H256 model (W&B `AIND-disRNN/mice_ignore_scaling`, group `nxd-3way@20260707-075401`;
seed 0 `-80a6d102`, seeds 1–2 `-3c79d1d1`, `-a3171d8e`)
in passes A and B with the freeze's fit/score session lists. Seed 0 gives the map; seeds 1–2
measure training noise.

**Why.** Model 0 of the rotation already exists. Pass A must reproduce study 02's held-out
per-mouse scores on the 146 mice whose sessions match (PLAN.md, check 1).

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group `test-fold-ref@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Status.** Not launched; waits for the wrapper prerequisite (PLAN.md).
