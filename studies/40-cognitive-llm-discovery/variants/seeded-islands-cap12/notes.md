# Variant `seeded-islands-cap12`

**What.** `seeded-islands-smoke` with looser program limits: at most 12 parameters and 8 latent variables
instead of ADR 0005's 8 and 6 (issue #118; MEvolve's `max_params` and `max_latent`, AllenNeuralDynamics/aind-dynamic-foraging-mevolve#60).
Everything else is the same: 100 proposals by Claude Sonnet 5.5 at effort medium, the same seed-0
cohorts and freeze, the same five islands and seeds (`analysis/seed_choice.json`), the library in the
prompt, $20 list. The evaluations run 16 at a time as Slurm steps of 56 CPUs, the CPUs of 8 nodes placed
wherever 56 are free (AllenNeuralDynamics/aind-dynamic-foraging-mevolve#59); each is the same 28-worker
evaluation as before.

**Deviation, on purpose.** ADR 0005 fixes 8 parameters and 6 latent variables for the study's programs.
This variant is a test of those limits, not a change to them: whether `seeded-islands` searches with
other limits is Han's decision after it (2026-11-05).

**Why.** In `seeded-islands-smoke` the limits bound: 90 of 94 scored proposals used 7 or 8 parameters,
and the best program used all 8 parameters and all 6 latent variables, with its bait rate written in
as a constant (0.15) because no parameter was left. A constant like that is tuned by the search on the
validation mice instead of fitted per mouse. This run should show whether Claude uses the extra room,
and what for. It is also the first search on many nodes, so it measures their throughput.

**Pre-registered (2026-11-05, before launch).**

1. **Pass:** the search finishes its 100 proposals within $20 and 6 hours, with no API errors, every
   evaluation step starts (no proposal fails for a Slurm reason), and the run's archive is logged.
2. **Reported against `seeded-islands-smoke`, not judged:** the share of scored proposals above 8
   parameters or 6 latent variables; the best program's parameter and latent counts, and the numbers
   written into its code; the best validation score and the best by lineage. One run each way, and
   the smoke run's steps were at or below the paired standard error between two programs on these
   mice (0.03 to 0.06, gate 2): a difference in the best score is one sample, not an effect of the
   limits.
3. **Throughput:** proposals per hour and node-minutes per proposal, against the smoke run's 17 per
   hour and 3.4 node-minutes on one node.
4. As in the smoke run, the programs found here do not enter `seeded-islands`; only the seed scores
   in the shared `seed_cache` are reused.

**Run.** `search.sbatch`: 16 tasks of 56 CPUs on `aind` nodes. MEvolve keeps Shinka's database on the first node's
local disk; each evaluation's program and results pass through `stage/` in the launch directory on
/allen. The exit trap copies only files `run.main` did not copy, so it cannot overwrite the archived run
(it did in `seeded-islands-smoke`).

**Data and archive.** `/allen/aind/scratch/han.hou/mevolve/runs/seeded-islands-cap12/<launch_id>/search`,
logged at the end of the run as the W&B artifact
`NeuralDynamics/dfbfm-mevolve/run-seeded-islands-cap12-<launch_id>` (type `mevolve-run`, in the
[dfbfm-mevolve](https://wandb.ai/NeuralDynamics/dfbfm-mevolve) project), with `archive.json` beside it.

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group `seeded-islands-cap12@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Status.** Not launched.
