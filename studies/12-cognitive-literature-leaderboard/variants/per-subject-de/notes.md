# Variant `per-subject-de`

**What.** One W&B grid sweep over 19 literature models (`model=baseline_rl_lit_<name>`, one dispatcher
config per model, [dispatcher PR 214](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/pull/214)), one run per model. Each run fits all 763 subjects (614 training pool + 149
held-out) with differential evolution, one parameter set per subject, on even sessions, and
scores odd sessions; study 01 r8's protocol and data config. Sweep: [`sweep_hpc.yaml`](sweep_hpc.yaml).

**Why.** The leaderboard of literature models on our subjects, for study 40's seed choice and
MEvolve's parity reference. The re-fit Hattori2019, Bari2019 and CompareToThreshold must
reproduce study 01 r8's per-subject values.

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group `per-subject-de@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

| launch | W&B group | what happened |
|---|---|---|
| [smoke-1](launch_record/smoke-1/results.md) | `per-subject-de@20261101-004845` | failed in Hydra composition (launcher bug, fixed in dispatcher 6450af3) |
| [smoke-2](launch_record/smoke-2/) | `per-subject-de@20261101-005459` | 2 models, 10 training + 149 held-out subjects; whole path works |
| [production](launch_record/production/) | [`per-subject-de@20261101-012148`](https://wandb.ai/NeuralDynamics/aind-dynamic-foraging-model/sweeps/3zwvpcce) | 19 models x 763 subjects, SLURM array 27565016 (10 at a time, one 112-CPU node each), submitted 2026-11-01 01:22 PT |
| [resubmit-hattori](launch_record/resubmit-hattori/resubmit_hattori.json) | same group | Hattori2019 alone, `--mem=500G`, after its task hit the 400G limit (OUT_OF_MEMORY); W&B sweep r2a15ad9, SLURM 27565201, 2026-11-01 07:11 PST |
| [loss-counting](launch_record/loss-counting/) | [`per-subject-de@20261101-071303`](https://wandb.ai/NeuralDynamics/aind-dynamic-foraging-model/sweeps/m7olpz1o) | the free LossCounting model ([`sweep_hpc_loss_counting.yaml`](sweep_hpc_loss_counting.yaml)), same env as production; SLURM 27565202, 2026-11-01 07:13 PST |

Launch from detached worktrees at studies `fab284d`, dispatcher `6450af3`, wrapper `331d373`;
foraging-models is the env's PyPI 0.26.0 = `0c95b4b` (byte-checked; recorded by a backfill
beside each launch record, as study 13 did). The NeuralDynamics W&B key is exported as
`WANDB_API_KEY` in the launcher's shell, never written to a file.

**Status.** Production running.
