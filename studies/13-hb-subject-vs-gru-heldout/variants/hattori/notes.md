# Variant `hattori`

**What.** HB-Hattori2019 as a two-level HB (`hattori2019_two_level` in foraging-models): subject-level mean and between-session spread over all 5 session-level parameters, under the fixed hyperprior, conditioned on each held-out subject's context sessions.

**Why.** Compares against the per-subject MLE Hattori (`cross_study.rl_baseline("hattori")`), the same forager with one parameter set per subject. The gap is the overall HB improvement (session variability, prior, and posterior averaging together), not session variability alone.

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group `hattori@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Route.** One held-out subject per SLURM CPU array task (`sweep_hpc_per_subject.yaml`), pooled
across runs via `heldout/eval_log_lik_sum`. A single batched sampler over all subjects is set by
the hardest subject's step size: the batched smoke took 2.3 h for 300 iterations, while one subject
per sampler on CPU runs ~100× faster per iteration (`launch_record/probe/README.md`).

**Launches so far** (all 2026-10-25 PT; records in `launch_record/`):

| record | group | route | outcome |
|---|---|---|---|
| `smoke/` | `hattori@20261025-203046` | Beaker, batched | never scheduled (hub clusters full); stopped |
| `smoke-hpc/` | `hattori@20261025-210309` | HPC A100, batched | failed at Hydra composition: `model.progress_bar` needed `+` |
| `smoke2/` | `hattori@20261025-210647` | Beaker, batched | never scheduled; stopped once the route changed |
| `smoke2-hpc/` | `hattori@20261025-210635` | HPC A100, batched | completed in 2.3 h (200 warmup, 100 draws): all 149 subjects and 1,013,172 scored trials (the comparators' count), no `test_likelihood`; 0/149 pass the gates at 100 draws (expected); **36% of sessions fail Pareto k̂ < 0.7** at 6,400 draws |
| `probe/` | none (diagnostic) | HPC A100 + CPU | route decision above |
| `smoke-per-subject/` | `hattori@20261025-235239` | HPC CPU, one subject per task | production settings on 3 subjects |

**Open question for scoring.** The k̂ failures on real data mean plain draw-averaging may not
suffice (ADR 0002 fallback: importance sampling). Fitting is unaffected: each run saves its
posterior draws, so sessions can be rescored offline without refitting.
