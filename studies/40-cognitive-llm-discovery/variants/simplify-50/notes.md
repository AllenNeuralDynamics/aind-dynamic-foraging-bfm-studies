# Variant `simplify-50`

**What.** One simplification run (ADR 0005; MEvolve ADR 0006, `"mode": "simplify"`): Claude rewrites
the search's best programs to be simpler, and a program counts only if its validation score stays at
or above the **50% floor**: 50% of the gain from Miller-RHG to the chosen search-best. Among those, the
search minimizes Halstead effort. The other two floors are `simplify-90` and `simplify-75`; together
the three trace fit against complexity.

**Why.** The search's best programs carry a 12th parameter whose add-on barely helps (r2). The floors
show how much of the gain survives when the program is made as plain as possible, and which mechanisms
are needed to keep it.

**Pre-registered (2026-11-08, before the search's best was scored on the test subjects; ADR 0008, its
floors amendment).**

| | |
|---|---|
| floor | `run.json` `floor` (×100), from `analysis/simplify/floors.json`; the driver refuses a mismatch |
| seeds | the selection candidates whose rescored validation score reaches the floor, the chosen search-best on island 0 (`seed_programs`, from `floors.json`) |
| score | correct if the validation score (48 validation subjects, 2-level EM fit of the 48 generation subjects, `max_em` 20, 2,048 draws, per-subject fallback) ≥ floor; then minimize Halstead effort (radon 6.0.1, docstrings dropped) |
| proposer | Claude Opus 5.5 at effort medium; no library in the prompt; the search's program limits (12 parameters, 6 latent variables) |
| budget | $40 at list price |
| stopping rule | 300 proposals without a correct program of strictly lower effort, or the budget |
| afterwards | the five correct programs of lowest effort go to the selection subjects; the floor program is the lowest-effort one that passes the fingerprint gate and reaches the selection floor; it is scored once on the test subjects (ADR 0008) |

**Run.** `simplify.sbatch`, as the search's driver (4 CPUs; evaluations as their own 24-CPU jobs). The
three runs share a seed cache; `simplify-90` starts first so the others read its seed scores.

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group `simplify-50@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Status.** Submitted 2026-11-08 23:04 PT as `simplify-50-1`, W&B group [`simplify-50@20261108-230437`](https://wandb.ai/NeuralDynamics/dfbfm-mevolve) (driver Slurm 27716605), studies `4d34ea8`, MEvolve `4465615`. It waits for the seed-cache job (Slurm 27716334, the five seeds scored once on 24 CPUs with the run's own cache keys) and for a rehearsal of `simplify-90` (Slurm 27716602: 6 real proposals, new calls stop at $2).
