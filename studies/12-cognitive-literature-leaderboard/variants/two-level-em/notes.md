# Variant `two-level-em`

**What.** The 19 ranked literature models as 2-level EM (regime 2): each session's parameters are
drawn from its subject's distribution (session → subject, no population level), fit by Laplace-EM
under study 13's hyperprior, and held-out sessions are scored by the session marginal. Run with
MEvolve's `mevolve.score` CLI at its reported settings (MEvolve ADR 0004: EM to convergence,
max_em = 100; 2,048 draws; seed 0) on all 763 subjects, from a clean MEvolve checkout at a pinned
full SHA and MEvolve's data freeze of snapshot 20260603. Job script: [`run_2level.sbatch`](run_2level.sbatch).
Models are labelled EM-<model>.

**Why.** Report r3: how the literature models rank with session-level variation within each
subject, and how much of the gap to 2-level HB (study 13) that closes. Gate 1c checked this
evaluator against study 13's NUTS fits for CompareToThreshold and Hattori2019.

**Launches.** Not dispatcher launches: no W&B sweep. Each launch is one `sbatch` of
`run_2level.sbatch`, recorded under `launch_record/` with its MEvolve SHA, data-freeze manifest
digest and SLURM job id. The group name `two-level-em@<launch_id>` identifies it (no W&B runs).

**Caveat.** Rescorla-Wagner's epsilon-greedy likelihood is flat or stepped in some parameters, and
the per-session Laplace step uses gradients, so EM-Rescorla-Wagner can report "converged" while
poorly fitted (MEvolve Phase 1). Gate 1c covered only CompareToThreshold and Hattori2019.

**Status.** First launch: Hattori2019 alone on all 763 subjects, to measure cost.
