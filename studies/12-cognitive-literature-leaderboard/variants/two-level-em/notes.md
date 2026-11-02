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

## Launch facts (until the MEvolve launch-record writer lands)

The studies launch-record v2 schema cannot yet name a MEvolve SHA; the MEvolve PM session is adding
an optional `mevolve` revision and `write_initial()` (study 40 PLAN build step 5). These facts are
kept here and will be written as v2 records, marked as written after launch.

| | cost run |
|---|---|
| label | `cost-hattori` |
| note | Study 12 r3 cost run: Hattori2019 as 2-level EM on all 763 subjects with MEvolve's reported settings, to measure wall time before the other 18 ranked models. |
| W&B group (identity; no W&B runs) | `two-level-em@20261102-003154` |
| SLURM job | 27567407 (one exclusive 6238R node, 56 workers), submitted 2026-11-02 00:31 PST |
| MEvolve | `6af7d58a0a2dadb4d4eec7ab72108c6350e8dfbe` (clean detached worktree) |
| studies | `b1816eee09b59316c8cd2b9626c6cfb40095c31a` |
| data freeze | `/allen/aind/scratch/han.hou/mevolve/freeze/20260603`, manifest sha256 `cfea3d1e21f49ed7baa42f11e1fdda66fd5901c519a47773c08716e062e7eef0` |
| env | `/allen/aind/scratch/han.hou/miniforge3/envs/mevolve` |
| output | `/allen/aind/scratch/han.hou/outputs/study12/two-level-em/20261102-003154/` |
| command | `sbatch --export=ALL,MEVOLVE_ROOT=<worktree>,OUT=<output>,MODELS="Hattori2019" -J s12-2level -o <output>/slurm-%j.out run_2level.sbatch` → `python -m mevolve.score --model Hattori2019 --pool all --freeze <freeze> --out <output> --workers 56` (defaults max_em 100, n_draws 2048, seed 0) |

**Status.** Cost run (Hattori2019, all 763 subjects) running since 2026-11-02 00:32 PST.
