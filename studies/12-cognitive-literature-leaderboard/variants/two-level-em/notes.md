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

## Launch facts

Written as v2 launch records after launch (`launch_record.write_initial(...,
recorded_after_launch=True)`, studies #89/#90): [`launch_record/cost-hattori/`](launch_record/cost-hattori/launch_record.json)
and [`launch_record/rest-18/`](launch_record/rest-18/launch_record.json). Each holds the studies and
MEvolve SHAs, the data freeze's manifest digest, the mevolve env's package digest, the SLURM job and
the sbatch line from SLURM's accounting, plus `run_2level.sbatch` as committed at that launch's
studies SHA. The table below is what the records were written from.

| | cost run | remaining 18 models |
|---|---|---|
| label | `cost-hattori` | `rest-18` |
| note | Study 12 r3 cost run: Hattori2019 as 2-level EM on all 763 subjects with MEvolve's reported settings, to measure wall time before the other 18 ranked models. | The other 18 ranked literature models as 2-level EM on all 763 subjects, same settings, one model per array task, after the cost run took 12.3 min. |
| W&B group (identity; no W&B runs) | `two-level-em@20261102-003154` | `two-level-em@20261102-004644` |
| SLURM job | 27567407 (one exclusive 6238R node, 56 workers), submitted 2026-11-02 00:31 PST | array 27567420, tasks 0–17, 9 at a time (same node shape), submitted 2026-11-02 00:46 PST |
| MEvolve | `6af7d58a0a2dadb4d4eec7ab72108c6350e8dfbe` (clean detached worktree) | same |
| studies | `b1816eee09b59316c8cd2b9626c6cfb40095c31a` | `72211037b41638c0a4edf153d8e96efa352247bb` |
| data freeze | `/allen/aind/scratch/han.hou/mevolve/freeze/20260603`, manifest sha256 `cfea3d1e21f49ed7baa42f11e1fdda66fd5901c519a47773c08716e062e7eef0` | same |
| env | `/allen/aind/scratch/han.hou/miniforge3/envs/mevolve` | same |
| output | `/allen/aind/scratch/han.hou/outputs/study12/two-level-em/20261102-003154/` | `/allen/aind/scratch/han.hou/outputs/study12/two-level-em/20261102-004644/` |
| command | `sbatch --export=ALL,MEVOLVE_ROOT=<worktree>,OUT=<output>,MODELS="Hattori2019" -J s12-2level -o <output>/slurm-%j.out run_2level.sbatch` → `python -m mevolve.score --model Hattori2019 --pool all --freeze <freeze> --out <output> --workers 56` (defaults max_em 100, n_draws 2048, seed 0) | `sbatch --array=0-17%9 ... MODELS="<the 18, in array order>" -o <output>/slurm-%A_%a.out run_2level.sbatch`; array order: Bari2019, Rescorla-Wagner, LossCounting, CompareToThreshold, ActorCritic, Chen-RLCK, Zid-HistoryKernel2-ForagingRL, Grossman-MetaLearning, Lebedeva-PR, Beron-RFLR, Miller-RHG, Eckstein-RL, Eckstein-BI, FeedbackDependent-RL, FeedbackDependent-RL+Bias, FeedbackDependent-RL+Bias+CK1, Alsio-DualRateSticky-RL, Lopez-DoubleTrace |

**Status.** Cost run done 2026-11-02 00:45 PST (12.3 min; 37 of 763 subjects at the EM cap; held-out pooled 0.72110, as MEvolve's gate check). Remaining 18 done 2026-11-02 03:55 PST (EM-Rescorla-Wagner last, 3.1 h, 733 of 763 subjects at the EM cap). Frozen in `analysis/two_level_em_fits.csv`; report r3.
