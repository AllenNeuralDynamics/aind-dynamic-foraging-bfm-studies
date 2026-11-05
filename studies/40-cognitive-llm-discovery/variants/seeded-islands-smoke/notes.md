# Variant `seeded-islands-smoke`

**What.** The `seeded-islands` configuration (ADR 0006) at 100 proposals, on real mice. It is PLAN
step 5's 100-proposal smoke launch (issue #115).

| | setting |
|---|---|
| mice | the seed-0 cohorts (`reference/cohorts_seed0.json`, ADR 0003): 48 generation mice fitted, 48 validation mice scored; the selection and test mice are not touched |
| data | the MEvolve freeze `20260603` |
| islands | 5: the template on island 0, then one seed each: Miller-RHG, Lopez-DoubleTrace, Lebedeva-PR, Eckstein-BI |
| seeds | by ADR 0006's rule, `analysis/seed_choice.json` (`make seed-choice`): the three literature models with the best pooled 2-level EM score from study 12 r3 on the 48 validation mice (72.614, 72.576, 72.573), then Eckstein-BI as the model with discrete states. The job checks that `run.json` lists the same seeds |
| prompt | the whole literature library as inspirations (`library_in_prompt`) |
| model | Claude Sonnet 5.5 at effort medium (gate 3), one repair attempt for an edit that does not apply |
| budget | 100 proposals, $20 list (the PLAN's line for this launch) |
| engine | migration every 10 generations at rate 0.1; repeats reuse their score (MEvolve #52); code embeddings are computed and only observed (MEvolve #56) |
| node | one full `aind` node: 2 evaluations at a time with 28 workers each, one per core; 12 hours |

**Why.** Every node-time number for the main search so far comes from simulated mice (r1, lesson 7:
node time, not dollars, limits the main search). This run should teach us:

1. the node-minutes and dollars per proposal on real mice, which set the compute plan of
   `seeded-islands`;
2. whether the production configuration runs end to end on real mice: five islands, the library in
   the prompt, the repeat cache, the observed embeddings and the archive;
3. a first look at the search curve: how many proposals beat the best seed, and on which islands.

**Pre-registered (2026-11-05, before launch).**

1. **Pass:** the search finishes its 100 proposals within $20 and 12 hours, with no API errors, and
   the run's archive artifact is logged. There is no score target: this is a smoke run.
2. **Reported, not judged:** dollars per proposal (from the ledger); node-minutes per proposal
   (wall clock of the search over proposals); evaluation seconds per program; proposals by outcome
   (scored, failed the static checks, failed to fit, edit not applied, repeat); the best validation
   score against the best seed's, by island; programs whose 2-level fit failed.
3. **Not carried over:** the programs found here do not enter `seeded-islands`. Only the seed scores
   are reused, through the shared `seed_cache`, whose key covers the code, data, cohorts, evaluator
   source and settings.
4. The search scores on the validation mice it optimises, so its best score is biased upward; no
   claim about a program's quality rests on this run.

**Run.** `search.sbatch` with `DRY=1` first: a $0 job that starts the embedding server, scores the
seeds into the shared cache and builds the runner, so a bad path fails before any Claude call. Then
the search itself.

**Data and archive.** The run directory is
`/allen/aind/scratch/han.hou/mevolve/runs/seeded-islands-smoke/<launch_id>/search`. At the end of
the run MEvolve logs it as the W&B artifact
`NeuralDynamics/dfbfm-mevolve/run-seeded-islands-smoke-<launch_id>` (type `mevolve-run`, in the
[dfbfm-mevolve](https://wandb.ai/NeuralDynamics/dfbfm-mevolve) project) and writes `archive.json`
beside the files, with the artifact's digest.

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group `seeded-islands-smoke@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Dry run (2026-11-05, 00:58 PT, $0).** Slurm job 27629696 on n297, from studies a6c3f29 and MEvolve
02b9558: the embedding server started, the seeds were scored on the 48 validation mice and cached
(Miller-RHG 72.608, Lopez-DoubleTrace 72.548, Lebedeva-PR 72.573, Eckstein-BI 71.780; 150 to 339 s
each, one seed at a time with all 56 workers; the search's own evaluations run two at a time with 28
workers each), each seed got its code embedding, and the runner was built. No Claude call.
The cached scores sit within 0.03 of study 12's (`seed_choice.json`), which used its own EM settings.

**Result: pass.** Read by hand from the run directory: `run_summary.json`, `ledger.jsonl`,
`proposals.csv`, Shinka's database and `gen_51/failure.json`. Launched 2026-11-05 at 01:18 PT as
`seeded-islands-smoke-1` (Slurm job 27629852 on n297, W&B group `seeded-islands-smoke@20261105-011815` in
[dfbfm-mevolve](https://wandb.ai/NeuralDynamics/dfbfm-mevolve)); the job completed at 06:57 PT.

| pre-registered | result |
|---|---|
| 1. 100 proposals within $20 and 12 hours, no API errors, archive logged | pass: 100 proposals, $3.43 list, 5 h 37 min of the run; 104 Claude calls (100 + 4 repair attempts), all `end_turn`, none failed; archive `NeuralDynamics/dfbfm-mevolve/run-seeded-islands-smoke-20261105-011815:v0`, digest `19f17a3875d4ea7c7dfe5507bccc6335`, 1,218 files, 20.4 MB |
| dollars per proposal | $0.034 list (median $0.025); a Claude call takes a median 7 s |
| node-minutes per proposal | 3.4 (20,212 s of the run over 100 proposals, one full node) |
| evaluation seconds per program | median 426 s, 10th to 90th percentile 315 to 529 s, longest 1,693 s (90 fresh fits, 28 workers each, two at a time) |
| proposals by outcome | 90 scored by a fresh fit; 4 repeats, scored from the repeat cache; 5 rejected by the static checks (3 syntax errors, 1 with 9 parameters, 1 whose `update` returned the wrong shape); 1 edit that did not apply after its repair attempt (proposal 51) |
| 2-level fits that failed | 10 of the 94 scored proposals kept their per-subject score (`two_level_failed`) |
| best validation score against the best seed | 72.690 (proposal 93) against 72.608 (Miller-RHG), +0.082; 29 of 94 scored proposals beat their parent |

By lineage (the seed a program descends from), best score against the seed's own:

| lineage | proposals | seed | best | gain |
|---|---|---|---|---|
| template | 22 | 70.020 | 72.690 | +2.670 |
| Miller-RHG | 40 | 72.608 | 72.677 | +0.069 |
| Lebedeva-PR | 18 | 72.573 | 72.668 | +0.095 |
| Lopez-DoubleTrace | 9 | 72.548 | 72.591 | +0.043 |
| Eckstein-BI | 10 | 71.780 | 72.606 | +0.826 |

**What it teaches.**

1. **Evaluation, not Claude, limits the search.** Proposals waited for the two evaluation slots;
   Claude answered in seconds. At 3.4 node-minutes per proposal, the stopping rule's minimum of 2,000
   proposals takes about 5 days of one node, so `seeded-islands` needs evaluations on many nodes
   (AllenNeuralDynamics/aind-dynamic-foraging-mevolve#59).
2. **The template's lineage won.** Proposal 3 rebuilt a literature-level model from the template
   (72.658, above every seed). The best program adds one task-specific idea, a per-side value that
   grows while the side is not chosen ("bait"), but its fitted weight is negative for most mice, so it
   acts as stickiness that grows with run length.
3. **The search hits the program limits.** 90 of 94 scored proposals use 7 or 8 parameters, and the
   best program uses all 8 parameters and all 6 latent variables, with its bait rate written in as a
   constant (0.15). Constants like this are tuned by the search on the validation mice, not fitted per
   mouse.
4. **Most steps are within noise.** Proposal 3 beat the best seed by 0.050; after it, each new best adds
   0.0001 to 0.009 to the previous one, below the paired standard error between two programs on these
   mice (0.03 to 0.06, gate 2).

**Status.** Done.
