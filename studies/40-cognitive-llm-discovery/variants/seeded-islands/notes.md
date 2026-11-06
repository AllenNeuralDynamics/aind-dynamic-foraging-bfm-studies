# Variant `seeded-islands`

**What.** One search with a different seed program on each island (ADR 0006): a near-empty template
plus literature models from the seed library, chosen by a rule fixed in advance. The whole library
appears in the prompt pool as inspirations. Budget: the pilot's $450 search budget.

**Stops** when the budget runs out, or when 2,000 proposals pass without the best validation score
rising by more than one standard error of a paired difference between two programs (defined exactly
under Pre-registered).

**Why.** The study's goal is the best interpretable model rather than a comparison of starting
points, and seeding from strong models from the literature gave CogFunSearch its best programs. Islands keep
the seeds' lineages apart long enough to develop, and migration lets their mechanisms combine.

**Deviations, on purpose.**

- *Program limits:* at most 12 parameters and 6 latent variables, not ADR 0005's first 8 and 6 (its
  amendment of 2026-11-05, after `seeded-islands-cap12`).
- *Proposer:* Claude Opus 5.5, not Sonnet 5.5 as in gates 3 and 4, the smoke and cap12 (Han, 2026-11-06,
  before any proposal of this search): compute, not the Claude budget, limits this search, so the
  stronger model is worth twice the price per token. Effort stays medium. The rehearsal checks the model's
  calls and prices before the search starts.

**Pre-registered (2026-11-06, before launch).**

| | |
|---|---|
| proposer | Claude Opus 5.5 at effort medium; ShinkaEvolve's edits (diffs, full rewrites, crossovers), one repair attempt per edit that does not apply |
| islands | 5: the template on island 0; Miller-RHG, Lopez-DoubleTrace, Lebedeva-PR, Eckstein-BI on islands 1 to 4 (`analysis/seed_choice.json`, ADR 0006); migration every 10 generations, rate 0.1 |
| prompt | the task, the 19 literature models as a library, the limits |
| programs | at most 12 parameters and 6 latent variables (ADR 0005 as amended); static checks; evaluation isolated (no file or network access for a candidate) |
| score | validation score: 2-level EM fit of the 48 generation mice, pooled held-out likelihood ×100 on the 48 validation mice (`reference/cohorts_seed0.json`), `max_em` 20, 2,048 draws |
| budget | $450 at list price (the ledger of Claude calls): no call starts once it is spent; calls already running finish |
| stopping rule | the anchor starts at the best score of generation 0 (the seeds and the template). Proposals are counted in the order they finish, every one once: scored, repeated, rejected by the checks, or a failed edit. A proposal scoring strictly above anchor + SE becomes the anchor and resets the count; small rises count once their total passes SE. The search stops when the count reaches 2,000: no new proposals start, the running ones finish and are kept. SE = **0.03** (×100), the low end of the paired standard errors between two programs on these mice (0.03 to 0.06, gate 2) |
| not used | the novelty judge; code embeddings are only recorded (MEvolve #56) |

What it reports (r3): the best validation score against proposals and dollars, by island and lineage;
the stop and its reason; the best programs. Its programs then enter step 7 (simplification at the three
floors, selection on the selection mice, test scores, ADR 0007); none is judged on validation alone.

**Run.** `search.sbatch` is a small driver job on `aind` (4 CPUs, 16 GB, seven days): the search, the
embedding server and Claude. Each evaluation is its own Slurm job of 24 CPUs (12 workers, one per core;
`eval_launch: "job"`, MEvolve #70), so the search uses whatever CPUs free up instead of waiting for one
large block: at most 48 evaluations and 24 proposals at a time. Time in the queue does not count against
an evaluation's 2 hours. The database is snapshotted to the launch directory every 10 minutes and at the
time limit; a crashed or timed-out run resumes from it as an intervention in the same W&B group
(`RESUME_OF`). Expected: an evaluation holds its 24 CPUs for about 13 minutes and uses about half of them
(`seeded-islands-cap12`: 80,319 s over 101 evaluations), so 2,000 proposals hold about 10,700 CPU-hours
in Slurm's accounting (one block of slots held 13,500), about 9 hours with 48 evaluations running and
no queueing. Opus is twice Sonnet's price per token: 12 of cap12's real prompts cost $0.09 to
$0.13 each with no cached prompt (2026-11-06, none refused), so with the cache about $0.05 to $0.10 per
proposal: $100 to $200 for 2,000 proposals, and the whole budget allows 4,500 to 9,000. A rehearsal of this script (`REHEARSAL=1`: 6 real Opus proposals,
about $1, on real mice) runs first, and the search starts only if it passes.

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group `seeded-islands@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Status.** `seeded-islands-1` (Sonnet 5.5, 32 slots of 24 CPUs; studies `549382d`, MEvolve `bbc9ac2`) was
submitted 2026-11-06 03:24 PT and its rehearsal passed, but it never started: the partition had no block
of 768 CPUs before Saturday. It was held, then cancelled, for the changes above; no proposal ran. Next launch:
`seeded-islands-2`.
