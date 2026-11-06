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
- *Where the standard error comes from:* the PLAN fixed it at gate 2, which measured it between
  literature models that differ a lot. The rule compares a new best with an earlier best of the same
  search, usually a close relative, so the error is measured on such pairs instead: every step of the
  best score in `seeded-islands-smoke` and `seeded-islands-cap12`, re-evaluated with each validation
  mouse's likelihood (`analysis/stop_se.json`, `make stop-se`).

**Pre-registered (2026-11-06, before launch).**

For scale (`analysis/stop_se.json`): in both earlier runs the whole gain from the best seed to the best program was about five standard errors (0.082 ± 0.016 and 0.092 ± 0.016), while 9 of the 13 single steps were smaller than the SE below.

| | |
|---|---|
| proposer | Claude Sonnet 5.5 at effort medium; ShinkaEvolve's edits (diffs, full rewrites, crossovers), one repair attempt per edit that does not apply |
| islands | 5: the template on island 0; Miller-RHG, Lopez-DoubleTrace, Lebedeva-PR, Eckstein-BI on islands 1 to 4 (`analysis/seed_choice.json`, ADR 0006); migration every 10 generations, rate 0.1 |
| prompt | the task, the 19 literature models as a library, the limits |
| programs | at most 12 parameters and 6 latent variables (ADR 0005 as amended); static checks; evaluation isolated (no file or network access for a candidate) |
| score | validation score: 2-level EM fit of the 48 generation mice, pooled held-out likelihood ×100 on the 48 validation mice (`reference/cohorts_seed0.json`), `max_em` 20, 2,048 draws |
| budget | $450 at list price (the ledger of Claude calls) |
| stopping rule | the anchor starts at the best score of generation 0 (the seeds and the template). Proposals are counted in the order they finish, every one once: scored, repeated, rejected by the checks, or a failed edit. A proposal scoring strictly above anchor + SE becomes the anchor and resets the count; small rises count once their total passes SE. The search stops when the count reaches 2,000: no new proposals start, the running ones finish and are kept. SE = **0.0114** (×100), the median of the paired standard errors of the 13 steps measured (0.0052 to 0.0203; gate 2's literature pairs gave 0.03 to 0.06) |
| not used | the novelty judge; code embeddings are only recorded (MEvolve #56) |

What it reports (r3): the best validation score against proposals and dollars, by island and lineage;
the stop and its reason; the best programs. Its programs then enter step 7 (simplification at the three
floors, selection on the selection mice, test scores, ADR 0007); none is judged on validation alone.

**Run.** `search.sbatch` on `aind` CPU nodes: slots of 24 CPUs (12 workers per evaluation), their number
chosen at launch from the free-CPU landscape so the job ends earliest; as many evaluations and half as
many proposals at a time. Seven days of Slurm time. The database is snapshotted to the launch directory
every 10 minutes and at the time limit; a crashed or timed-out run resumes from it as an intervention in
the same W&B group (`RESUME_OF`). Expected, from `seeded-islands-cap12` (406 CPU-minutes and $0.025 per
proposal): 2,000 proposals take about 13,500 CPU-hours (34 hours on 16 slots, 17 on 32) and $50 to $70;
the whole budget would allow 13,000 to 18,000 proposals (5 to 13 days). A $0 rehearsal of this script
(`FAKE=1`, 6 canned proposals on real mice) runs first, and the search starts only if it passes.

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group `seeded-islands@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Status.** Not launched.
