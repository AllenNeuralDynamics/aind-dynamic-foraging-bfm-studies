# Variant `engine-smoke`

**What.** Gate 3 of the PLAN: the first run of ShinkaEvolve with MEvolve's shim, seeds and
evaluator, as a test of the engine, not of the science. Two runs of 10 proposals each by Claude
Sonnet 5.5, one at effort `medium` and one at `high`, with everything else equal:

| setting | value |
|---|---|
| subjects | 8 generation + 8 validation, `smoke_generation` and `smoke_validation` in `cohorts_smoke_seed0.json`: drawn with seed 0 after the 48/48/100 draft cohorts, so disjoint from them, stratified the same way, without the GRU's three extra test subjects |
| islands | 5: the template on island 0; Miller-RHG, Lopez-DoubleTrace, Lebedeva-PR and Eckstein-BI on islands 1 to 4 (ADR 0006's rule, applied to the draft validation subjects) |
| evaluator | 2-level EM, at most 20 EM iterations, 2,048 draws; 16 workers per evaluation, 3 evaluations at once |
| cost cap | $3.50 per run, enforced from MEvolve's ledger of every API call |
| code | MEvolve at the commit in the launch record (AllenNeuralDynamics/aind-dynamic-foraging-mevolve#29), ShinkaEvolve pinned at 8adc053a |

**Why.** Before the $450 search, check that the engine works end to end and what it costs:

- [x] no API errors; refusals and cut-off replies counted as failed proposals
- [x] prompt caching works (cache reads > 0 after the first call)
- [x] the ledger's cost matches the Anthropic Console within 5%
- [x] each seed on its island, the template on island 0, children on their parent's island
- [x] ShinkaEvolve's web viewer shows the run
- [x] medium vs high effort: valid and improving proposals per dollar

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group `engine-smoke@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Status.** Launched 2026-11-03 03:03 PT as `engine-smoke@20261103-030341` (SLURM 27586076;
record in `launch_record/engine-smoke-1/`). Its W&B group is in `NeuralDynamics/aind-dynamic-foraging-model`
and stays there; the run configs now name `dfbfm-mevolve`, so a new launch logs there (issue #99). Outputs: `/allen/aind/scratch/han.hou/mevolve/runs/engine-smoke/20261103-030341/`
(`medium/` and `high/` each hold the Shinka database, `ledger.jsonl` with every API call and its cost,
`proposals.csv`, `run_report.md`, `run_summary.json`; `comparison.md` sets the two side by side). A
preflight of two calls (effort low) before launch: the key, the model and the request shape work, and
the cache was written once (14,370 tokens) and read on the second call; it cost $0.04.

**Viewer.** SLURM job 27586077 starts ShinkaEvolve's web viewer on n268, port 8765, when it gets
resources, for two days. In VS Code connected to n268, forward port 8765 and open http://localhost:8765. It
has no login and listens on all interfaces of n268 (cluster network only); cancel it with `scancel` when
done. Screenshots of every viewer tab, a summary plot and one page with both
(`figures/smoke_gallery.html`) are in the run directory under `figures/`; they were taken from a copy of
the run, served on localhost for the length of one command.

## Result (2026-11-03)

The measured numbers in this section are in [`analysis/gate3_engine_smoke.json`](../../analysis/gate3_engine_smoke.json)
(`make -C studies/40-cognitive-llm-discovery gate3`), computed from the frozen ledgers, Shinka's program
table and the Console figures in [`analysis/engine_smoke/`](../../analysis/engine_smoke/)
(`make freeze-engine-smoke`).

The engine works end to end with real Claude calls, and gate 3 passes. Ten proposals per run test the
engine, not the science: the best program beats the best seed (Lebedeva-PR, 73.409) by only 0.04.

**Calls.** 22 calls (2 preflight, 10 per run): no API errors, no refusals, no cut-off replies; every
call ended with `end_turn`.

**Ledger against the Console.** Token counts match. The Console figures are from Han's Console for the
workspace, typed in by hand from his screenshots (`analysis/engine_smoke/console.json`): the Cost page
as of 2026-11-03 19:48 UTC, the Caching page as of 11:00 UTC. The Caching page counts in hourly buckets;
the 22 calls ran 11:02–11:24 UTC, and its totals match all of them. The workspace held only these 22
calls.

| | ledger | Console |
|---|---|---|
| cache reads | 258,660 | 259K |
| cache writes | 57,480 (4 × 14,370) | 57.6K (259K ÷ 4.50 reads per write) |
| uncached input | 51,092 | 51K |
| all input | 367,232 | 367K |
| cost at list price | $0.616 | $0.59 |
| cost billed | — | $0.29 |

The ledger's cost is 4% above the Console's list price, inside the 5% test and on the safe side for a
cap. The price table matches Anthropic's list prices for Sonnet 5.5 ($2 and $10 per million input and
output tokens, $2.50 for 5-minute cache writes, $0.20 for cache reads), so the 4% is not yet explained;
the Console's cost download, split by token type, would show which line differs. The bill is about 49%
of list price, after the organisation's discount: the ledger's cap and the PLAN's budget, both at list
price, are about twice what is billed.

**Caching.** The cache prefix (the task and the 19 literature programs, 14,370 tokens) was written 4
times: once in the preflight, and by the first 3 calls of the medium run. Those 3 went out within 0.23 s
of each other, before any of them had written the cache; the preflight's copy, written 532 s earlier, had
expired. Every other call read it, including all 10 calls of the high run, which started 191 s (3
minutes) after medium's last call while its cache was still warm. Within a run, the longest gap between
calls was 126 s (start to start); it and the 191 s are inside the 5-minute cache lifetime. The uncached
input, 14% of all input tokens, is the part that changes with each call (the parent program and its
feedback, about 2,500 tokens). Output is 52% of all cost.

**Islands.** Each literature seed is alone on its island and the template alone on island 0. Shinka's
`island_idx` is where a program is now, after migrations: 7 programs sit on an island other than their
parent's, 6 moved there by migrations recorded in `migration_history` at generations 9 and 10, and 1
(high run, proposal 9) was born on island 0 just before its parent migrated from island 0 to island 4.

**Medium vs high effort.**

| | medium | high |
|---|---|---|
| proposals | 10 | 10 |
| edits that did not apply to the parent | 2 | 0 |
| programs rejected by the static checks | 0 | 1 (syntax error) |
| valid programs scored | 8 | 9 |
| improved on the parent by more than 0.05 | 4 | 2 |
| best validation score | 73.449 | 73.430 |
| output tokens per call | 1,310 | 1,868 |
| cost per proposal, calls that read the cache | $0.020 | $0.027 |
| run cost at list price | $0.306 (pays the 3 cache writes) | $0.270 (no writes) |

High is not better per dollar: it costs 38% more per proposal, for its 43% more output, and improved
fewer parents. By the PLAN's rule (keep high only if clearly better per dollar) the search uses medium.
The four large gains (+0.6 to +2.8) all came from editing the weak template or Eckstein-BI, the
lowest-scoring seeds. The 2 edits that did not apply failed because Shinka's search text did not match
the parent; the search allows one attempt per proposal (`max_patch_attempts` 1), so whether to allow one
repair attempt is open. Two of high's programs (proposals 1 and 3) failed the 2-level fit on the
validation mice and kept their per-subject score (`two_level_failed`); medium had none.
