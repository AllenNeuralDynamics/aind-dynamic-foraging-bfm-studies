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

- [ ] no API errors; refusals and cut-off replies counted as failed proposals
- [ ] prompt caching works (cache reads > 0 after the first call)
- [ ] the ledger's cost matches the Anthropic Console within 5%
- [ ] each seed on its island, the template on island 0, children on their parent's island
- [ ] ShinkaEvolve's web viewer shows the run
- [ ] medium vs high effort: valid and improving proposals per dollar

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group `engine-smoke@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Status.** Launched 2026-11-03 03:03 PT as `engine-smoke@20261103-030341` (SLURM 27586076;
record in `launch_record/engine-smoke-1/`). Outputs: `/allen/aind/scratch/han.hou/mevolve/runs/engine-smoke/20261103-030341/`
(`medium/` and `high/` each hold the Shinka database, `ledger.jsonl` with every API call and its cost,
`proposals.csv`, `run_report.md`, `run_summary.json`; `comparison.md` sets the two side by side). A
preflight of two calls (effort low) before launch: the key, the model and the request shape work, and
the cache was written once (14,370 tokens) and read on the second call; it cost $0.04.

**Viewer.** SLURM job 27586077 starts ShinkaEvolve's web viewer on n268, port 8765, when the smoke job
ends, for two days. In VS Code connected to n268, forward port 8765 and open http://localhost:8765. It
has no login and listens on all interfaces of n268 (cluster network only); cancel it with `scancel` when
done.
