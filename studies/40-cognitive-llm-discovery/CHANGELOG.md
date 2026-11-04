# Changelog — 40-cognitive-llm-discovery

## 2026-11-04

- Gate 3's numbers are regenerable (issue #97, review of #98): `make freeze-engine-smoke` freezes the
  engine-smoke ledgers, Shinka's program table and the hand-typed Console figures into
  `analysis/engine_smoke/`, and `make gate3` (in `all`) writes `analysis/gate3_engine_smoke.json`, which
  the notes' Result section now cites; the ledger's cost reads $0.616 (was $0.615), and the Console's
  Caching page is dated 11:00 UTC, not 19:48.

## 2026-11-03

- Variant `engine-smoke` (issue #97): gate 3's engine smoke, two 10-proposal ShinkaEvolve runs
  (Claude Sonnet 5.5 at effort medium and high) on 8 + 8 smoke subjects drawn after the draft cohorts,
  with the five-island seed layout of ADR 0006; code in AllenNeuralDynamics/aind-dynamic-foraging-mevolve#32.
- Gate 3 passes (`engine-smoke@20261103-030341`, results in `variants/engine-smoke/notes.md`): 22 Claude
  calls with no errors; the ledger's token counts match the Console and its cost is 4% above the
  Console's list price; caching works; seeds sit on their islands. Effort medium is kept for the search:
  high cost 38% more per proposal and improved fewer parents. PLAN.md notes that its budget is at list
  price, about twice the bill after the organisation's discount.
- W&B (issue #99): new runs log to `NeuralDynamics/dfbfm-mevolve`, shared with MEvolve; the README
  frontmatter and the engine-smoke run configs name it. `engine-smoke@20261103-030341` stays in
  `aind-dynamic-foraging-model`.

## 2026-11-02

- "Session level" retired (issue #87): fits are named by their levels and method (per session, per
  subject, 3-level HB, 2-level HB, 2-level EM; table in MEvolve's `CONTEXT.md`). ADR 0002 renamed to
  `0002-fit-programs-with-the-2-level-model.md`. PLAN.md cites study 12 r3 (`r3_2level_em.json`) for
  the literature models as 2-level EM, and ADR 0006 states the seed-order rule once.

## 2026-10-31

- Study created (issue #73): README design and motivation, `seeded-ctt` variant, PLAN.md (draft,
  before the design grill), CONTEXT.md, ADR 0001 (one score, then simplify) and ADR 0002 (fit programs
  at the session level). Adds the `cognitive/mevolve` model tag.
