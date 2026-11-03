# Changelog — 40-cognitive-llm-discovery

## 2026-11-03

- Variant `engine-smoke` (issue #97): gate 3's engine smoke, two 10-proposal ShinkaEvolve runs
  (Claude Sonnet 5.5 at effort medium and high) on 8 + 8 smoke subjects drawn after the draft cohorts,
  with the five-island seed layout of ADR 0006; code in AllenNeuralDynamics/aind-dynamic-foraging-mevolve#32.

## 2026-11-02

- "Session level" retired (issue #87): fits are named by their levels and method (per session, per
  subject, 3-level HB, 2-level HB, 2-level EM; table in MEvolve's `CONTEXT.md`). ADR 0002 renamed to
  `0002-fit-programs-with-the-2-level-model.md`. PLAN.md cites study 12 r3 (`r3_2level_em.json`) for
  the literature models as 2-level EM, and ADR 0006 states the seed-order rule once.

## 2026-10-31

- Study created (issue #73): README design and motivation, `seeded-ctt` variant, PLAN.md (draft,
  before the design grill), CONTEXT.md, ADR 0001 (one score, then simplify) and ADR 0002 (fit programs
  at the session level). Adds the `cognitive/mevolve` model tag.
