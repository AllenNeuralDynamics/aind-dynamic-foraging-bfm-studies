# Changelog — 15-behavioral-fingerprints

## 2026-11-04

- ADR 0008 (decided with Han): replay uses a binomial noise floor, rollouts keep the split-half
  floor (foraging-models `binomial_floor`); CONTEXT names both floors.

## 2026-11-03

- Study created (#94); ADRs 0001–0007 and CONTEXT.md from the design grill with Han.
- P0 (r1, draft): the fingerprint library (aind-dynamic-foraging-models `feat/fingerprints-v1`)
  matches MEvolve's replay fingerprints (19 programs × 149 subjects, ≤1.4e-14) and rebuilds the
  wrapper's committed RL rollout rows (≤3.3e-16); 12.5% of the test subjects' measurable coupled
  blocks were certainly held past `block_max` (24.6% counting ignored trials, which the task can
  add), which gym rollouts cannot reproduce. W&B project renamed to
  `dfbfm_15-behavioral-fingerprints` (studies #99; no runs logged yet). ADR 0004 names the run-length
  distribution's bins.
