# Changelog — 15-behavioral-fingerprints

## 2026-11-05

- P2 (r3, draft): rollout fingerprints of the same models on gym tasks built from each score
  session (ADR 0006; amended with Han's decisions: engaged trials, sessions without block
  parameters left out, the holds check on `BlockMinReward` only). 125 of the 149 test subjects
  have a task. No model reaches the split-half floor; the best (CompareToThreshold per subject,
  the GRU, Bari2019 2-level) score about 1.4. 2-level fits, better in replay for 16 of 18 models,
  are better in rollout for 9; their losses curve falls where the mice's rises. The
  `BlockMinReward` hold moves the fingerprints by at most 0.45 of the floor, so ADR 0006 stands.
  Tallies are stored per model in `analysis/p2_rollouts/`.

## 2026-11-04

- ADR 0009 (decided with Han): one fingerprint score per model and mode, the geometric mean of
  its floor ratios, for leaderboards beside the grid; amends ADR 0001. r2 shows the replay score.
- P1 (r2, draft): replay fingerprints of study 12's 19 models (per subject and 2-level EM),
  and a per-subject GLM (Su2022 design, fit here) on the 149 test subjects'
  held-out sessions. Every replay reproduces study 12's stored held-out score (≤1.8e-11), except
  Rescorla-Wagner's failed 2-level fit, which is left out. The best 2-level models sit 1.2–1.6×
  the binomial floor. Tallies are stored per model in `analysis/p1_replay/` (ADR 0004).
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
