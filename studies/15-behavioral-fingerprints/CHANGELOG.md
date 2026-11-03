# Changelog — 15-behavioral-fingerprints

## 2026-11-03

- Study created (#94); ADRs 0001–0007 and CONTEXT.md from the design grill with Han.
- P0 (r1, draft): the fingerprint library (aind-dynamic-foraging-models `feat/fingerprints-v1`)
  matches MEvolve's replay fingerprints (19 programs × 149 subjects, ≤1.4e-14) and rebuilds the
  wrapper's committed RL rollout rows (≤3.3e-16); 24.6% of the test subjects' measurable coupled
  blocks ran past `block_max`, which gym rollouts cannot reproduce. ADR 0004 names the run-length
  distribution's bins.
