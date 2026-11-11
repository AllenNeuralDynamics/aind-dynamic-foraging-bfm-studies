# One fingerprint library, in aind-dynamic-foraging-models, behind MEvolve's program protocol

The statistics, the replay and rollout drivers, and the comparison live in one library in
`aind-dynamic-foraging-models`. Studies, MEvolve and the wrapper import it. It replaces two copies
that disagree today: the wrapper's `post_training_analysis/generative_analysis.py`, which counts
only sampled choices, and MEvolve's `fingerprints.py`, which handles only replay, with different
bins and aggregation.

Every statistic has one shape: a key computed from the history before trial t, and an outcome at t.
Animal sessions and rollouts fill each bin with observed outcomes; replay fills it with the model's
probability of that outcome. So one tally function serves all three.

A model enters through MEvolve's program protocol (`init_latents`, `predict`, `update`), with one
adapter per family: a literature forager wraps its `act` and `learn`; the GLM keeps its lag buffer
as latent variables; the GLM-HMM keeps its filtered state belief; the GRU steps the wrapper's
runner. Each adapter must reproduce its model's own teacher-forced probabilities on a session to
1e-6 before its fingerprints count.

The wrapper keeps its rollout loop, because it owns the GRU runners, and calls the library for the
statistics. Its own statistic functions stay frozen until a GRU rollout reproduces through the
library, then are deprecated; they are not deleted while W&B keys still cite them (decided with
Han, 2026-11-03).

## Considered options

- **In MEvolve.** Its fingerprints are the newest code, but the wrapper and the studies would then
  depend on the search engine's repository.
- **The statistics in aind-dynamic-foraging-basic-analysis, the adapters in foraging-models.**
  Cleaner in principle, but every change would need two PRs in two packages.
- **`studies/util`.** MEvolve and the wrapper cannot import from the studies repository.

## Consequences

- MEvolve moves its fitter to foraging-models after its gate 4 for the same reason; its fingerprint
  gate can then read this library. That switch is a proposal to the MEvolve sessions, not a decision
  made here. MEvolve fixed its gate rule before any search, so the library also emits MEvolve's
  current `replay_gap` (pooled, trial-weighted, mean of three statistics) unchanged: the gate's
  code can move without its rule changing.
