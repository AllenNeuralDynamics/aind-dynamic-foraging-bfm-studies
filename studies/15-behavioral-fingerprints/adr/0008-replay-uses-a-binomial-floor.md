# Replay uses a binomial floor; rollouts keep the split-half floor

A noise floor is the gap a perfect model would still have, and what is left over depends on the
mode. In replay the model predicts the subject's own trials, so if it were the true process each
outcome would be a coin flip with its probability, and only that coin-flip noise is left. Its
floor is the binomial floor: for each bin the subject keeps, with `n` of its trials and observed
switch rate `r`, the expected miss E|X/n − r| for X ~ Binomial(n, r), weighted by `n` as the gap
is (foraging-models `fingerprints.binomial_floor`). A rollout makes new sessions, so a perfect
model also differs from the subject by the ordinary differences between sessions; its floor stays
the split-half floor of ADR 0003. A perfect model then scores about 1 in both modes. Decided with
Han, 2026-11-04; this settles the question ADR 0003 left open.

The binomial floor uses only the subject's trial counts and outcomes, so every model in a column
shares it, and the tallies do not change (`fingerprints-v1`). It covers the rate statistics; a
distribution has no replay prediction.

## Considered options

- **One split-half floor for both modes** (ADR 0003 as first written). It holds the differences
  between sessions, which a replay can follow (each session's own history, and a 2-level fit's
  per-session parameters). With sessions whose switch rates differ, the true model replays at
  0.50 × the split-half floor but 0.97 × the binomial floor (the library's test), so the shared
  floor makes a perfect replay look twice as good as the data allows.
- **A model-specific floor**, from outcomes drawn with each model's own predictions. Exact for
  that model, but every model would get its own denominator, so a column of the grid would no
  longer compare models.

## Consequences

- The binomial floor sits slightly below a perfect model's gap when a bin pools sessions with
  different rates, because the pooled rate overstates the coin-flip noise; a perfect replay then
  scores a little under 1 (0.97 in the test). That errs on the side of a high floor.
