# Fit each held-out subject alone: no population level, no training subjects

Study 08's three-level fit (population over subjects over sessions) did not mix at D≈99–614, so
no rung was usable. This study fits a two-level HB to each held-out subject on its own context
sessions, under a fixed hyperprior, and never fits the training subjects. Every fit is small
and independent, so the sampling problem that sank 08 does not arise.

## Consequences

- There is no D dial and no zero-shot score: the result is one number per forager, plotted as
  a horizontal line against the GRU's D curve.
- The gap between the two-level HB and the per-subject MLE (the same forager, one parameter set
  per subject) is the overall HB improvement. It is not session-to-session variability alone: the
  HB also brings a fixed prior and averages over its posterior instead of plugging in a point
  estimate. Isolating the spread would need a zero-spread Bayesian control.

## Considered options

- **Two-stage empirical Bayes** (fit training subjects, fit a population to them, condition
  held-out subjects on it). Rejected: it restores the population level and the D dial, which
  is study 08's question, not this one.
- **Fit training subjects too, without pooling.** Rejected: nothing in the held-out score
  depends on them.
