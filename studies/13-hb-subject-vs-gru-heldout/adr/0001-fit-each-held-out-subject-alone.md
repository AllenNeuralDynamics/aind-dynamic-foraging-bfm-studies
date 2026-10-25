# Fit each held-out subject alone: no population level, no training subjects

Study 08's three-level fit (population over subjects over sessions) did not mix at D≈99–614, so
no rung was usable. This study fits a two-level HB to each held-out subject on its own context
sessions, under a fixed hyperprior, and never fits the training subjects. Every fit is small
and independent, so the sampling problem that sank 08 does not arise.

## Consequences

- There is no D dial and no zero-shot score: the result is one number per forager, plotted as
  a horizontal line against the GRU's D curve.
- The two-level HB nests the per-subject MLE (the same forager with zero between-session
  spread), so the gap between them measures what session-to-session variability is worth.

## Considered options

- **Two-stage empirical Bayes** (fit training subjects, fit a population to them, condition
  held-out subjects on it). Rejected: it restores the population level and the D dial, which
  is study 08's question, not this one.
- **Fit training subjects too, without pooling.** Rejected: nothing in the held-out score
  depends on them.
