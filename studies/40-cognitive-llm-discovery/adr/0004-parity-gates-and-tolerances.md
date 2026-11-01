# Parity gates and their tolerances

Before any search, MEvolve's evaluator must reproduce the lab's own numbers, because the differences
this study cares about are near 0.001 in held-out likelihood (about 0.7 nats per 500-trial session),
so an evaluator off by more than a few parts in 10,000 would blur every comparison (decided with Han,
2026-10-31).

| gate | check | pass |
|---|---|---|
| 1a dynamics | every program in the seed library (study 12's 19 literature models), given study 12's fitted parameters (`cross_study.literature_fitted_params`), against its reference forager in `aind-dynamic-foraging-models` | per-trial probabilities within 1e-6; pooled likelihood within 1e-5 |
| 1b fitter | the gradient fitter, for each of the 19 literature models, against study 12's per-subject differential-evolution fits (`cross_study.literature_per_subject`; study 12 checks its Hattori, Bari and CTT refits against study 01 r8) | pooled held-out likelihood within 0.0002, or better; for at least 95% of subjects the context-session fit no worse by more than 0.5 nats |
| 1c session level | CTT and Hattori, the only literature models with a hierarchical Bayesian reference, fitted at the session level against study 13's HB-CTT and HB-Hattori | each within 0.001, and the gap between the two models within 0.0005 |

If both models land 0.001 to 0.002 low together at gate 1c, the offset is documented and the headline
is rescored with study 13's exact method (full posterior sampling). Beyond 0.002, the search falls back
to one parameter set per subject (ADR 0002).

## Considered options

- **A looser 1b tolerance (0.0005)**, as first planned: too close to the differences being measured.
- **A stricter 1c tolerance (0.0005)**: study 13 samples the full posterior and rescues hard sessions by
  importance sampling, while the search uses one best fit and a few hundred draws, so an exact match is
  not expected.
