---
schema_version: 1
id: r1
slug: hattori-pooled
status: draft
authors:
- han
wandb_groups:
- hattori@20261026-002131
inputs:
  script: analysis/r1_hattori_pooled.py
  data: analysis/hattori_runs.json
  plugin: analysis/hattori_plugin.json
  freeze: analysis/freeze_hattori.py
  freeze_plugin: analysis/freeze_plugin.py
upstream:
- study: 01-gru-scaling-law
  report: r8
- study: 01-gru-scaling-law
  report: r1
reproduce: make -C studies/13-hb-subject-vs-gru-heldout r1
tags:
  model: [cognitive/bh]
  question: [comparison]
---
# r1 — HB-Hattori2019 two-level vs GRU and per-subject MLE on held-out subjects

**Question.** Fitted to each held-out subject alone (no population level, no training subjects),
where does the two-level HB-Hattori2019 land between the per-subject MLE (the same forager with no
session-to-session variability) and the GRU, on the same held-out eval sessions?

**Method.** One subject per HPC CPU array task (launch `hattori@20261026-002131`, records in
`variants/hattori/launch_record/production/`). Each subject is conditioned on its even-indexed
sessions under the fixed hyperprior (ADR 0003): 4 chains, 2000 warmup, 1500 draws each. It is
scored on its odd-indexed eval sessions by the session-marginal likelihood: 6,000 posterior
draws × 24 fresh session draws (ADR 0002). Pooled = exp(Σ log-likelihood / Σ trials) over
subjects; Monte Carlo errors add in quadrature across runs. Comparators are read from study 01:
r8 for the per-subject MLE and r1 for the GRU H128 curve.

<!-- BEGIN result-1 -->
| arm | held-out likelihood (trial-weighted) |
|---|---|
| HB-Hattori2019 two-level, session-marginal (headline) | **0.7213** ± 1.7e-06 (MC) |
| HB-Hattori2019 two-level, plug-in: one fixed parameter set per subject (typical-session posterior median), no within-session inference | 0.7107 |
| HB-Hattori2019 two-level, trial-marginal | 0.7090 |
| per-subject MLE Hattori (study 01 r8) | 0.7127 |
| per-subject MLE CTT (study 01 r8) | 0.7170 |
| GRU H128 (study 01 r1) | D≈10 0.7219, D≈30 0.7250, D≈100 0.7262, D≈300 0.7267, D≈614 0.7268 |

- Subjects 149; scored trials 1,013,172 (comparators: 1,013,172).
- Convergence gates (R-hat < 1.01, bulk ESS ≥ 400, no divergences): 103 of 149 subjects pass; worst R-hat 1.173, lowest ESS 16; 304 divergences. Converged subjects only: 0.7166.
- Pareto k̂ gate: 262 of 2138 scored sessions fail (k̂ ≥ 0.7 or unjudged).
<!-- END result-1 -->

## What it says

Read off the frozen result above (2026-10-26):

- **The HB improves on the per-subject MLE by ~0.009.** The two-level HB beats the same forager
  fitted per subject with one parameter set (MLE Hattori) and the best classical family (MLE CTT).
  This is the overall HB–MLE improvement, not session-to-session variability alone: the HB also
  differs from the MLE by its fixed prior and by averaging over the posterior instead of plugging
  in a point estimate. Isolating the session spread would need a zero-spread Bayesian control.
- **It roughly ties the smallest GRU and stays below the rest.** It matches the GRU at D≈10. The
  GRU from D≈30 up is still clearly ahead; those gaps are many times the GRU's ~0.0004 seed spread.
- **Where the gain over the MLE comes from (plug-in control).** Scoring each subject with one fixed
  parameter set (the posterior median of its typical session), with no between-session spread and no
  within-session inference, gives slightly *less* than the MLE. So the HB's prior and partial
  pooling, on their own, do not beat the per-subject MLE. The HB's whole advantage over its own
  plug-in comes from modelling how sessions differ and inferring, within each session from its
  earlier trials, where that session sits. One subject's degenerate MLE fit (r3) is regularized by the
  prior, but in aggregate that is not where the gain comes from.
- **The scoring method decides the comparison.** Scored by the trial-marginal (study 08's method),
  the same fits fall *below* the per-subject MLE. ADR 0002 explains why the session-marginal is the
  fair score: it predicts each trial from the session's earlier trials, as the GRU does.

Why this stays `draft`:

- **k̂.** 12% of scored sessions fail the Pareto k̂ gate. Their plain draw-average rests on few
  draws and is likely biased low, so the headline probably understates the HB. Resolving it means
  rescoring those sessions from the saved draws with importance sampling (ADR 0002's fallback); no
  refit is needed.
- **Convergence.** Some subjects miss the R-hat/ESS gates or have divergences (counts above). PLAN.md's
  rule is one refit with longer chains for failing subjects. The converged-only number is given
  beside the headline. Requiring zero divergences as well leaves about two thirds of subjects,
  and that subset scores lower than the headline: the subjects that diverge are not a random
  third, so the converged-only number describes a different population and is no like-for-like
  comparison with the all-subject comparators.
- **Uncertainty.** The MC error covers only the scoring average, not posterior or seed variation.
  There is one seed per subject, so no run-to-run variance is measured.
- **Rescue.** 16 subjects ran out of memory at 64G during scoring and were rerun at 200G as a
  recorded intervention (`variants/hattori/launch_record/production/hpc_rescue-oom.json`). Their
  first runs are `crashed` in the freeze and excluded.
