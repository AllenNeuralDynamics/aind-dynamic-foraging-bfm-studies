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
  freeze: analysis/freeze_hattori.py
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
**Incomplete: 32 of 149 subjects finished; not a result.**

| arm | held-out likelihood (trial-weighted) |
|---|---|
| HB-Hattori2019 two-level, session-marginal (headline) | **0.7433** ± 4.4e-06 (MC) |
| HB-Hattori2019 two-level, trial-marginal | 0.7326 |
| per-subject MLE Hattori (study 01 r8) | 0.7127 |
| per-subject MLE CTT (study 01 r8) | 0.7170 |
| GRU H128 (study 01 r1) | D≈10 0.7219, D≈30 0.7250, D≈100 0.7262, D≈300 0.7267, D≈614 0.7268 |

- Subjects 32; scored trials 145,985 (comparators: 1,013,172).
- Convergence gates (R-hat < 1.01, bulk ESS ≥ 400): 32 of 32 subjects pass; worst R-hat 1.006, lowest ESS 1005; 66 divergences. Converged subjects only: 0.7433.
- Pareto k̂ gate: 33 of 369 scored sessions fail (k̂ ≥ 0.7 or unjudged).
<!-- END result-1 -->

## What it says

Written once the full cohort is in; until then the block above is marked incomplete. Caveats
that hold regardless:

- The MC error covers only the scoring average, not posterior uncertainty or seed-to-seed
  variation. The GRU seed spread (~0.0004, study 01) is the yardstick for whether a gap matters.
- Sessions failing the k̂ gate are kept in the headline. Their estimates may be biased low, which
  would understate the HB.
- One seed per subject: no run-to-run variance is measured.
