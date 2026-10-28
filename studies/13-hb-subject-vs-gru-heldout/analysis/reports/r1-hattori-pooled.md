---
schema_version: 1
id: r1
slug: hattori-pooled
status: live
authors:
- han
wandb_groups:
- hattori@20261026-002131
- hattori@20261027-181430
- hattori@20261027-223626
- hattori@20261027-181415
inputs:
  script: analysis/r1_hattori_pooled.py
  data: analysis/hattori_final_runs.json
  production: analysis/hattori_runs.json
  refit: analysis/hattori_refit_runs.json
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

**Refit and k̂ rescue (ADR 0004).** The 46 subjects failing a convergence gate in production were
refit once more, non-centred at target acceptance 0.95 with 3000 warmup + 4 × 3000 draws
(`hattori@20261027-223626`; a centred refit before it was worse on every gate and is not used).
The other 103 keep their production fit, scored again through the same data path
(`hattori@20261027-181430`; the plain scores reproduce production exactly). In both, each session
failing Pareto k̂ on the plain draw-average was rescored by
importance sampling against the exact predictive mixture: a t proposal at the session's mode,
mixed defensively with the mixture, 24,000 draws. The headline uses the importance-sampled value
for those sessions and the plain value elsewhere; both estimate the same integral.

<!-- BEGIN result-1 -->
| arm | held-out likelihood (trial-weighted) |
|---|---|
| HB-Hattori2019 two-level, session-marginal (headline: non-centred refit of the gate-failing subjects, k̂-failing sessions importance-sampled; ADR 0004) | **0.7213** ± 2.0e-06 (MC); 0.721295 |
| … same fits, plain draw-average on every session | 0.721296 |
| … production (before the refit and the k̂ rescue) | 0.721295 |
| HB-Hattori2019 two-level, plug-in: one fixed parameter set per subject (typical-session posterior median), no within-session inference | 0.7107 |
| HB-Hattori2019 two-level, zero-spread: every posterior draw of the typical session, no between-session spread, session-marginal | 0.7174 |
| HB-Hattori2019 two-level, trial-marginal | 0.7090 |
| per-subject MLE Hattori (study 01 r8) | 0.7127 |
| per-subject MLE CTT (study 01 r8) | 0.7170 |
| GRU H128 (study 01 r1) | D≈10 0.7219, D≈30 0.7250, D≈100 0.7262, D≈300 0.7267, D≈614 0.7268 |

- Subjects 149; scored trials 1,013,172 (comparators: 1,013,172).
- Convergence gates (R-hat < 1.01, bulk ESS ≥ 400, no divergences): 139 of 149 subjects pass; worst R-hat 1.210, lowest ESS 13; 45 divergences. Converged subjects only: 0.7194.
- Pareto k̂ gate: 256 of 2138 scored sessions fail on the plain draw-average (production: 262); 256 were rescored by importance sampling, after which 37 still fail (k̂ ≥ 0.7 or unjudged).
- Refit (ADR 0004) of the 46 subjects failing a gate in production. Centred (rejected): divergences 304 → 11,007, R-hat/ESS passes 39 → 24. Non-centred at target acceptance 0.95 (used): divergences 304 → 45, R-hat/ESS passes 39 → 45, 36 of 46 passing every gate.
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
- **Where the gain over the MLE comes from (two controls).** All three are HB fits on the same
  subjects, scored on the same eval sessions:
  - **plug-in:** one fixed parameter set, the median of the typical-session posterior. It scores
    slightly *below* the per-subject MLE. That step changes the prior, the pooling and the point
    estimator together, so it shows that a single HB point summary doesn't beat the MLE, not that
    the prior contributes nothing.
  - **zero-spread:** every posterior draw of the typical session, no between-session spread,
    session-marginal. This gives most of the gain over the plug-in. It averages over uncertainty about
    the subject, and within each session it reweights those draws by the session's earlier trials.
    That alone already matches the best per-subject MLE (CTT).
  - **full HB:** adds between-session spread. It gains a smaller further step on top of zero-spread.

  So the HB's advantage is mostly posterior averaging with within-session inference over the
  subject's uncertain parameters, plus a smaller part from explicitly modelling how sessions differ.
  How much the prior contributes inside those steps is not isolated here; r3's degenerate MLE fit is
  one case where it clearly matters.
- **The scoring method decides the comparison.** Scored by the trial-marginal (study 08's method),
  the same fits fall *below* the per-subject MLE. ADR 0002 explains why the session-marginal is the
  fair score: it predicts each trial from the session's earlier trials, as the GRU does.

- **Neither the refit nor the k̂ rescue moves the headline.** The three HB rows at the top agree to
  within 2e-6 (production, the same fits scored plainly, and the headline), against about a
  million scored trials. That is agreement between two Monte
  Carlo estimates of the same quantity, not proof that either is unbiased (the log of an unbiased
  likelihood estimate is not unbiased, and some sessions still fail the gate); it does show that
  the headline hinges neither on the sessions k̂ flagged nor on the subjects that failed to converge.
- **With the refit most subjects converge, and the converged-only number moves toward the
  headline.** The converged-only subset now covers most subjects (count above), so it is nearly the
  same population as the headline and the comparators.

Caveats:

- **k̂.** After importance sampling a minority of the rescored sessions still fail the gate
  (count above; split about evenly between subjects that pass and fail the convergence gates). For those
  the t proposal fits poorly (k̂ up to ~8); their estimates are reported as they are, not dropped.
- **Convergence.** After the refit a few subjects still miss a gate (counts above; the worst R-hat
  and lowest ESS are one subject). They stay in the headline, as the comparators score every
  subject; the converged-only number beside it excludes them. Two refits were run where PLAN.md
  planned one: the centred one failed, and Han approved the non-centred one (ADR 0004).
- **Uncertainty.** The MC error covers only the scoring average, not posterior or seed variation.
  There is one seed per subject, so no run-to-run variance is measured.
- **Rescue.** 16 subjects ran out of memory at 64G during scoring and were rerun at 200G as a
  recorded intervention (`variants/hattori/launch_record/production/hpc_rescue-oom.json`). Their
  first runs are `crashed` in the freeze and excluded.
