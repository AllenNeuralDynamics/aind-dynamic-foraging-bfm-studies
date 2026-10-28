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
- hattori@20261027-201810
inputs:
  script: analysis/r1_hattori_pooled.py
  data: analysis/hattori_final_runs.json
  production: analysis/hattori_runs.json
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

**k̂ rescue (ADR 0004).** Every subject's production draws were scored again through the same
data path (`hattori@20261027-181430` and `hattori@20261027-201810`; the plain scores reproduce
production exactly), and each session failing Pareto k̂ on the plain draw-average was rescored by
importance sampling against the exact predictive mixture: a t proposal at the session's mode,
mixed defensively with the mixture, 24,000 draws. The headline uses the importance-sampled value
for those sessions and the plain value elsewhere; both estimate the same integral.

<!-- BEGIN result-1 -->
| arm | held-out likelihood (trial-weighted) |
|---|---|
| HB-Hattori2019 two-level, session-marginal (headline: k̂-failing sessions importance-sampled, ADR 0004) | **0.7213** ± 2.0e-06 (MC); 0.721297 |
| … same fits, plain draw-average on every session (= production) | 0.721295 |
| HB-Hattori2019 two-level, plug-in: one fixed parameter set per subject (typical-session posterior median), no within-session inference | 0.7107 |
| HB-Hattori2019 two-level, zero-spread: every posterior draw of the typical session, no between-session spread, session-marginal | 0.7174 |
| HB-Hattori2019 two-level, trial-marginal | 0.7090 |
| per-subject MLE Hattori (study 01 r8) | 0.7127 |
| per-subject MLE CTT (study 01 r8) | 0.7170 |
| GRU H128 (study 01 r1) | D≈10 0.7219, D≈30 0.7250, D≈100 0.7262, D≈300 0.7267, D≈614 0.7268 |

- Subjects 149; scored trials 1,013,172 (comparators: 1,013,172).
- Convergence gates (R-hat < 1.01, bulk ESS ≥ 400, no divergences): 103 of 149 subjects pass; worst R-hat 1.173, lowest ESS 16; 304 divergences. Converged subjects only: 0.7166.
- Pareto k̂ gate: 262 of 2138 scored sessions fail on the plain draw-average (production: 262); 262 were rescored by importance sampling, after which 44 still fail (k̂ ≥ 0.7 or unjudged).
- Refit (ADR 0004): one centred refit of the 46 subjects failing a gate was worse on every gate (divergences 304 → 11,007) and is not used; every subject keeps its production fit (`variants/hattori/launch_record/refit/README.md`).
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

- **The starved plain average was not biased.** Rescoring the sessions that failed k̂ by
  importance sampling moves the headline by less than 1e-5 (the first two rows
  above): over all rescored sessions the two estimators differ by +3 nats in total, against
  about a million scored trials. The k̂ failures flagged unreliable *weights*, not a biased total.

Caveats:

- **k̂.** After importance sampling a minority of the rescored sessions still fail the gate
  (count above; split about evenly between subjects that pass and fail the convergence gates). For those
  the t proposal fits poorly (k̂ up to ~8); their estimates are reported as they are, not dropped.
- **Convergence.** Some subjects miss the R-hat/ESS gates or have divergences (counts above). The one
  refit PLAN.md allows was run (centred, target acceptance 0.95, longer chains) and was worse on
  every gate, so it is not used (ADR 0004); a non-centred refit at higher target acceptance is the
  untried remedy. The converged-only number is given beside the headline. Requiring zero divergences as well leaves about two thirds of subjects,
  and that subset scores lower than the headline: the subjects that diverge are not a random
  third, so the converged-only number describes a different population and is no like-for-like
  comparison with the all-subject comparators.
- **Uncertainty.** The MC error covers only the scoring average, not posterior or seed variation.
  There is one seed per subject, so no run-to-run variance is measured.
- **Rescue.** 16 subjects ran out of memory at 64G during scoring and were rerun at 200G as a
  recorded intervention (`variants/hattori/launch_record/production/hpc_rescue-oom.json`). Their
  first runs are `crashed` in the freeze and excluded.
