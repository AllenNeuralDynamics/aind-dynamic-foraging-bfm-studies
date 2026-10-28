---
schema_version: 1
id: r5
slug: ctt-pooled
status: live
authors:
- han
wandb_groups:
- ctt@20261028-003745
- ctt@20261028-020002
- hattori@20261027-181430
- hattori@20261027-223626
inputs:
  script: analysis/r5_ctt_pooled.py
  data: analysis/ctt_final_runs.json
  production: analysis/ctt_runs.json
  hattori: analysis/hattori_final_runs.json
  freeze: analysis/freeze_ctt.py
  figure: analysis/fig_r5_ctt.svg
upstream:
- study: 01-gru-scaling-law
  report: r8
- study: 01-gru-scaling-law
  report: r1
reproduce: make -C studies/13-hb-subject-vs-gru-heldout r5
tags:
  model: [cognitive/bh]
  question: [comparison]
---
# r5 — HB-CTT two-level vs HB-Hattori, per-subject MLE and GRU on held-out subjects

**Question.** The best per-subject classical model is CTT (compare-to-threshold). Fitted as a
two-level HB to each held-out subject alone, where does it land between its own per-subject MLE,
HB-Hattori (r1) and the GRU?

**Method.** As r1, with the CTT forager in place of Hattori2019 (variant `ctt`). CTT keeps one
value that is compared with a threshold to decide stay or switch. Its four free parameters are a
learning rate, the threshold, a softmax inverse temperature and a left bias; as in the MLE
baseline it has one learning rate and no choice kernel. Each subject is fitted on its even-indexed
sessions and scored on its odd-indexed ones by the session-marginal likelihood
(`ctt@20261028-003745`). The 42 subjects that failed a convergence gate were refit as in hattori:
non-centred, target acceptance 0.95, longer chains (`ctt@20261028-020002`, ADR 0004). Sessions
failing Pareto k̂ are importance-sampled.

<!-- BEGIN result-1 -->
![HB-CTT against the other models](../fig_r5_ctt.svg)

*(a) Pooled held-out likelihood. Dashed lines: the GRU H128 curve (study 01 r1); darker means more training subjects. (b) Per-subject difference of HB-CTT from the per-subject MLE CTT and from HB-Hattori.*

| model | held-out likelihood (trial-weighted) |
|---|---|
| HB-CTT two-level, session-marginal (headline) | **0.7225** ± 9.9e-07 (MC) |
| … production, before the refit | 0.7225 |
| HB-Hattori2019 two-level (r1) | 0.7213 |
| per-subject MLE CTT (study 01 r8) | 0.7170 |
| per-subject MLE Hattori (study 01 r8) | 0.7127 |
| GRU H128 (study 01 r1) | D≈10 0.7219, D≈30 0.7250, D≈100 0.7262, D≈300 0.7267, D≈614 0.7268 |

- Per subject: HB-CTT beats the MLE CTT for 145 of 149 subjects (median +0.0043), and HB-Hattori for 100 (median +0.0014).
- Subjects 149; scored trials 1,013,172 (comparators: 1,013,172).
- Convergence gates (R-hat < 1.01, bulk ESS ≥ 400, no divergences): 137 of 149 pass; worst R-hat 1.018, lowest ESS 376; 54 divergences. Converged only: 0.7255.
- Refit: the 42 subjects failing a gate in production were refit as in hattori (ADR 0004). Production: 107 of 149 passing every gate, 1,189 divergences.
- Pareto k̂: 107 of 2138 sessions failed on the plain draw-average and were importance-sampled; 13 still fail. All-plain 0.722466; headline 0.722465.
- Trial-marginal (study 08's score): 0.7128.
<!-- END result-1 -->

## What it says

- **HB-CTT is the best classical model here.** It beats HB-Hattori by about 0.001 and its own
  per-subject MLE by about 0.0055. Per subject, it beats the MLE CTT for almost every subject.
- **It passes the smallest GRU but not the larger ones.** HB-CTT sits just above the GRU at D≈10 and
  clearly below it from D≈30 up. The gap to the largest GRU is about 0.004, many times the GRU's
  ~0.0004 seed spread.
- **The refit and the k̂ rescue leave the headline unchanged.** The refit clears most convergence
  failures, and the production and final headlines agree to 1e-4.

Caveats:

- **Convergence.** A few subjects still fail a gate after the refit (counts above). They stay in
  the headline, as the comparators score every subject. Their scores are low: the converged-only
  number is about 0.003 above the headline, so it describes a slightly easier set of subjects.
- **Uncertainty.** The MC error covers only the scoring average, not posterior or seed variation.
  There is one seed per subject.
