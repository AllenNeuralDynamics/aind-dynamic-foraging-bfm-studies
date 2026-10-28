---
schema_version: 1
id: r1
slug: round-1
status: draft
authors:
- han
wandb_groups:
- lag-scan@20261027-200129
- lag-scan@20261027-223410
- hurdle@20261027-212905
- reference@20261027-213006
inputs:
  script: analysis/r1_round_1.py
  data: analysis/frozen_round1.json
  freeze: analysis/freeze_round1.py
upstream:
- study: 02-gru-scaling-law-ignore
  report: r1
reproduce: make -C studies/14-glmhmm-vs-gru-heldout r1
tags:
  model: [glmhmm]
  question: [comparison]
---
# r1 — GLM-HMM round 1: held-out score vs K, and what the states are

**Question.** On the GRU's own cohort, split and metric code, how well does a GLM-HMM over
left / right / ignore predict held-out mice as the number of states K grows? Which emission
(hurdle or reference) does better, and how close does it get to the 3-way GRU? What do the
states of the selected model look like?

**Method.** Snapshot 20260603, ignores included (study 02's 3-way setup).
- **Global fit:** the 614 training mice's train (even) sessions, 4,887,818 trials. K = 1 first,
  then 20 restarts from its weights plus noise; the best log joint is kept. EM cap 500,
  tolerance 1e-6 nats/trial.
- **Per-mouse refits:** MAP refits (σ = 2, α = 2) starting from the global fit.
- **Scores:** the forward filter's one-step-ahead predictions, through the wrapper's GRU metric
  code.
- **Selection score:** the training mice's per-mouse refits, scored on their eval (odd) sessions.
  It chooses L (lag scan, K = 1) and K (argmax).
- **Held-out score:** the same refit protocol on the 149 held-out mice, reported and never used
  to choose.
- **History length** L = 10, from the lag scan. A follow-up launch added L = 15.
- **Comparator:** study 02 r1's 3-way GRU at D = 614, read from its committed `frozen_runs.json`.
- **States:** taken from the per-mouse refits' smoothed posteriors and Viterbi paths over each
  held-out mouse's sessions. The refits start from the global states, and the alignment
  diagnostic (result-3) checks they stay in order.

<!-- BEGIN result-1 -->
Geometric-mean per-trial likelihoods. **selection**: training mice, per-mouse refits scored on their eval sessions (chooses K and L). **held-out**: 149 mice, their eval sessions; *matched* = per-mouse refit (the GRU's fine-tune protocol), *zero-shot* = global fit only. GRU = study 02 r1's 3-way GRU at D = 614, mean of 3 seeds. Bold = each emission's best selection score. One seed per GLM-HMM cell: no error bars.

| emission | K | selection | held-out matched | zero-shot | L/R engaged | engage | ignore PR-AUC |
|---|---|---|---|---|---|---|---|
| hurdle | 1 | 0.6227 | 0.6081 | 0.5997 | 0.7187 | 0.8188 | 0.6020 |
| hurdle | 2 | 0.6278 | 0.6139 | 0.6086 | 0.7195 | 0.8259 | 0.6139 |
| hurdle | 3 | **0.6289** | 0.6148 | 0.6139 | 0.7207 | 0.8258 | 0.6136 |
| hurdle | 4 | 0.6284 | 0.6143 | 0.6153 | 0.7203 | 0.8255 | 0.6122 |
| reference | 1 | 0.6228 | 0.6079 | 0.5996 | 0.7191 | 0.8182 | 0.5984 |
| reference | 2 | **0.6262** | 0.6119 | 0.6088 | 0.7189 | 0.8238 | 0.6072 |
| reference | 3 | 0.6261 | 0.6116 | 0.6138 | 0.7194 | 0.8228 | 0.6048 |
| GRU H64 (study 02) | — | — | 0.6249 | — | 0.7282 | 0.8316 | 0.6320 |
| GRU H128 (study 02) | — | — | 0.6273 | — | 0.7303 | 0.8326 | 0.6362 |
| GRU H256 (study 02) | — | — | 0.6288 | — | 0.7315 | 0.8334 | 0.6389 |

**Lag scan** (K = 1, selection score; L = 15 from the follow-up launch):

| emission | L = 3 | L = 5 | L = 7 | L = 10 | L = 15 |
|---|---|---|---|---|---|
| hurdle | 0.6152 | 0.6201 | 0.6217 | 0.6227 | 0.6230 |
| reference | 0.6164 | 0.6210 | 0.6223 | 0.6228 | 0.6223 |

![scores](../fig_r1_scores.svg)
<!-- END result-1 -->

<!-- BEGIN result-2 -->
**hurdle, K = 3** (run [u4j7ukti](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/u4j7ukti)). States ordered by occupancy: the mean smoothed posterior over the held-out mice's trials (both splits). Dwell: expected 1/(1 − A_kk) from the global transitions, and the mean Viterbi run. *Predicted*: the global model's p(left, right, ignore), averaged over the state's posterior weight. *Observed*: rates on the trials Viterbi assigns to the state.

| state | occupancy | mature / non-mature / n.a. | dwell exp. / obs. | predicted L / R / I | obs. p(ignore) | obs. p(R \| resp) | stay after rew. / no rew. | p(better side) | sessions using ≥10% |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.500 | 0.57 / 0.47 / 0.41 | 331 / 314 | 0.48 / 0.50 / 0.03 | 0.03 | 0.51 | 0.96 / 0.89 | 0.57 | 0.70 |
| 2 | 0.265 | 0.22 / 0.36 / 0.27 | 472 / 180 | 0.35 / 0.35 / 0.30 | 0.31 | 0.50 | 0.93 / 0.85 | 0.59 | 0.55 |
| 3 | 0.236 | 0.21 / 0.18 / 0.32 | 210 / 210 | 0.47 / 0.50 / 0.02 | 0.03 | 0.52 | 0.81 / 0.71 | 0.57 | 0.39 |

![hurdle weights](../fig_r1_weights_hurdle.svg)

![hurdle states](../fig_r1_states_hurdle.svg)

![hurdle example sessions](../fig_r1_examples_hurdle.svg)

**reference, K = 2** (run [74wl3w23](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/74wl3w23)). States ordered by occupancy: the mean smoothed posterior over the held-out mice's trials (both splits). Dwell: expected 1/(1 − A_kk) from the global transitions, and the mean Viterbi run. *Predicted*: the global model's p(left, right, ignore), averaged over the state's posterior weight. *Observed*: rates on the trials Viterbi assigns to the state.

| state | occupancy | mature / non-mature / n.a. | dwell exp. / obs. | predicted L / R / I | obs. p(ignore) | obs. p(R \| resp) | stay after rew. / no rew. | p(better side) | sessions using ≥10% |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.641 | 0.70 / 0.52 / 0.63 | 509 / 360 | 0.48 / 0.50 / 0.02 | 0.02 | 0.51 | 0.91 / 0.83 | 0.57 | 0.82 |
| 2 | 0.359 | 0.30 / 0.48 / 0.37 | 598 / 225 | 0.37 / 0.38 / 0.25 | 0.25 | 0.51 | 0.93 / 0.86 | 0.59 | 0.64 |

![reference weights](../fig_r1_weights_reference.svg)

![reference states](../fig_r1_states_reference.svg)

![reference example sessions](../fig_r1_examples_reference.svg)
<!-- END result-2 -->

<!-- BEGIN result-3 -->
*Restarts at best*: final log joint within 0.0001 nats/trial of the best restart. *Aligned subjects*: per-mouse refits whose states match the global states, by weight distance, in the same order.

| emission | K | restarts at best | median gap (nats/trial) | restarts converged | EM iters (best) | aligned subjects | run time (h) |
|---|---|---|---|---|---|---|---|
| hurdle | 1 | 1/1 | 0.0e+00 | 1/1 | 2 | 149/149 | 0.1 |
| hurdle | 2 | 20/20 | 1.7e-05 | 20/20 | 30 | 147/149 | 0.3 |
| hurdle | 3 | 20/20 | 9.0e-06 | 20/20 | 32 | 141/149 | 0.4 |
| hurdle | 4 | 2/20 | 4.5e-04 | 20/20 | 48 | 122/149 | 1.3 |
| reference | 1 | 1/1 | 0.0e+00 | 1/1 | 2 | 149/149 | 0.1 |
| reference | 2 | 20/20 | 6.6e-07 | 20/20 | 34 | 148/149 | 1.2 |
| reference | 3 | 20/20 | 5.9e-06 | 20/20 | 27 | 132/149 | 1.4 |

![restarts](../fig_r1_restarts.svg)
<!-- END result-3 -->

## What it says

TODO once every run is in.

## Next steps

TODO once every run is in.
