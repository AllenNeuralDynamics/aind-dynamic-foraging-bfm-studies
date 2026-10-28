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

**Question.** We fit a GLM-HMM to the GRU's own mice, sessions and metric code. How well does it
predict new sessions as the number of states K grows? Which of its two emissions does better?
How close does it get to the 3-way GRU? And what do its states look like?

**The model.** A GLM-HMM assumes the mouse is in one of K hidden *states* on every trial. In
each state it chooses left, right or ignore by its own logistic regression on the last
L = 10 trials. From one trial to the next it stays in its state or moves to another, by a
fixed *transition matrix* A.
- The **hurdle** emission splits each choice into two decisions. First, respond or not, from
  unsigned history (past ignores and rewards). Then left or right, from signed history (past
  rewarded and unrewarded choices).
- The **reference** emission is one three-way regression on all the history, with ignore as the
  baseline class. It is the form Faeze used.

**How it is fitted.** Snapshot 20260603, ignores included (study 02's 3-way setup).
1. **Global fit:** one model for all 614 training mice, on their even-numbered sessions
   (4,887,818 trials). EM is run from 20 random starts and the best fit is kept.
2. **Per-mouse refit:** each mouse's own model, started from the global fit and fitted to its
   even sessions. A prior keeps each weight near 0 (Gaussian, SD 2) and each transition row
   near uniform (Dirichlet, α = 2).

**Three scores.** Each predicts every trial of a mouse's odd-numbered sessions (never fitted)
from the trials before it, and averages the probability given to the mouse's actual choice
(a geometric mean; chance is 1/3).
- **Refit score:** training mice, after the per-mouse refit. This is study 14's selection
  score, and it chose L (lag scan) and K.
- **Global-fit score, same mice:** training mice, global fit only. This is how Faeze chose K.
- **Held-out scores:** the 149 held-out mice, with refit ("matched", the GRU's protocol) and with
  the global fit only ("zero-shot"). Reported, never used to choose.

**Comparator:** study 02 r1's 3-way GRU at D = 614, read from its committed `frozen_runs.json`.
Its raw 3-way score (≈ 0.63) is on the same scale as ours. Its headline 0.73 is `LR_engaged`,
left vs right on the trials the mouse responded, with chance 1/2.

**States.** For each held-out session, the *posterior* gives the probability of each state on
each trial, given the whole session. The *Viterbi path* is the single most likely state
sequence. The state figures use the per-mouse refits' posteriors. The refits start from the
global states, and result-3 checks that they keep the same order.

<!-- BEGIN result-1 -->
Each score is the geometric-mean probability given to the mouse's actual choice on sessions never fitted (chance 1/3). *Refit*: training mice, after the per-mouse refit (study 14's selection score; bold = each emission's best). *Global fit, same mice*: training mice, global fit only (Faeze's way of choosing K). *Held-out*: 149 new mice, with refit (matched, the GRU's protocol) and global fit only (zero-shot). GRU = study 02 r1's 3-way GRU at D = 614, mean of 3 seeds. One seed per GLM-HMM cell, so no error bars.

| emission | K | refit (selection) | global fit, same mice | held-out, refit | held-out, global fit | L/R engaged | engage | ignore PR-AUC |
|---|---|---|---|---|---|---|---|---|
| hurdle | 1 | 0.6227 | 0.6143 | 0.6081 | 0.5997 | 0.7187 | 0.8188 | 0.6020 |
| hurdle | 2 | 0.6278 | 0.6228 | 0.6139 | 0.6086 | 0.7195 | 0.8259 | 0.6139 |
| hurdle | 3 | **0.6289** | 0.6280 | 0.6148 | 0.6139 | 0.7207 | 0.8258 | 0.6136 |
| hurdle | 4 | 0.6284 | 0.6294 | 0.6143 | 0.6153 | 0.7203 | 0.8255 | 0.6122 |
| hurdle | 5 | 0.6278 | 0.6305 | 0.6136 | 0.6163 | 0.7200 | 0.8250 | 0.6110 |
| hurdle | 6 | 0.6266 | 0.6309 | 0.6121 | 0.6167 | 0.7195 | 0.8234 | 0.6051 |
| hurdle | 7 | 0.6262 | 0.6317 | 0.6115 | 0.6175 | 0.7188 | 0.8234 | 0.6057 |
| hurdle | 8 | 0.6252 | 0.6322 | 0.6104 | 0.6178 | 0.7184 | 0.8224 | 0.6029 |
| hurdle | 9 | 0.6243 | 0.6325 | 0.6095 | 0.6183 | 0.7179 | 0.8215 | 0.5998 |
| hurdle | 10 | 0.6241 | 0.6329 | 0.6095 | 0.6186 | 0.7178 | 0.8217 | 0.6020 |
| reference | 1 | 0.6228 | 0.6143 | 0.6079 | 0.5996 | 0.7191 | 0.8182 | 0.5984 |
| reference | 2 | **0.6262** | 0.6229 | 0.6119 | 0.6088 | 0.7189 | 0.8238 | 0.6072 |
| reference | 3 | 0.6261 | 0.6280 | 0.6116 | 0.6138 | 0.7194 | 0.8228 | 0.6048 |
| reference | 4 | 0.6243 | 0.6292 | 0.6099 | 0.6151 | 0.7181 | 0.8219 | 0.6000 |
| reference | 5 | 0.6231 | 0.6303 | 0.6085 | 0.6161 | 0.7171 | 0.8210 | 0.5975 |
| reference | 6 | 0.6209 | 0.6309 | 0.6062 | 0.6165 | 0.7160 | 0.8190 | 0.5911 |
| reference | 7 | 0.6194 | 0.6318 | 0.6046 | 0.6173 | 0.7149 | 0.8181 | 0.5888 |
| GRU H64 (study 02) | — | — | — | 0.6249 | — | 0.7282 | 0.8316 | 0.6320 |
| GRU H128 (study 02) | — | — | — | 0.6273 | — | 0.7303 | 0.8326 | 0.6362 |
| GRU H256 (study 02) | — | — | — | 0.6288 | — | 0.7315 | 0.8334 | 0.6389 |

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
| hurdle | 5 | 12/20 | 4.2e-05 | 20/20 | 66 | 94/149 | 1.9 |
| hurdle | 6 | 8/20 | 1.2e-04 | 20/20 | 109 | 31/149 | 2.9 |
| hurdle | 7 | 5/20 | 4.4e-04 | 20/20 | 180 | 42/149 | 4.1 |
| hurdle | 8 | 2/20 | 4.1e-04 | 20/20 | 295 | 20/149 | 5.2 |
| hurdle | 9 | 4/20 | 3.0e-04 | 19/20 | 281 | 6/149 | 7.8 |
| hurdle | 10 | 1/20 | 6.7e-04 | 18/20 | 371 | 6/149 | 9.5 |
| reference | 1 | 1/1 | 0.0e+00 | 1/1 | 2 | 149/149 | 0.1 |
| reference | 2 | 20/20 | 6.6e-07 | 20/20 | 34 | 148/149 | 1.2 |
| reference | 3 | 20/20 | 5.9e-06 | 20/20 | 27 | 132/149 | 1.4 |
| reference | 4 | 8/20 | 2.0e-04 | 20/20 | 65 | 116/149 | 2.6 |
| reference | 5 | 3/20 | 4.0e-04 | 20/20 | 136 | 75/149 | 4.4 |
| reference | 6 | 8/20 | 4.4e-04 | 20/20 | 384 | 51/149 | 7.8 |
| reference | 7 | 5/20 | 4.1e-04 | 20/20 | 126 | 27/149 | 9.4 |

![restarts](../fig_r1_restarts.svg)
<!-- END result-3 -->

## What it says

TODO once every run is in.

## Next steps

TODO once every run is in.
