# 14-glmhmm-vs-gru-heldout — reports

| report | question | verdict |
|---|---|---|
| [r1 — GLM-HMM round 1: held-out score vs K, and what the states are](r1-round-1.md) | How well does a GLM-HMM predict held-out mice as K grows, against the 3-way GRU, and what are its states? | Below the GRU (best: hurdle K = 3, 0.6148 vs 0.6273); K depends on the scoring rule (refit peaks at 3, global fit rises to 10); states are session-long (draft) |
| [r2 — How each state spreads across subjects](r2-state-spread.md) | Is each GLM-HMM state one strategy shared across mice, or a label over different per-mouse strategies, and how does that change with K? | — (draft) |
