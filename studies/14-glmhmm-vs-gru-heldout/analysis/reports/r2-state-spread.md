---
schema_version: 1
id: r2
slug: state-spread
status: draft
authors:
- han
wandb_groups:
- hurdle@20261027-212905
- reference@20261027-213006
inputs:
  script: analysis/r2_state_spread.py
  data: analysis/frozen_r2.json
  weights_hurdle: analysis/frozen_r2_weights_hurdle.npz
  weights_reference: analysis/frozen_r2_weights_reference.npz
  freeze: analysis/freeze_r2.py
reproduce: make -C studies/14-glmhmm-vs-gru-heldout r2
tags:
  model: [glmhmm]
  question: [interpretation]
---
# r2 — How each state spreads across subjects

**Question.** The global GLM-HMM gives each state one set of weights for all mice (r1). Is a
state one strategy that every mouse shares, or a label over strategies that differ from mouse
to mouse? If the states are shared, each mouse's own version of a state should sit near the
other mice's versions of the same state, and away from the other states. Does that hold, and
how does it change with K?

**Method.** For each emission and each K = 1–10 (the r1 K grids, L = 10):
1. **Assign trials to states.** The Viterbi path (the most likely state sequence) of every
   session under the global fit. The global fit gives a state the same meaning in every mouse;
   per-mouse refits reorder states (r1 result-3).
2. **One GLM per {subject, state}.** Take the subject's trials on that state's path. Fit the
   state's own emission GLM to them, with the per-mouse refit's prior (Gaussian, SD 2, centred
   at 0), starting from the global state's weights. History lags come from the whole session,
   so no lag crosses a cut between epochs.
3. **Two cohorts, shown apart.**
   - *Held-out mice* (149): the global fit never saw them.
   - *Training mice* (614): they shaped the global states, so some clustering is expected by
     construction.

   Every session of each mouse is used.
4. **Cluster scores.**
   - *Silhouette*: for each dot, the mean distance to the nearest other state's dots minus the
     mean distance to its own state's dots, divided by the larger of the two, then averaged. It
     is 1 for tight, separate clusters and 0 for clusters that overlap.
   - *Null*: the same score with the state labels shuffled among each mouse's own dots. This
     keeps the mouse-to-mouse spread and breaks only the state labels.
   - *η² by state*: the share of the weight variance that lies between state means.
     *η² by subject*: the same for subject means.
   - *k-means ARI*: cluster the dots into K groups without labels, then compare the groups to
     the states (0 = chance, 1 = the same partition).

<!-- BEGIN result-1 -->
TODO: written by the producer, never by hand.
<!-- END result-1 -->

<!-- BEGIN result-2 -->
TODO: written by the producer, never by hand.
<!-- END result-2 -->

## What it says

TODO after the full freeze.

## Next steps

1. **Q-learning parameters per {subject, state}** (deferred, #70): the same cells in the
   space of a Q-learning model's parameters.
