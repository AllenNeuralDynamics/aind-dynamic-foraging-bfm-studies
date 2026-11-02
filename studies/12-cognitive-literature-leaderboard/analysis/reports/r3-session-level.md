---
schema_version: 1
id: r3
slug: session-level
status: draft
authors:
- han
wandb_groups: []
inputs:
  script: analysis/r3_session_level.py
reproduce: make -C studies/12-cognitive-literature-leaderboard r3
---
# r3 — Literature models at the session level (Laplace-EM, regime 2)

**Question.** If each literature model gets session-level partial pooling within each subject
(regime 2), how do the models rank, and how much of the gap to the hierarchical baselines does
the extra flexibility close?

**Plan (not run yet).** All 19 ranked literature models on all 763 subjects, scored with MEvolve's
evaluator at its reported settings (MEvolve ADR 0004, MEvolve main 275158b): Laplace-EM within each
subject under study 13's hyperprior, EM run to convergence (max_em = 100), held-out sessions scored
by the session marginal over 2,048 draws. Gate 1c checks this evaluator against study 13's NUTS fits
for CompareToThreshold and Hattori2019. Study 12 launches it through a MEvolve CLI, as its own
variant with a launch record, and freezes per-subject rows. Because 17 of the 19 models have no HB
reference, this report gives three checks per model:
- median weight ESS of the session marginal;
- the share of subjects whose EM stopped at the iteration cap before converging;
- session-level score minus per-subject score (r1). A negative value flags a failed Laplace fit.

<!-- BEGIN result-1 -->
TODO: written by the producer, never by hand.
<!-- END result-1 -->

## What it says

TODO
