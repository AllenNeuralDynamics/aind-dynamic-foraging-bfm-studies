---
schema_version: 1
id: r3
slug: 2level-em
status: draft
authors:
- han
wandb_groups: []
inputs:
  script: analysis/r3_2level_em.py
reproduce: make -C studies/12-cognitive-literature-leaderboard r3
---
# r3 — Literature models as 2-level EM (session → subject, Laplace-EM; regime 2)

**Question.** Fit as 2-level EM (regime 2: each session's parameters drawn from its subject's
distribution, no population level, fit by Laplace-EM), how do the literature models rank, and
how much of the gap to the 2-level HB baselines (study 13) does that structure close? Models are
labelled EM-<model> (EM-CTT, EM-Hattori2019, ...).

| label | levels | method | where |
|---|---|---|---|
| per subject | none, one parameter set per subject | MLE | r1 (regime 1) |
| 2-level EM | session → subject | Laplace-EM | this report (regime 2) |
| 2-level HB | session → subject | NUTS | study 13 (HB-CTT, HB-Hattori) |

**Plan (not run yet).** All 19 ranked literature models on all 763 subjects, scored with MEvolve's
evaluator at its reported settings (MEvolve ADR 0004, MEvolve main 275158b): Laplace-EM within each
subject under study 13's hyperprior, EM run to convergence (max_em = 100), held-out sessions scored
by the session marginal over 2,048 draws. Gate 1c checks this evaluator against study 13's NUTS fits
for CompareToThreshold and Hattori2019. Study 12 launches it through a MEvolve CLI, as its own
variant with a launch record, and freezes per-subject rows. Because 17 of the 19 models have no HB
reference, this report gives three checks per model:
- median weight ESS of the session marginal;
- the share of subjects whose EM stopped at the iteration cap before converging;
- 2-level EM score minus per-subject score (r1). A negative value flags a failed Laplace fit.

<!-- BEGIN result-1 -->
TODO: written by the producer, never by hand.
<!-- END result-1 -->

## What it says

TODO
