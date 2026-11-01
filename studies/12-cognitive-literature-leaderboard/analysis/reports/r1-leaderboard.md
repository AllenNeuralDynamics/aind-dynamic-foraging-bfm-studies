---
schema_version: 1
id: r1
slug: leaderboard
status: draft
authors:
- han
wandb_groups: []
inputs:
  script: analysis/r1_leaderboard.py
  freeze: analysis/freeze_fits.py
upstream:
- study: 01-gru-scaling-law
  report: r1
- study: 01-gru-scaling-law
  report: r2
- study: 01-gru-scaling-law
  report: r8
- study: 13-hb-subject-vs-gru-heldout
  report: r5
reproduce: make -C studies/12-cognitive-literature-leaderboard r1
tags:
  model: [cognitive]
  question: [comparison]
---
# r1 — Literature models on the held-out subjects: the leaderboard

**Question.** Fit per subject on our subjects, how well does each literature model predict
held-out sessions, and how far is the best one from the hierarchical cognitive baselines
(HB-CTT, HB-Hattori) and from the GRU?

**Method.** Each of the 20 literature models ([CONTEXT.md](../../CONTEXT.md)) gets one parameter
set per subject, fit by differential evolution on the subject's even sessions and scored on its
odd sessions (study 01 r8's protocol). A score pools over subjects: exp(Σ log p / Σ n). The
leaderboard ranks 19 of them (Win-Stay-Lose-Shift is a strawman: fit, but not ranked) on the 149
held-out subjects, the cohort that HB-CTT and HB-Hattori (study 13) were
scored on. The GRU's held-out list differs by 3 subjects each way (study 13 r2), so the paired
comparison with the GRU uses the 146 subjects on both lists.

## Leaderboard

<!-- BEGIN result-1 -->
TODO: written by the producer, never by hand.
<!-- END result-1 -->

## Check: the re-fit reproduces study 01 r8

<!-- BEGIN result-2 -->
TODO: written by the producer, never by hand.
<!-- END result-2 -->

## What it says

TODO once the production launch is frozen.
