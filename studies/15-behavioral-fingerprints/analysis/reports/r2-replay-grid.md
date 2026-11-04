---
schema_version: 1
id: r2
slug: replay-grid
status: draft
authors:
- han
issue: 94
tags:
  model:
  - cognitive
  question:
  - comparison
wandb_groups: []
inputs:
  script: analysis/r2_replay_grid.py
  data:
  - analysis/p1_replay.json
  - analysis/p1_glm.json
  - analysis/p1_replay_gaps.json
  figure: analysis/fig_r2_replay_grid.svg
  freezes:
  - analysis/freeze_p1_session_groups.py
  - analysis/freeze_p1_replay.py
  - analysis/freeze_p1_glm.py
  - analysis/freeze_p1_gaps.py
upstream:
- study: 12-cognitive-literature-leaderboard
  report: r1
- study: 12-cognitive-literature-leaderboard
  report: r3
reproduce: make -C studies/15-behavioral-fingerprints r2
---
# r2 — P1: replay fingerprints of the literature models

**Question.** When each literature model predicts the subjects' own held-out trials, how close
are its switch probabilities to the subjects' in each history bin, as a multiple of what the
subjects' finite trials allow?

**Design.** Study 12's 19 ranked models, with its fitted parameters per subject (r1) and as
2-level EM (r3), plus Win-Stay-Lose-Shift (per subject, a strawman) and a per-subject GLM with
the Su2022 design (fit here on the even sessions; `freeze_p1_glm.py`). Each model replays the
held-out (odd) sessions of the 149 test subjects (ADR 0007); a 2-level model replays with the
session marginal's predictions (ADR 0005). Each replay first reproduces study 12's stored
held-out score for that subject, so the grid describes the models study 12 ranked. The gap is
compared with the binomial floor (ADR 0008). Warm-up sessions are left out (ADR 0004).

<!-- BEGIN result-1 -->
TODO: written by the producer, never by hand.
<!-- END result-1 -->

## What it says

TODO
