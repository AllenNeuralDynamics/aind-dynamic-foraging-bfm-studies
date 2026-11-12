---
schema_version: 1
id: 17-gru-crossfit-predictability
question: How predictable is each mouse's behaviour, session by session and trial
  by trial, when the GRU never saw that mouse?
status: planned
wandb:
  entity: NeuralDynamics
  project: dfbfm_17-gru-crossfit-predictability
motivated_by:
- study: 01-gru-scaling-law
  note: The reference GRU (v2 H128) trained on all 614 training-pool mice, so only the 149 test
    mice have held-out GRU predictions, and every comparison with the GRU spends them.
- issue: 148
  note: Study 40 r5 found mouse-specific structure that a confirmation tier must test; study 41
    (#149) needs a held-out GRU ceiling and residual map for the training-pool mice.
tags:
  model:
  - gru
  question:
  - transfer
  - methods
issues:
- 148
---
# Study: 17-gru-crossfit-predictability

W&B project [`dfbfm_17-gru-crossfit-predictability`](https://wandb.ai/NeuralDynamics/dfbfm_17-gru-crossfit-predictability).

**Question.** How predictable is each mouse's behaviour, session by session and trial by trial, when the GRU never saw that mouse?

## Design

Train the study-01 reference GRU recipe in about 5 folds, each holding out a different set of
mice, so that every one of the 763 mice gets held-out GRU predictions. The existing 149-mouse
test split stays one fold as it is, so earlier results stay comparable. The output is a
per-trial, per-session and per-mouse predictability map, frozen and registered as a shared
reference for study 41 (#149) and MEvolve scientist mode.

The fold plan and the held-out protocol are **not fixed yet**. [`PLAN.md`](PLAN.md) separates
what was agreed from what is still open. No training starts until the open points are settled.

## Variants

| variant | what | status | W&B group |
|---|---|---|---|
| [`test-fold-ref`](variants/test-fold-ref/notes.md) | the test fold: the 149 test mice, held out from the study-01 reference GRU | planned | — |

## Verdict

TODO once reports are live; also set `status` and `verdict` in the frontmatter.
