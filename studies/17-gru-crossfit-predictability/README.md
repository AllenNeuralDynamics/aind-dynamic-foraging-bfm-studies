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
  - scaling
  - methods
issues:
- 148
---
# Study: 17-gru-crossfit-predictability

W&B project [`dfbfm_17-gru-crossfit-predictability`](https://wandb.ai/NeuralDynamics/dfbfm_17-gru-crossfit-predictability).

**Question.** How predictable is each mouse's behaviour, session by session and trial by trial, when the GRU never saw that mouse?

## Design

Rotate 5 folds over all 763 mice: the freeze's 149 test subjects are one fold, and the 614
training-pool mice form 4 more. Each fold model, a 3-way GRU (study 02 recipe, H256), trains on
every mouse outside its fold and scores that fold's mice in two passes, so every session of every
mouse gets a held-out prediction. The test fold reuses study 02's `nxd-3way` H256 seed-0 model; the 4
pool folds are new trainings (Beaker on-prem H200; the full-size recipe does not fit a Cedar A100). The output is a per-trial, per-session and per-mouse
predictability map, frozen and registered as a shared reference for study 41 (#149) and MEvolve
scientist mode. Details and checks: [`PLAN.md`](PLAN.md); terms: [`CONTEXT.md`](CONTEXT.md).

## Variants

| variant | what | status | W&B group |
|---|---|---|---|
| [`test-fold-ref`](variants/test-fold-ref/notes.md) | test fold: rescore study 02's `nxd-3way` model (seed 0; seeds 1–2 for noise) in passes A and B | planned | — |
| [`pool-folds`](variants/pool-folds/notes.md) | 4 new fold models (H256), one per pool fold | training done (4 of 4, early stop at 90k) | `pool-folds@20261113-005356` (Beaker; earlier Cedar/Beaker launches in notes) |
| [`pool-folds-2way`](variants/pool-folds-2way/notes.md) | the 4 fold models as 2-way (ignored trials dropped), H256, for models that skip ignores | running (Beaker) | `pool-folds-2way@20261113-112505` (folds 1-3), `pool-folds-2way@20261113-015357` (fold 4); cancelled/failed copies in notes |
| [`scaling-folds`](variants/scaling-folds/notes.md) | cross-validated scaling curve: 16 models at D ~ 10, 30, 100, 300 per pool fold, plus the test fold at D=300 | training done (17 of 17) | `scaling-folds@20261113-005618` (D100), `scaling-folds@20261113-015432` (D10/30 Beaker), `scaling-folds@20261113-015941` + `@20261113-045923` (D10 folds 1-2 Cedar), `scaling-folds@20261113-015410` (D300), `scaling-folds@20261113-015420` (test D300) |
| [`cedar-smoke`](variants/cedar-smoke/notes.md) | first Cedar launch: the recipe cut to 300 steps on one A100; tests the platform, not a result | done (1st OOM, 2nd finished) | `cedar-smoke@20261112-175329`, `cedar-smoke@20261112-182835` |

## Verdict

TODO once reports are live; also set `status` and `verdict` in the frontmatter.
