---
schema_version: 1
id: 30-gru-vs-glmhmm-latents
question: Can a linear readout of the 3-way GRU's latents (study 02) recover the GLM-HMM's
  state posterior (study 14) trial by trial, beyond what the GRU's own predictions
  and simple choice/reward history carry, and how does that change with the number
  of states K?
status: active
wandb:
  entity: NeuralDynamics
  project: dfbfm_30-gru-vs-glmhmm-latents
motivated_by:
- study: 14-glmhmm-vs-gru-heldout
  report: r1
  note: the GLM-HMM explains the GRU's cohort and split with a few discrete states (engagement and
    history-reliance), but scores below the GRU; does the GRU carry those states, and what does it add?
- study: 02-gru-scaling-law-ignore
  report: r1
  note: the 3-way GRU predicts left/right/ignore best of all models, but nothing says which latent
    structure carries that; an interpretable model on the same trials gives named targets to look for
tags:
  model:
  - gru
  - glmhmm
  question:
  - interpretation
  - comparison
issues:
- 136
---
# Study: 30-gru-vs-glmhmm-latents

W&B project [`dfbfm_30-gru-vs-glmhmm-latents`](https://wandb.ai/NeuralDynamics/dfbfm_30-gru-vs-glmhmm-latents).

**Question.** Can a linear readout of the 3-way GRU's latents (study 02) recover the GLM-HMM's state posterior (study 14) trial by trial, beyond what the GRU's own predictions and simple choice/reward history carry, and how does that change with the number of states K?

First study of the 30s block: comparing embeddings and latents across models (issue #136).

## Design

The GRU (study 02, 3-way, H256 / D614) and the GLM-HMM (study 14, hurdle and reference, K = 2–10) were fitted
on the same cohort, split and trials. Study 30 trains no new model; it reads both and asks how they relate.

- **Target.** The GLM-HMM's one-step predictive state posterior, p(z_t | trials < t). Study 14's saved
  `filtered_k` includes trial t and is not used as a target.
- **Readout.** A softmax linear readout (multinomial logistic regression, soft labels) from the GRU latent that
  predicts trial t, cross-validated by mouse.
- **Controls.** The same readout on the GRU's output logits, on history features, on the untrained GRU and on
  trial number; shuffle nulls that swap sessions between mice or shift the target in time.
- **Descriptive views.** State logits, targeted dimensionality reduction, Rastermap.

| variant | what varies | status |
|---|---|---|
| [`training-mice-pilot`](variants/training-mice-pilot/notes.md) | 50 GRU-training mice, odd sessions, global GLM-HMM parameters; post hoc, no launch | done ([r1](analysis/reports/r1-pilot-decoding.md)) |

Planned next (see r1, Next steps): the 149 held-out mice with per-mouse refits, more GRU seeds, a one-hot 3-way
GRU, a nonlinear history decoder, tests of discreteness, and the embedding comparison.

Glossary: [`CONTEXT.md`](CONTEXT.md).

## Verdict

None yet. r1 (pilot, draft): on training mice, a linear readout of the GRU latents recovers the GLM-HMM
predictive state posterior at every K (within-session R² 0.72–0.88), including the engaged states that history
and the GRU's own output cannot separate, and shuffle nulls stay at or below 0.03. This shows shared variables,
not discrete states.
