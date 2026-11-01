---
schema_version: 1
id: 12-cognitive-literature-leaderboard
question: How well do the models in the literature, each fit per subject on our subjects,
  predict held-out sessions, and how do they rank against the hierarchical cognitive
  baselines and the GRU?
status: planned
wandb:
  entity: NeuralDynamics
  project: aind-dynamic-foraging-model
motivated_by:
- study: 01-gru-scaling-law
  report: r8
  note: the GRU beats the per-subject Hattori2019, Bari2019 and CompareToThreshold fits on the 149 held-out subjects; these are the only literature models ever fit to our subjects
- study: 09-gru-cross-species-transfer
  note: the other literature models in aind-dynamic-foraging-models were fit only to external datasets, so their rank on our subjects is unknown
- issue: 73
  note: study 40 needs a leaderboard of literature models on our subjects to choose seed programs, and MEvolve needs per-subject fits as its parity reference
tags:
  model:
  - cognitive
  question:
  - comparison
issues:
- 77
---
# Study: 12-cognitive-literature-leaderboard

W&B project [`aind-dynamic-foraging-model`](https://wandb.ai/NeuralDynamics/aind-dynamic-foraging-model).

**Question.** How well do the models in the literature, each fit per subject on our subjects, predict held-out sessions, and how do they rank against the hierarchical cognitive baselines and the GRU?

## Design

Fit each literature model ([CONTEXT.md](CONTEXT.md)) once per subject, on all 763 cohort
subjects (614 training pool + 149 held-out), with study 01 r8's protocol, so every cohort a
later study draws has a score.

| | |
|---|---|
| Fit | one parameter set per subject, differential evolution on even sessions, seed 0 |
| Score | odd sessions; pooled trial-weighted exp(Σ log p / Σ n), p clipped to [1e-10, 1 − 1e-10] |
| Data | `mice_snapshot_scaling`, snapshot 20260603, min_sessions 10, heldout_every_n 5, eval_every_n 2, ignore_policy exclude, mature_only false |
| Models | the 17 `ForagerCollection.FORAGER_PRESETS` plus `ForagerFeedbackDependentRLBias` and `ForagerFeedbackDependentRLBiasCK1` (19), with package-default bounds and initial states, as study 09 |
| Check | the re-fit Hattori2019, Bari2019 and CompareToThreshold reproduce study 01 r8's per-subject values |

| variant | what it varies |
|---|---|
| [`per-subject-de`](variants/per-subject-de/notes.md) | one W&B sweep over the 19 literature models, one run per model |

The answer is the leaderboard of literature models on the 149 test subjects (and on the 146 also
on the GRU's held-out list), with HB-CTT and HB-Hattori (study 13) and the GRU (study 01) as
reference lines. Study 40 reads the per-subject scores and fitted parameters through
`studies/util/cross_study_literature.py` (`literature_per_subject`, `literature_fitted_params`), and can
re-pool them over any subset of the 763, including the GRU's own test list (`gru_split`).

| regime | what | report | status |
|---|---|---|---|
| 1, per subject | one parameter set per subject (this variant) | r1 | in progress |
| 2, session level | each session's parameters partially pooled toward the subject's typical values (no population level); held-out sessions scored by the session marginal, as in study 13; all 17 presets on all 763 subjects | r2 | planned: computed with [MEvolve](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-mevolve)'s evaluator after its gate 1c passes, so the literature models and study 40's programs share one evaluator |

Study 40's requirements (issue #73, decision records 0005 to 0007) set the regime 2 design.

## Deviations from study-conventions

- **Shared W&B project.** Runs log to `NeuralDynamics/aind-dynamic-foraging-model`, which study 40
  also uses, not to a project of their own: the BFM stack is moving off a personal W&B account
  that is almost full (decided 2026-11-01). Groups (`per-subject-de@<launch_id>`) and
  `meta.study` keep the two studies apart.

## Verdict

TODO once reports are live; also set `status` and `verdict` in the frontmatter.
