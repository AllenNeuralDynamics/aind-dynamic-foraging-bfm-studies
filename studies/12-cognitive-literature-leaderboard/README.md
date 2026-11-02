---
schema_version: 1
id: 12-cognitive-literature-leaderboard
question: How well do the models in the literature, each fit per subject on our subjects,
  predict held-out sessions, and how do they rank against the hierarchical cognitive
  baselines and the GRU?
status: active
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
| Fit | one parameter set per subject, differential evolution on even sessions; DE is not seeded (the run's seed 0 sets only the agent's own RNG), so each fit is one DE run, as in study 01 r8 |
| Score | odd sessions; pooled trial-weighted exp(Σ log p / Σ n), p clipped to [1e-10, 1 − 1e-10] |
| Data | `mice_snapshot_scaling`, snapshot 20260603, min_sessions 10, heldout_every_n 5, eval_every_n 2, ignore_policy exclude, mature_only false |
| Models | the 17 `ForagerCollection.FORAGER_PRESETS`, `ForagerFeedbackDependentRLBias` and `ForagerFeedbackDependentRLBiasCK1`, and the free LossCounting model (20), with package-default bounds and initial states, as study 09. Win-Stay-Lose-Shift is fit but not ranked (a strawman with no lapse term) |
| Check | the re-fit Hattori2019, Bari2019 and CompareToThreshold reproduce study 01 r8's per-subject values |

| variant | what it varies |
|---|---|
| [`per-subject-de`](variants/per-subject-de/notes.md) | one W&B sweep over 19 literature models, one run per model, plus a second launch for LossCounting; production group [`per-subject-de@20261101-012148`](https://wandb.ai/NeuralDynamics/aind-dynamic-foraging-model/sweeps/3zwvpcce), running |
| [`ctt-credit`](variants/ctt-credit/notes.md) | CompareToThreshold in stay-credit coordinates (same likelihood, wider bounds), compared with per-subject-de's CompareToThreshold in r2 |

**Headline ranking: r3** (2-level EM; failed 2-level fits placed by their per-subject score), because the hierarchical baselines are 2-level too; r1 is the per-subject reference. The answer is the leaderboard of literature models on the 149 test subjects (and on the 146 also
on the GRU's held-out list), with HB-CTT and HB-Hattori (study 13) and the GRU (study 01) as
reference lines. Study 40 reads the per-subject scores and fitted parameters through
`studies/util/cross_study_literature.py` (`literature_per_subject`, `literature_fitted_params`), and can
re-pool them over any subset of the 763, including the GRU's own test list (`gru_split`).

| regime | what | report | status |
|---|---|---|---|
| 1, per subject | one parameter set per subject (variant `per-subject-de`) | r1 | done (draft): best Lopez-DoubleTrace 0.7187 |
| 2, 2-level EM (session → subject) | each session's parameters partially pooled toward the subject's typical values (no population level), fit by Laplace-EM under study 13's hyperprior, held-out sessions scored by the session marginal over 2,048 draws (MEvolve ADR 0004); all 19 ranked models on all 763 subjects, via variant [`two-level-em`](variants/two-level-em/notes.md) | r3 | done (draft): best EM-Lebedeva-PR 0.7243, above 2-level HB CTT |

Study 40's requirements (issue #73, decision records 0005 to 0007) set the regime 2 design.

## Deviations from study-conventions

- **Shared W&B project.** Runs log to `NeuralDynamics/aind-dynamic-foraging-model`, which study 40
  also uses, not to a project of their own: the BFM stack is moving off a personal W&B account
  that is almost full (decided 2026-11-01). Groups (`per-subject-de@<launch_id>`) and
  `meta.study` keep the two studies apart.

## Verdict

TODO once reports are live; also set `status` and `verdict` in the frontmatter.
