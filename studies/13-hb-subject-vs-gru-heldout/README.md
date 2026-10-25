---
schema_version: 1
id: 13-hb-subject-vs-gru-heldout
question: Does a subject-level hierarchical Bayesian cognitive baseline (sessions
  vary around each mouse, no population level), fit only on each held-out mouse's
  conditioning sessions, match the GRU's held-out-subject likelihood?
status: planned
wandb:
  entity: AIND-disRNN
  project: mice_data_scaling
motivated_by:
- study: 08-hb-vs-gru-heldout
  note: the population-subject-session fit did not mix at D~99-614, so no HB rung was usable;
    does dropping the population level and fitting each held-out mouse alone give a usable baseline?
issues:
- 27
---
# Study: 13-hb-subject-vs-gru-heldout

W&B project [`mice_data_scaling`](https://wandb.ai/AIND-disRNN/mice_data_scaling), shared with
studies 01 and 08.

**Question.** Does a subject-level hierarchical Bayesian cognitive baseline (sessions vary around each mouse, no population level), fit only on each held-out mouse's conditioning sessions, match the GRU's held-out-subject likelihood?

## Design

Each held-out mouse is fitted on its own, with no population level and no training mice:
the parameters of each session are drawn from that mouse's own distribution, under fixed weak
hyperpriors (Sue's original subject-session model). The fit uses the mouse's conditioning
sessions only (`eval_every_n=2`); its eval and test sessions are then scored by marginalising
the unseen session's parameters over the posterior. The cohort and split are study 08's, so the
numbers sit next to the GRU (`auto_heldout_finetune`) and the per-mouse MLE baselines.

| variant | model | status |
|---|---|---|
| [`hattori`](variants/hattori/notes.md) | Hattori2019, all 5 parameters vary by session | not launched |
| [`ctt`](variants/ctt/notes.md) | compare-to-threshold, config of the per-mouse MLE CTT baseline | blocked on a JAX CTT likelihood |

The headline score is the session-level marginal (log-mean-exp of the whole-session likelihood
over posterior draws, with its Monte Carlo error), pooled per trial. Study 08's per-trial
marginal is reported alongside it for continuity. The question is answered by where each
variant lands between the per-mouse MLE (no session variability) and the GRU.

Design decisions and their reasons: issue #27; PLAN.md to follow.

## Deviations from study-conventions

- **Shared W&B project.** The convention is one project per study. Like study 08, this study
  logs into study 01's `mice_data_scaling` on purpose, so its held-out numbers sit beside the
  GRU and HB runs they are compared with instead of needing a manual join across projects.

## Verdict

TODO once reports are live; also set `status` and `verdict` in the frontmatter.
