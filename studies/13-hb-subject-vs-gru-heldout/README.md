---
schema_version: 1
id: 13-hb-subject-vs-gru-heldout
question: Does a two-level hierarchical Bayesian cognitive baseline (sessions vary
  around each subject, no population level), conditioned only on each held-out subject's
  context sessions, match the GRU's held-out-subject likelihood?
status: planned
wandb:
  entity: AIND-disRNN
  project: mice_data_scaling
motivated_by:
- study: 08-hb-vs-gru-heldout
  note: the population-subject-session fit did not mix at D~99-614, so no HB rung was usable;
    does dropping the population level and fitting each held-out subject alone give a usable baseline?
issues:
- 27
tags:
  model: [cognitive/bh]
  question: [comparison]
---
# Study: 13-hb-subject-vs-gru-heldout

W&B project [`mice_data_scaling`](https://wandb.ai/AIND-disRNN/mice_data_scaling), shared with
studies 01 and 08.

**Question.** Does a two-level hierarchical Bayesian cognitive baseline (sessions vary around each subject, no population level), conditioned only on each held-out subject's context sessions, match the GRU's held-out-subject likelihood?

## Design

Each held-out subject gets its own two-level HB (`CONTEXT.md`): session-level parameters
drawn from that subject's distribution under a fixed hyperprior, with no population level and
no training subjects (Sue's original subject-session model). It is conditioned on the
subject's context sessions only, and its held-out sessions are scored by marginalising their
unseen session-level parameters. The cohort and split are study 08's, so the numbers sit next
to the GRU (`auto_heldout_finetune`) and the per-subject MLE baselines.

| variant | model | status |
|---|---|---|
| [`hattori`](variants/hattori/notes.md) | HB-Hattori2019, all 5 parameters vary by session | not launched |
| [`ctt`](variants/ctt/notes.md) | HB-CompareToThreshold, as the per-subject MLE CTT baseline | blocked on a JAX CTT likelihood |

The headline score is the session-level marginal (log-mean-exp of the whole-session likelihood
over posterior draws, with its Monte Carlo error), pooled per trial. Study 08's per-trial
marginal is reported alongside it for continuity. The question is answered by where each
variant lands between the per-subject MLE (no session variability) and the GRU.

Plan and build order: [PLAN.md](PLAN.md); decisions and why: [adr/](adr/); terms: [CONTEXT.md](CONTEXT.md); issue #27.

## Deviations from study-conventions

- **Shared W&B project.** The convention is one project per study. Like study 08, this study
  logs into study 01's `mice_data_scaling` on purpose, so its held-out numbers sit beside the
  GRU and HB runs they are compared with instead of needing a manual join across projects.

## Verdict

TODO once reports are live; also set `status` and `verdict` in the frontmatter.
