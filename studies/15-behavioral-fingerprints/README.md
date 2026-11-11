---
schema_version: 1
id: 15-behavioral-fingerprints
question: How closely does each model's behaviour, replayed on the subjects' real
  histories and rolled out in the task, match the subjects' model-free choice statistics,
  measured against the animals' own noise floor?
status: planned
wandb:
  entity: NeuralDynamics
  project: dfbfm_15-behavioral-fingerprints
motivated_by:
- study: 01-gru-scaling-law
  report: r9
  note: the GRU and three per-subject RL models were rolled out and compared on p(switch) by history; the statistics lived in the wrapper, took sampled choices only, used gym default task parameters, and the GRU and RL sides ran on different wrapper versions
- study: 12-cognitive-literature-leaderboard
  note: ranks 19 literature models by held-out likelihood on our subjects, per subject and as 2-level EM, but says nothing about how their behaviour compares with the subjects'
- issue: 73
  note: MEvolve gates pre-registration on replay and rollout fingerprints (study 40 ADR 0001), computed by its own code with bins and aggregation that differ from the wrapper's
tags:
  model:
  - cognitive
  question:
  - comparison
  - generative-run
  - methods
issues:
- 94
---
# Study: 15-behavioral-fingerprints

W&B project [`dfbfm_15-behavioral-fingerprints`](https://wandb.ai/NeuralDynamics/dfbfm_15-behavioral-fingerprints) (no runs yet).

**Question.** How closely does each model's behaviour, replayed on the subjects' real histories and rolled out in the task, match the subjects' model-free choice statistics, measured against the animals' own noise floor?

## Design

Put every model through the same model-free choice statistics, twice: replayed on the subjects'
real choices and rewards, and rolled out as an agent in the task. Decisions are in [`adr/`](adr/),
vocabulary in [CONTEXT.md](CONTEXT.md).

| | |
|---|---|
| Statistics | `fingerprints-v1`: p(switch) by the last 1 to 3 trials, by consecutive losses, by last reward × run length (the hazard), after a switch; run-length distributions; each by stage group × task variant (ADR 0004) |
| Modes | replay and rollout for every model; a grid, never one score (ADR 0001) |
| Rollouts | parameter-matched, in the gym, 10 per session; the headline uses uncoupled and final-stage sessions, because the gym cannot hold an early coupled block (ADR 0006); 2-level EM fits draw session parameters per rollout (ADR 0005) |
| Comparison | per statistic, the subject-balanced gap as a multiple of a noise floor: binomial in replay, split-half in rollout, so a perfect model scores about 1 in both (ADR 0003, 0008); one fingerprint score per model and mode, the geometric mean of its floor ratios, for leaderboards (ADR 0009) |
| Data | study 12's 149 test subjects, held-out (odd) sessions only; the 146 on the GRU's held-out list marked (ADR 0007) |
| Code | one library in `aind-dynamic-foraging-models`, behind MEvolve's program protocol (ADR 0002) |

| variant | models |
|---|---|
| [`literature-models`](variants/literature-models/notes.md) | study 12's 19 ranked models, per subject and 2-level EM; a per-subject GLM (Su2022 design) |
| [`gru-heldout-replay`](variants/gru-heldout-replay/notes.md) | study 01's v1 GRU at D = 614 (3 seeds), re-fine-tuned offline on the held-out subjects to save its per-trial predictions for replay |

Later: the GLM-HMM (study 14), with ignore statistics; MEvolve's pre-registered programs from
study 40. The GRU joined P1 and P2 (Han, 2026-11-04).

| phase | what | check |
|---|---|---|
| P0 | spec, library, golden tests; reward-pair pools; how often real coupled blocks run past `block_max` | reproduces the wrapper's statistics on study 05's cached RL rollout histories (r9's RL rows) and MEvolve's `replay_gap` on its gate-1b fits; done, [r1](analysis/reports/r1-p0-checks.md) |
| P1 | replay fingerprints of `literature-models` and the GRU (`gru-heldout-replay`) | each replay reproduces the held-out score study 12 stored for the subject (to 1e-6), and the GRU's study 01 r2's (to 1e-4); done, [r2](analysis/reports/r2-replay-grid.md) |
| P2 | rollout fingerprints of the same models and the GRU, and the noise floor, reported like r2 (the floor-ratio grid beside study 01 r9's per-bin scatter); a sensitivity check with the coupled task's holds on and off (ADR 0006) | the holds move the fingerprints less than the noise floor, or a new ADR turns them on; done (`BlockMinReward` ≤ 0.45 of the floor; ADR 0006 stands), [r3](analysis/reports/r3-rollout-grid.md) |
| P3 | GLM-HMM | |
| P4 | MEvolve imports the library (a proposal to its sessions) | MEvolve's `replay_gap` unchanged |

## Deviations from study-conventions

- **No model in the name.** The study measures every model family the same way (literature
  models, GLM, GLM-HMM, GRU, MEvolve programs), so the name has no `{model}` part; the
  frontmatter `tags.model` lists the families it fits (decided with Han, 2026-11-03).

## Verdict

TODO once reports are live; also set `status` and `verdict` in the frontmatter.
