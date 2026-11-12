---
schema_version: 1
id: 41-cognitive-strategy-profiles
question: Which strategies do individual mice use in dynamic foraging, and does a
  compact mechanistic model, through its per-mouse and per-session parameters or strategy
  weights, explain how behaviour varies between mice, sessions, task variants and
  stages of learning, in a way that replicates on mice it was not fitted to?
status: planned
wandb:
  entity: NeuralDynamics
  project: dfbfm_41-strategy-profiles
motivated_by:
- study: 40-cognitive-llm-discovery
  report: r5
  note: the top programs of study 40's search tie because each add-on reliably helps some mice and hurts others; this mouse-specific part is real but small, and nothing simple explains it
tags:
  model:
  - cognitive
  question:
  - interpretation
  - transfer
issues:
- 149
---
# Study: 41-cognitive-strategy-profiles

W&B project [`dfbfm_41-strategy-profiles`](https://wandb.ai/NeuralDynamics/dfbfm_41-strategy-profiles).

**Question.** Which strategies do individual mice use in dynamic foraging, and does a compact mechanistic model, through its per-mouse and per-session parameters or strategy weights, explain how behaviour varies between mice, sessions, task variants and stages of learning, in a way that replicates on mice it was not fitted to?

## Design

Each mouse's strategy profile, found with MEvolve's scientist mode
([MEvolve #87](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-mevolve/issues/87)): an
agent loop that forms hypotheses on discovery mice, registers each claim with its model and its
model-free signature, and confirms it on mice it never saw. The plan, the definition of
"understanding" and the mouse tiers are in [PLAN.md](PLAN.md); the tiers are frozen in
[`reference/tiers.json`](reference/tiers.json).

| variant | what it varies |
|---|---|
| [`half-refit`](variants/half-refit/notes.md) | study 40 r5's 17 programs, refitted on each half of every monitor mouse's sessions separately: is r5's mouse-specific part fit quality or mechanism? |

The answer is a set of replicated claims: which mechanisms each mouse uses, how their weights vary
between mice, sessions, task variants and stages of learning, and what remains unexplained.

## Verdict

TODO once reports are live; also set `status` and `verdict` in the frontmatter.
