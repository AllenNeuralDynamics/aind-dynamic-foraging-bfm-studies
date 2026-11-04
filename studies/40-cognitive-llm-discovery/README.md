---
schema_version: 1
id: 40-cognitive-llm-discovery
question: Can LLM-driven evolutionary search over small JAX cognitive programs, with
  parameters that vary by subject and by session, find an interpretable model whose
  held-out likelihood on the 149 held-out mice closes the gap from the best hierarchical
  cognitive baseline (HB-CTT) to the GRU, and that generates realistic behaviour in
  closed loop with the task?
status: planned
wandb:
  entity: NeuralDynamics
  project: dfbfm-mevolve
motivated_by:
- study: 13-hb-subject-vs-gru-heldout
  report: r5
  note: the best hierarchical cognitive baseline, HB-CTT, still trails the GRU on the held-out subjects; no hand-designed cognitive model has closed that gap
- study: 13-hb-subject-vs-gru-heldout
  report: r4
  note: inferring each held-out session's own parameters from its earlier trials covers most of the distance from the per-subject fit to the GRU, so programs here are fitted the same way
- study: 01-gru-scaling-law
  report: r8
  note: the GRU beats every per-subject classical fit on the 149 held-out subjects; this study keeps that protocol and its comparators
tags:
  model:
  - cognitive/mevolve
  question:
  - comparison
  - interpretation
issues:
- 73
---
# Study: 40-cognitive-llm-discovery

W&B project [`dfbfm-mevolve`](https://wandb.ai/NeuralDynamics/dfbfm-mevolve), shared with the MEvolve harness.
Runs logged before 2026-11-03 (`engine-smoke@20261103-030341`) stay in
[`aind-dynamic-foraging-model`](https://wandb.ai/NeuralDynamics/aind-dynamic-foraging-model).

**Question.** Can LLM-driven evolutionary search over small JAX cognitive programs, with parameters that vary by subject and by session, find an interpretable model whose held-out likelihood on the 149 held-out mice closes the gap from the best hierarchical cognitive baseline (HB-CTT) to the GRU, and that generates realistic behaviour in closed loop with the task?

## Design

An LLM (Claude) proposes short JAX programs; [MEvolve](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-mevolve)
fits each program's parameters per subject and per session, scores it on held-out sessions of
validation subjects, and feeds diagnostics back; ShinkaEvolve runs the evolutionary search. The
best programs are then simplified under a fit floor, scored once on the test subjects, and rolled
out in the task simulator to compare their behaviour with the mice's.

| variant | what it varies |
|---|---|
| [`seeded-islands`](variants/seeded-islands/notes.md) | one search, a different seed on each island: a near-empty template and literature models from the seed library |
| [`engine-smoke`](variants/engine-smoke/notes.md) | gate 3: the first ShinkaEvolve runs, 10 proposals each at effort medium and high, on a small separate cohort; tests the engine, not the science |
| [`gate4-recovery`](variants/gate4-recovery/notes.md) | gate 4, second half: 100 proposals on mice simulated from a new program in neither the seeds nor the library (Hattori2019 plus within-session satiety); the search must reach its own score |

The answer is the fraction of the gap from HB-CTT to the GRU that the best program closes on the
test subjects, with its behavioural fingerprints and closed-loop match beside it.

Plan and build order: [PLAN.md](PLAN.md) (draft, before the design grill); vocabulary:
[CONTEXT.md](CONTEXT.md) and MEvolve's
[CONTEXT.md](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-mevolve/blob/main/CONTEXT.md);
decisions and why: [adr/](adr/); issue #73.

## Deviations from study-conventions

- **Launches are not dispatcher launches.** MEvolve's launch wrapper writes each initial launch
  record (through `studies/util/launch_record.py`), not the dispatcher's launchers.
- **W&B entity.** Runs log to `NeuralDynamics`, not `AIND-disRNN`: the study's runs are too large for
  the personal account's storage cap.
- **W&B project.** New runs log to `dfbfm-mevolve`, the harness's project, which the study shares with
  MEvolve's own runs, rather than to a project of the study's own (Han, 2026-11-03; issue #99).

## Verdict

TODO once reports are live; also set `status` and `verdict` in the frontmatter.
