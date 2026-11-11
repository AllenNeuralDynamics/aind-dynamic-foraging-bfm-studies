---
schema_version: 1
id: 40-cognitive-llm-discovery
question: Can LLM-driven evolutionary search over small JAX cognitive programs, with
  parameters that vary by subject and by session, find an interpretable model whose
  held-out likelihood on the 149 held-out mice closes the gap from the best hierarchical
  cognitive baseline (HB-CTT) to the GRU, and that generates realistic behaviour in
  closed loop with the task?
status: active
verdict: The search's best program closes 0.59 of the HB-CTT-to-GRU gap on test (CI 0.53-0.64); simplified
  programs keep 0.55 and 0.53; the GRU still matches most mouse-average curves best.
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
| [`gate4-recovery`](variants/gate4-recovery/notes.md) | gate 4, second half: 100 proposals on mice simulated from a hand-set program (two reward timescales plus a choice kernel), seeded with simplified programs, library off; the search must reach its own score |
| [`gate4-noseed`](variants/gate4-noseed/notes.md) | gate 4 control: the same search with no seed programs (template only, one island); does Claude invent the slow reward trace? |
| [`seeded-islands-smoke`](variants/seeded-islands-smoke/notes.md) | `seeded-islands` at 100 proposals on real mice: the smoke launch before the search |
| [`seeded-islands-cap12`](variants/seeded-islands-cap12/notes.md) | the smoke launch with looser program limits (12 parameters, 8 latent variables) |
| [`simplify-90`](variants/simplify-90/notes.md), [`simplify-75`](variants/simplify-75/notes.md), [`simplify-50`](variants/simplify-50/notes.md) | simplification: the lowest-effort rewrite of the search's best programs that keeps 90, 75 or 50% of its gain over Miller-RHG |

The answer is the fraction of the gap from HB-CTT to the GRU that the best program closes on the
test subjects, with its behavioural fingerprints and closed-loop match beside it.

![The procedure: (a) four cohorts of mice; (b) the search loop, with the score from the validation mice and Claude's feedback from the generation mice; (c) simplification, pre-registration on the selection mice and one score on the test mice, as the gap closed from HB-CTT to the GRU](analysis/fig_procedure.svg)

*The procedure ([`analysis/fig_procedure.svg`](analysis/fig_procedure.svg), `make procedure-figure`; an interactive
version is [`analysis/explainers/study40_procedure.html`](analysis/explainers/study40_procedure.html)).*

Plan and build order: [PLAN.md](PLAN.md) (settled in the design grill; amended by the ADRs); vocabulary:
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

Round 1 ([r2](analysis/reports/r2-search-curve.md)–[r4](analysis/reports/r4-simplification.md)):

- The search climbed +0.167 over Miller-RHG on the validation mice, nearly all in its first 500
  proposals. The early gain holds on 100 unused mice; the late gain does not.
- On the test mice, the search's best program (`satiety_engagement`) closes 0.59 of the gap from
  HB-CTT to the GRU (CI 0.53–0.64). Its mouse-average curves beat Miller-RHG's, but the GRU stays
  best on 8 of 9.
- Simplified programs keep 0.55 (75% floor) and 0.53 (50% floor) of the gap at a fraction of the
  complexity. No program met the 90% floor, so round 1 has three test programs, not four.
- Round 2 directions: [FUTURE_DIRECTIONS.md](FUTURE_DIRECTIONS.md).
