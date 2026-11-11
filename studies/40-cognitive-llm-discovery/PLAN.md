# Plan — study 40

Design settled in a planning session and a design grill on 2026-10-31 and 2026-11-01 (issue #73).
Vocabulary: [CONTEXT.md](CONTEXT.md) and
[MEvolve's](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-mevolve/blob/main/CONTEXT.md);
decisions and their reasons: [adr/](adr/) here and MEvolve's
[docs/adr/](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-mevolve/tree/main/docs/adr).
No other study's number is retyped here: comparators are read from committed JSON.

## Question and answer shape

Can LLM-driven evolutionary search find a short, readable JAX program, fitted per subject and per
session, that closes the gap from the best hierarchical cognitive baseline (HB-CTT, study 13 r5) to
the GRU on held-out subjects, and that behaves like the mice in closed loop with the task?

The answer is four pre-registered programs (ADR 0007; round 1 delivered three, see Reports), each with its **gap closed** on the 146 shared
test subjects, its complexity, and its replay and rollout fingerprints beside the literature models'.

## Fixed by design

| | choice | record |
|---|---|---|
| objective | maximize pooled held-out likelihood on the validation subjects; behaviour never enters the score | ADR 0001 |
| behaviour | replay fingerprints for every program, in the feedback; rollout fingerprints at pre-registration; both gate pre-registration against the best literature model | ADR 0001, 0007 |
| fitting | the 2-level model (each session's parameters around the subject's typical values, no population level), fitted as 2-level EM (Laplace-EM), session facts as program inputs, every score split by stage group and task variant; fallback to one parameter set per subject | ADR 0002 |
| cohorts | generation and validation (48 each), 100 selection subjects (the seed-0 draft, adopted 2026-11-04), test = 149 held-out subjects with the headline on the 146 shared ones; one fit direction; stratified draws that exclude the GRU's three extra test subjects | ADR 0003 |
| parity | gates 1a, 1b, 1c with fixed tolerances | ADR 0004 |
| interpretable | at most 8 parameters and 6 latent variables, static checks, Halstead effort, floors at 90%, 75% and 50% of the gain over the best literature model | ADR 0005 |
| search | one search, 5 islands (template, top 3 literature models, Eckstein's Bayesian-inference model or the 4th), seed library = the literature models study 12 ranks | ADR 0006 |
| pre-registration | best program plus one per floor, each the best of 5 validation candidates on the selection subjects, scored once on test | ADR 0007 |
| engine | ShinkaEvolve, pinned, with a compatibility shim | MEvolve ADR 0001 |
| repos | tool in MEvolve, science here | MEvolve ADR 0002 |
| W&B | `NeuralDynamics/aind-dynamic-foraging-model`, own key in `~/.config/mevolve/` | MEvolve ADR 0003 |

## Inputs read from other studies

| what | source | reader |
|---|---|---|
| literature models, per subject (regime 1), fitted parameters | study 12 r1 | `cross_study.literature_per_subject`, `literature_fitted_params` |
| literature models as 2-level EM (regime 2) | study 12 r3 (`analysis/r3_2level_em.json`), computed with MEvolve's evaluator after gate 1c | study 12's regime 2 reader, to come |
| HB-CTT and HB-Hattori, per subject | study 13 r5 and r1 | read by path until the reference registry (issue #63) has entries |
| GRU v2 H128, per subject | study 01 | registry `gru_ceiling.v2_h128` once #63 merges |
| the GRU's own split | study 12 | `cross_study.gru_split` |

## Program interface (MEvolve)

```python
PARAMS = {"alpha_rew": (0.0, 1.0), "beta": (0.0, 100.0), ...}   # at most 8, named, with bounds
N_LATENT = 2                                                    # at most 6, reset each session
LATENT_NAMES = ("q_left", "q_right")
def init_latents(p, session): ...    # session facts: stage (-1 unknown), coupled, baited, index
def predict(latents, p): ...         # logits over [left, right]
def update(latents, choice, reward, p): ...
```
Parameters are fitted by the evaluator, never chosen by the LLM. A program with discrete states
carries its state probabilities among its latent variables.

## Gates (nothing downstream starts before a gate passes)

| gate | pass |
|---|---|
| 0 data | the split built through the wrapper's loader matches the committed subject lists and scored-trial counts |
| 1a dynamics | each literature program reproduces its reference forager: per-trial within 1e-6, pooled within 1e-5 |
| 1b fitter | gradient fits within 0.0002 of study 12's differential-evolution fits, pooled; 95% of subjects within 0.5 nats on context sessions |
| 1c 2-level EM | CTT and Hattori as 2-level EM within 0.001 of study 13's 2-level HB scores, their gap within 0.0005; else the documented fallbacks of ADR 0004 |
| 2 timing | Stage B under about 3 minutes; cohort sizes and the paired standard error for the stopping rule fixed here |
| 3 engine smoke (about $7) | Shinka in-process with the shim; no API errors; prompt caching works; cost within 5% of the Console; web UI works; 10 proposals at medium effort against 10 at high, keeping high only if clearly better per dollar |
| 4 recovery (≤ $20) | the 2-level EM fitter recovers simulated typical values and spreads; a short search reaches the generating program's score within 0.002 |

## Budget

| item | Claude spend |
|---|---|
| engine smoke | $7 |
| recovery | $20 |
| 100-proposal smoke launch | $20 |
| search (`seeded-islands`) | $450 |
| simplification, three floors | $120 |
| **total** | **about $617**, under the $650 Console workspace limit |

Dollar figures are list prices. After the organisation's discount the bill is about half of list
price (gate 3, [`engine-smoke` notes](variants/engine-smoke/notes.md)).

Compute (measured 2026-11-05, `seeded-islands-cap12`): one evaluation of a proposal (48 + 48 mice, 2-level EM)
is about 9,000 CPU-seconds; a search runs evaluations as Slurm steps of 24 CPUs on many nodes, about 406
CPU-minutes per proposal in all. The search's minimum of 2,000 proposals is about 13,500 CPU-hours (17 hours
on 32 slots); its whole budget would allow 13,000 to 18,000 proposals (88,000 to 122,000 CPU-hours).

## Build order

| step | what | needs |
|---|---|---|
| 1 | MEvolve: data freeze and split lists (gate 0) | — |
| 2 | MEvolve: evaluator, per-subject and 2-level EM fits, session marginal, session facts, split diagnostics, replay fingerprints (gates 1b, 1c, 2) | study 12 regime 1 fitted parameters |
| 3 | MEvolve: the ranked literature models (19 as of 2026-11-01, including LossCounting; not Win-Stay-Lose-Shift) as programs with parity tests (gate 1a) | study 12 configs (dispatcher PR 214) |
| 4 | MEvolve: Shinka integration, shim, seed placement on islands, launch wrapper (gate 3) | — |
| 5 | studies: `write_initial()` for MEvolve launches; recovery runs (gate 4); 100-proposal smoke launch | — |
| 6 | the `seeded-islands` search | study 12 scores for the seed choice |
| 7 | simplification at three floors; selection; rollouts; test scores; reports | study 12 r3 for the leaderboard |

## Reports

| report | content |
|---|---|
| r1 | program recovery on simulated subjects (gate 4), written first, so it took r1 (2026-11-04) |
| r2 | search curve: best validation score against proposals, mechanisms, and the gain on 100 unused (monitor) mice |
| r3 | the search's best on the selection and test subjects: gap closed, fingerprint gate, mouse-average curves |
| r4 | simplification at three floors: fit against complexity on the test subjects |

As delivered in round 1 (2026-11-11). The plan had r2 as the evaluator's parity with the reference
fits (gates 1a–1c); that report was not written, and the gates are recorded in MEvolve's parity tests
and issues (AllenNeuralDynamics/aind-dynamic-foraging-mevolve#9, #10, #11). The plan had four
pre-registered programs; the 90% floor had none under ADR 0008's rule, so round 1 has three.

## Risks

- Selection among thousands of programs inflates validation scores: handled by the selection cohort.
- The cheap session marginal is noisy for sessions far from a subject's typical values: common random
  draws, reported effective sample sizes, gate 1c.
- ShinkaEvolve is someone else's moving code: pinned commit, shim tests, the Console limit as backstop.
- 28% of sessions have no recorded stage: an explicit unknown level, never dropped.
- The task simulator uses default block schedules: rollouts are curriculum-matched, not session-matched.
- Autowater trials are not filtered by the loader: parity holds, but the physiology reading must allow
  for it.

## Later (not in round 1)

- A from-scratch search with no library in the prompt.
- Three choices (left, right, ignore), a GLM-HMM-style seed with states, and its comparators.
- Parameters that drift within a session (particle filter), timing inputs, latent variables carried
  across sessions.
- Search research on a Shinka fork: a Pareto archive, tempered islands, tree search.
- Physiology joins of the exported latent variables; reviewed programs ported back into
  `aind-dynamic-foraging-models`.
