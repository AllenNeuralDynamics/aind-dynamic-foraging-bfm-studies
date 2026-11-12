# Plan — study 41

Set up on 2026-11-12 from issue #149. The question moved here from study 40 (its
`FUTURE_DIRECTIONS.md`, "Decision (Han, 2026-11-12)"). Inputs from study 40 were confirmed by its
session on 2026-11-12. No other study's number is retyped here: every comparator is read from
committed JSON.

## Why this study

Study 40 optimised one number, the pooled held-out likelihood, and that number went flat. Its r5 found
why the top programs tie: each add-on (satiety, engagement, experience, reward rate, bait,
win-stay/lose-shift) reliably helps some mice and hurts others. This mouse-specific part is real
(r5 reads it from split halves) but small, and nothing simple explains it yet. This study asks what
each mouse's strategy is, and why mice differ.

## What "understanding" means here

A compact, mechanistic account that:

1. **predicts** choices close to the predictable limit;
2. **generates** realistic behaviour in closed loop with the task;
3. is **identifiable**: each component has a model-free signature in the choices and is recoverable
   from simulated data;
4. **explains variation** between mice, sessions, task variants and stages of learning, through its
   parameters or strategy weights;
5. **generalises** to mice and tasks it was not fitted to;
6. later, **connects to physiology** through its latent variables.

**Priority.** Item 4 is the target, and it must replicate (item 5). Items 1 and 2 are constraints, not
targets: a model must stay near the best predictive score and behave like the mice, but a gain in
likelihood alone is not a result.

## Mouse tiers

Frozen in [`reference/tiers.json`](reference/tiers.json) by `analysis/freeze_tiers.py` (data freeze
snapshot `20260603`, manifest sha256 checked against study 40's cohort file), before any look.

| tier | mice | source | use |
|---|---|---|---|
| discovery | 296 | study 40's generation, validation, selection and monitor cohorts | free exploration; claims are found here |
| confirmation | 315 | the other training-pool mice, minus 3 excluded | only for a registered claim, with a limited number of looks |
| test | 149 | the freeze's held-out pool | scored once at the end, for continuity with studies 01, 12, 13 and 40 |
| later | — | mice recorded after the snapshot | the strongest test, when they exist |

The 3 excluded mice (764791, 808057, 823164) are in the training pool but are the GRU's extra test
subjects; no draw uses them. Nothing chosen in this study uses the test mice. The GRU's held-out list
differs from the freeze's held-out pool by 3 mice each way; study 40 put its headline on the 146 shared
mice, and a test score here follows that.

## Method

The study runs with **MEvolve's scientist mode** (`"mode": "scientist"`, MEvolve
[#87](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-mevolve/issues/87), defined in
MEvolve's CONTEXT.md and ADR 0007): an agent loop that looks where models fail, forms a hypothesis,
tests it, has a reviewer try to break it, and confirms it on unused mice. Its tools enforce the tiers
in code: no confirmation or test data outside `confirm`.

**Fitting.** The 2-level Laplace-EM as in study 40 (its ADR 0002 and 0008; MEvolve ADR 0004), with the
per-mouse fallback of MEvolve PR #82. Pin MEvolve `9d16704` or later. Most of study 40's search
programs did not converge within `max_em` 100 (r5); a mixture model has more weights, so every new
model gets a recovery test on simulated mice before it is fitted to real ones.

**Claims.** A claim is registered before it is confirmed. Each claim names its model, its model-free
signature in the raw choices, its recovery test, and the confirmation test with its pass rule.

**Notebook.** The agents write to a run directory on shared disk, outside any git checkout (they write
often). The run config names `reference/tiers.json` and its sha256; the harness refuses to start if
either digest or the freeze manifest does not match, or if two tiers share a mouse. After a run, a
`freeze-notebook` target (to come) copies the run's `agent/` directory, without the data views, into
`analysis/notebook/` and commits it: claims, reviews, looks, budget, episode summaries and notes, in the
format of MEvolve's ADR 0007. Looks are sealed: the agent never sees a look's result, so its claims
cannot adapt to the confirmation subjects. `confirm` writes each result to `<run_dir>/confirm/results.jsonl`,
outside the agent's directory; the same freeze copies that file, and the reports read results only
from it.

**The GRU map.** Confirmation compares models with the cross-validated GRU (studies #148), which gives
held-out GRU predictions for every mouse. Discovery can start before that map exists, from study 40's
frozen r5 scores and the data freeze.

## First hypotheses

1. **Fit quality or mechanism?** r5 scored both halves of a mouse's sessions with one fitted parameter
   set, so a fit that happens to suit a mouse counts in both halves. Refit each half separately and
   test again (variant `half-refit`).
2. **Mixture of agents.** A core model plus add-ons, with per-mouse weights on the core and on each
   add-on (0 = off). Study 40 has no factored core: this study writes the core and the add-on modules
   from the r5 programs (study 40 `analysis/individual/programs/`; the closest to a core are
   `arch28_cc3398d1.py`, the core plus win-stay/lose-shift, and the 50% floor program
   `simplify/simplify-50/programs/c1_cd994c73.py`).
3. **Switching weights.** The weights switch between hidden states, as in TBS v5's mixture-of-agents
   HMM (`/allen/aind/scratch/han.hou/tbs-foraging-v5/code/v5.py`, so far run on simulated mice only).
   Compare the states with study 14's GLM-HMM states (r3).
4. **Extreme mice.** Check the mice at the ends of each weight with study 15's behavioural
   fingerprints.

## The rest of the plan (study 40 `FUTURE_DIRECTIONS.md` §1)

| | analysis | from |
|---|---|---|
| first step | refit chosen programs on all discovery mice; keep per-mouse and per-session parameters and per-trial predictions (r5 kept only per-session log-likelihoods) | §1 |
| B1 | null-spread test: simulate each mouse without an add-on, refit with it, compare the null spread of the gain with the real spread | §1 B |
| B2 | which mechanisms each mouse needs: cross-validated per-mouse gain; random-effects model selection for the share of mice using each | §1 B |
| B3 | strategy types: factor or cluster a mice × programs score matrix | §1 B |
| B4 | reliability: early vs late, even vs odd sessions; a trait is read only once it is stable | §1 B |
| B5 | GRU embedding alignment: which parameters the GRU's subject embedding encodes | §1 B |
| B6 | biological covariates (sex, genotype, age, rig, date), from the lab's metadata | §1 B |
| C1–C4 | task variants within mouse, learning stages, the optimal agent per variant, where models fail | §1 C |
| D | rules for solid claims (above: tiers, recovery and null simulations, model-free signatures, questions written down first) | §1 D |

## Inputs read from other studies

| what | source | reader |
|---|---|---|
| study 40's cohorts and excluded mice | `40-cognitive-llm-discovery/reference/cohorts_seed0_monitor.json` | read by path in `analysis/freeze_tiers.py`; a `cross_study` reader to come |
| r5 per-session scores of 17 programs on the 100 monitor mice | study 40 `analysis/individual/sessions.json`, `analysis/individual/scores/` | `cross_study` reader to come |
| r5 programs | study 40 `analysis/individual/programs/` | read by path |
| cross-validated GRU predictions | studies #148 | when it is registered |
| GLM-HMM states | study 14 r3 | to come |
| behavioural fingerprints | study 15 | to come |

## Open questions (for Han)

- Do the robustness analyses across study 40's ensemble of programs (§1 A1–A3) belong here or in
  study 40? Study 40 plans no further reports.
- The W&B project for scientist-mode runs (the ADR leaves it to the study): this study's project, `dfbfm_41-strategy-profiles`, or
  to `dfbfm-mevolve` as study 40 did?
- How many looks at the confirmation tier, and how a look is counted.
</content>
</invoke>
