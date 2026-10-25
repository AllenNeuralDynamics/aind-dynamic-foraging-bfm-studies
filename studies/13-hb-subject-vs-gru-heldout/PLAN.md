# Plan — study 13

Design settled in a grill session on 2026-10-25 (issue #27). Vocabulary: `CONTEXT.md`;
decisions and their reasons: `adr/`.

## Question and answer shape

Where does a two-level HB, conditioned on each held-out subject's context sessions, land
between the per-subject MLE baseline (the same forager, no between-session spread) and the
GRU? One trial-weighted pooled likelihood per forager, against the GRU's D curve.

## Fixed by design

| | choice | why |
|---|---|---|
| model | two-level HB per held-out subject, no population, training subjects unused | ADR 0001 |
| hyperprior | fixed, as `hattori2019_two_level`; all parameters vary by session | ADR 0003 |
| cohort and split | study 08's: snapshot `20260603`, `min_sessions=10`, `heldout_every_n=5`, context sessions per `eval_every_n=2`, scored on the eval and test held-out sessions | the comparators were scored on it |
| headline score | session-marginal likelihood, with Monte Carlo error; trial-marginal beside it | ADR 0002 |
| compute | Beaker GPU, all held-out subjects in one batched fit | fits are independent and already vectorise |

## Variants

| variant | forager | needs first |
|---|---|---|
| `hattori` | HB-Hattori2019: `learn_rate_rew`, `learn_rate_unrew`, `forget_rate_unchosen`, `softmax_inverse_temperature`, `bias_l` | session-marginal scorer |
| `ctt` | HB-CompareToThreshold, as the per-subject MLE CTT baseline: one learning rate, threshold, softmax inverse temperature, bias, no choice kernel | JAX CTT likelihood with a parity test against `ForagerCompareThreshold`, its two-level model, and the scorer |

## Comparators (read, never retyped)

- Per-subject MLE Hattori and CTT: `studies/util/cross_study.rl_baseline("hattori" | "ctt")`
  (study 01 r8, `pooled_likelihood_trial_weighted`, 149 subjects).
- GRU by D: `studies/util/cross_study.gru_h128_curve()` (study 01 r1); its seed spread is the
  yardstick for whether a gap means anything.
- Study 08's trial-marginal rungs, for continuity only.

## Convergence

Study 08's per-subject gates (R-hat, ESS, divergences). A subject that fails is refit once with
longer chains and the centred parameterisation. The headline covers every held-out subject;
a second number excludes subjects still failing. No subject is dropped silently, because the
comparators score all of them.

## Build order

1. foraging-models: session-marginal scorer for the two-level model, with a test that it equals
   the exact predictive on a case small enough to integrate.
2. foraging-models: a fit entry point for two-level HB on held-out subjects only (fixed
   hyperprior in place of the population posterior that `fit_adaptation_batched` takes).
3. `hattori` manifests; smoke on a few subjects; launch.
4. foraging-models: CTT likelihood, parity test, two-level model; then `ctt`.
5. Report against the comparators.
