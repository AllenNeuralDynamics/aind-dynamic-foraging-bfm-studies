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
| cohort and split | study 08's: snapshot `20260603`, `min_sessions=10`, `heldout_every_n=5`; per held-out subject, even-indexed sessions are context and odd-indexed are scored (`eval_every_n=2`, wrapper `compute_train_eval_session_ids`) | the GRU fine-tune and per-subject MLE were scored on these same eval sessions (study 01 r8). The GRU's `heldout/test_likelihood` is a different quantity and is not a comparator |
| headline score | session-marginal likelihood by plain draw-averaging (64 session draws per posterior draw), with Monte Carlo error, weight ESS and a Pareto k̂ < 0.7 gate per session; trial-marginal beside it | ADR 0002 |
| W&B keys | `heldout/eval_likelihood` carries the headline; new keys carry the trial-marginal, SE and ESS. **No `heldout/test_likelihood`**: for the GRU that key is a zero-shot score, and this study has no population level, so nothing here matches it ([wrapper#107](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper/issues/107)) | keeps matched and zero-shot numbers apart |
| compute | Beaker GPU, all held-out subjects in one batched fit, 4 chains | fits are independent and already vectorise; chains make the per-subject R-hat gate possible |

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

Study 08's gates (`analysis/summarize_hb_posteriors.py`: max R-hat < 1.01, bulk ESS ≥ 400,
no divergences), applied per subject. 08 conditioned held-out subjects with one chain, so they
were never gated; this study runs several. A subject that fails is refit once with
longer chains and the centred parameterisation. The headline covers every held-out subject;
a second number excludes subjects still failing. No subject is dropped silently, because the
comparators score all of them.

## Build order

1. Done: session-marginal scorer ([foraging-models#78](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/pull/78))
   and the fixed-hyperprior multi-chain fit ([#79](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/pull/79)).
2. foraging-models: per-session Pareto k̂ (diagnostic only). Wrapper `heldout_two_level`
   estimator ([wrapper#108](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper/pull/108)).
3. `hattori` manifests; smoke on a few subjects; launch.
4. foraging-models: CTT likelihood, parity test, two-level model; then `ctt`.
5. Report against the comparators.
