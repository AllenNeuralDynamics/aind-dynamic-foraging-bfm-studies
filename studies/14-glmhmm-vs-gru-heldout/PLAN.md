# Plan — study 14

Design settled 2026-10-26 (issue #48). Code:
- model: AllenNeuralDynamics/aind-dynamic-foraging-models#81, `glm_hmm` with the `[glmhmm]` extra;
- trainer: AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper#110, `GLMHMMTrainer`;
- config: AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher#202, `model=glmhmm`.

## Question and answer shape

Where does a GLM-HMM over left / right / ignore land against the 3-way GRU on the same
held-out trials, and how many states does it need? Answer shape: a K curve per emission of
the pooled held-out scores, beside study 02's GRU, then a description of the states.

## Fixed by design

| | choice | why |
|---|---|---|
| framework | dynamax 1.0.3 + `tfp-nightly==0.26.0.dev20261013`, pinned in `[glmhmm]` | the 2025 nightlies fail dynamax's HMM tests under numpy 2.4 (comment on #48) |
| emissions | `hurdle` (per state, engagement GLM on unsigned lags × side GLM on signed lags) and `reference` (per state, one multinomial over all lags, ignore = zero logit) | hurdle maps onto study 02's decomposition; reference is Faeze's and `ssm`'s parameterisation |
| regressors | RewC, UnrC (signed: +1 R, −1 L, 0 ignore), Ignored, Rewarded, lags 1..L, + bias | trial-indexed lags as the GRU sees them; no session-time or stage regressor |
| M-step | exact Newton with a Gaussian prior; Dirichlet MAP for π and A | dynamax's model class has no prior and takes 50 Adam steps |
| data | `mice_snapshot_scaling`, `ignore_policy=include`, `mature_only: false`, full training pool | identical to study 02's 3-way GRU; stage is retained (`current_stage_actual`) for a split of the score, not as an input |
| protocol | global fit (σ = 100, α = 1) on the training mice's train sessions → per-mouse MAP refit (σ = 2, α = 2, prior centred at 0), started from the global fit, on each mouse's context sessions → forward-filter score on its eval sessions | Ashwood et al. 2022's global → individual recipe, on the GRU's split |
| headline keys | `heldout/eval_likelihood` (raw 3-way), `heldout/final/eval_likelihood_LR_engaged`, `heldout/final/eval_likelihood_engage`, `heldout/final/engage_ignore_pr_auc`; zero-shot `heldout/zero_shot/*` on the same eval trials | the keys study 02's `wandb_keys.py` reads for the GRU, computed by the same functions. Zero-shot is kept off `heldout/test_likelihood`, which for the GRU covers every held-out session (as study 13) |
| pooling | `exp(sum(heldout/eval_log_lik_sum) / sum(heldout/num_eval_trials))` over runs | the exact pooled, trial-weighted normalized likelihood, on the comparators' probability scale (as study 13's `r1_hattori_pooled.py`) |
| K selection | per-mouse refits on the training mice, scored on their eval sessions (`glmhmm/select/eval_likelihood`) | never the held-out mice; never the global likelihood, which rewards states that encode between-mouse differences |
| state alignment | Hungarian match of each mouse's states to the global ones by posterior overlap (weights as a check), with match quality and occupancy by stage | for interpreting states only; the score does not depend on labels |

## Variants

| variant | grid | needs first |
|---|---|---|
| `lag-scan` | K = 1; L ∈ {3, 5, 7, 10} × {hurdle, reference}. L stops at 10: Su2022 history-regression weights are near zero by lag 10, so longer histories buy little and cost compute | merged and released code |
| `hurdle` | K ∈ 1..10, L from `lag-scan` | `lag-scan` result |
| `reference` | K ∈ 1..10, L from `lag-scan` | `lag-scan` result |
| `faeze-crosscheck` (diagnostic, later) | her setup: `mature_only: true`, her mice and sessions, L = 7, her K (up to 10), reference emission | from Faeze: design matrix, ssm settings (prior_sigma, transition_alpha, restarts, tolerance), mice and sessions, curricula, fitted parameters. Compare side contrasts (right − left) and class probabilities, not raw reference weights: in a state that almost never ignores, the common mode of the left and right logits is not identified (models recovery, `glm_hmm/recovery/RESULTS.md`) |
| `all-data-inference` (+ `-mature`, later) | every mouse and session; K, L, emission and priors from the benchmark | snapshot coverage and flags for ophys/ephys sessions; a wrapper switch for no held-out reservation |

`all-data-inference` serves later neural studies with per-trial smoothed and filtered
posteriors and Viterbi paths, keyed by (subject_id, ses_idx, trial) with stage. It reports no
`heldout/*` metrics, and nothing about it is tuned on neural data.

## Comparators (read, never retyped)

- Study 02 `nxd-3way`, held-out `heldout/final/*` by H and D (`analysis/scaling.json`), through
  a reader added to `studies/util/cross_study.py` when the r1 producer is written.
- For context only: the 2-way per-mouse MLE and study 13's HB score left/right on ignore-free
  histories, so they are not the same information as LR_engaged here.

## Compute

HPC CPU, one W&B run per (K, emission, L) cell, `--sbatch-extra "--cpus-per-task=8 --mem=64G
--time=48:00:00"`. These estimates come from the models PR's scale probe (1M trials, K = 10,
L = 5: about 5–6 s per EM iteration, under 2 GB). A full-pool global fit (about 4.5M train trials) at
K = 10 is roughly 25 s per iteration, about 7 h for 10 restarts, plus about 2–3 h of
per-mouse refits.

## Build order

1. Code PRs merged (models → wrapper → dispatcher); models released; `[glmhmm]` installed in
   both HPC envs.
2. Smoke: `lag-scan` at one cell on a few mice (`data.test_subject_ids`, small
   `data.subject_ratio`), recorded under `launch_record/smoke/`.
3. `lag-scan` launch → fix L (best `glmhmm/select/eval_likelihood` at K = 1).
4. `hurdle` and `reference` K grids at that L.
5. Reports: r1 held-out score vs K for both emissions against study 02; r2 what the states
   are (aligned weights, A, dwell times, occupancy by stage).
   In r2, report each state's p(ignore) and p(right | respond), not raw logits: an engagement or common-mode weight of a state that never ignores is bounded, not identified.
