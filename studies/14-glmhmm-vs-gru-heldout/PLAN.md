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

`all-data-inference` serves later studies of the ophys/ephys sessions with per-trial smoothed
and filtered posteriors and Viterbi paths, keyed by (subject_id, ses_idx, trial) with stage.
It reports no `heldout/*` metrics, and nothing about it is tuned on the ophys/ephys
recordings.

## Comparators (read, never retyped)

- Study 02 `nxd-3way`, held-out `heldout/final/*` by H and D (`analysis/scaling.json`), through
  a reader added to `studies/util/cross_study.py` when the r1 producer is written.
- For context only: the 2-way per-mouse MLE and study 13's HB score left/right on ignore-free
  histories, so they are not the same information as LR_engaged here.

## Compute

One W&B run per (K, emission, L) cell. Measured 2026-10-27 on this study's data:
- **Data:** 614 training mice, 9,227 train sessions, 4.89M trials.
- **Setup:** one restart from the K = 1 weights, on 8 CPU cores, a V100 or an A100.
- **Source:** the timing jobs recorded on #48.

| K = 10 cell | CPU (8 cores) | GPU |
|---|---|---|
| EM iteration, L = 5 | 8–13 s | about 3 s |
| EM iteration, L = 10 | about 25 s (hurdle), about 47 s (reference) | about 5 s |
| iterations to converge (1e-6 nats/trial), L = 5 | hurdle 317, reference 220 | same |
| global fit, 10 restarts, L = 5 | about 7–8 h | about 1.4–2.6 h |
| per-mouse refits (614 for K selection + 149 held-out) | about 1.8 h | slower on a GPU: keep on CPU |

- **Backend.** Until the CUDA solver pin lands
  (AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper#111), run on HPC CPU with
  `--sbatch-extra "--cpus-per-task=8 --mem=64G --time=48:00:00"`.
  - At L = 10 a K = 10 reference cell is about 30 h on CPU, so if `lag-scan` picks L = 10,
    wait for the GPU.
  - On Beaker, the templates use one GPU on the GLM-HMM image, which carries the pin
    (`beaker/glmhmm/Dockerfile`, AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper#111).
    That image has to be built first.
- **Iteration cap.** `max_em_iters` 300 stops the K = 10 hurdle fit 17 iterations short.
  - That costs 4e-5 nats/trial of prediction, but it compares restarts below their optimum.
  - Raise the cap to 500 before the K grids (AllenNeuralDynamics/aind-dynamic-foraging-models#82).
- **Wall-clock.** Lower K is cheaper, and cells run in parallel, so a grid's wall-clock is its
  K = 10 cell. `lag-scan` (K = 1) converges in about 2 iterations, so each of its cells is
  dominated by the refits.

## Build order

1. Code PRs merged (models → wrapper → dispatcher); models released; `[glmhmm]` installed in
   both HPC envs.
2. Smoke: `lag-scan` at one cell on a few mice (`data.test_subject_ids`, small
   `data.subject_ratio`), recorded under `launch_record/smoke/`.
3. `lag-scan` launch → fix L (best `glmhmm/select/eval_likelihood` at K = 1).
4. `hurdle` and `reference` K grids at that L.
5. Reports: r1 held-out score vs K for both emissions against study 02, and what the states
   are (aligned weights, A, dwell times, occupancy by stage); r2 how each state spreads
   across subjects (one GLM per {subject, state}, cluster scores vs K; #70).
   In state reports, report each state's p(ignore) and p(right | respond), not raw logits: an engagement or common-mode weight of a state that never ignores is bounded, not identified.
   Read states only as far as study 15 (#50) shows such states are recoverable. Study 15
   simulates from known GLM-HMMs on the foraging gym's uncoupled-baiting task, including a
   factorial of 8 designed strategies.
