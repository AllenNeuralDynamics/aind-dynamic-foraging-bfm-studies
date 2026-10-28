---
schema_version: 1
id: r2
slug: vs-animals-and-rl
status: draft
authors: [pck, claude]
wandb_groups: []
inputs:
  scripts:
    - analysis/c2_rl_rollouts.py
    - analysis/c1_impulse.py
    - analysis/c1_kernels.py   # Su2022 design (corrected 2026-10-19)
    - analysis/c1_figure.py
    - analysis/c2_compare.py
    - analysis/c3_decoders.py
    - analysis/c4_behaviour.py
    - analysis/c4e_signatures.py
    - analysis/report_r2.py
  data:
    - analysis/c2_rl_rollouts.json
    - analysis/c1_impulse.json
    - analysis/c1_kernels.json
    - analysis/c2_compare.json
    - analysis/c3_decoders.json
    - analysis/c4_behaviour.json
    - analysis/c4e_signatures.json
upstream:
  - study: 01-gru-scaling-law
    wandb_group: e4-e8-d614-source@20260906-195409
  - study: 01-gru-scaling-law
    wandb_group: rl-baseline-bari@20260713-010244
  - study: 01-gru-scaling-law
    wandb_group: rl-baseline-ctt@20260713-010305
  - study: 01-gru-scaling-law
    wandb_group: rl-baseline-hattori@20260713-010255
reproduce: make -C studies/50-gru-dynamics-embedding r2
---

# Result 2 — model C against the animals and the classical RL models (PLAN.md phase C)

**Scope.** Model C seed 0 on the training mice's eval split (4.34M trials, 614 mice), against
Study 01's per-mouse fits of three RL models — Bari (Q-learning + choice kernel), CTT
(compare-to-threshold) and Hattori (asymmetric learning rates) — replayed on exactly the same
trials. Each replay reproduces its run's logged eval likelihood for every mouse.

## Summary

1. **The GRU's history dependence matches the animals'; the RL models' do not.** The same
   logistic regression (Su2022: rewarded and unrewarded choice, 10 back) fit to the animals
   and to the GRU's predictions gives near-identical kernels — rewarded choice 1.807 vs 1.796
   one back, unrewarded 0.876 vs 0.871, largest reward-effect gap 0.009 over 10 lags, against
   0.12–0.23 for Hattori, CTT and Bari. Bari's unrewarded-choice effect collapses after one
   trial (1.01 → 0.13); Bari and Hattori keep a positive reward effect at long lags.
2. **The late negative reward weight is real in the regression but not a reversed response.**
   With 20 lags the reward effect turns slightly negative after ~9 trials (−0.01 to −0.03 at
   lags 9–11) and stays near −0.05 to −0.09 from about lag 12 out to 20, for animals and GRU alike (with 10 lags the last lag absorbs older history
   and reads −0.11). Flipping one reward in the GRU's input moves its log-odds by +1.00 next
   trial, decaying to +0.017 at 10 — positive on average and in the median at every lag, but
   increasingly state-dependent (negative in 3% of flips at k=1, 41% at k=10). Logistic
   weights are therefore not the GRU's causal kernel at long lags; the mechanism (task
   structure correlated with long reward history, nonlinear state dependence) is not isolated.
3. **The GRU is better than every RL model for every mouse**, by a median +0.012 likelihood
   over the strongest (CTT), while agreeing closely trial by trial (r = 0.98 on P(right)).
4. **Its edge grows where switching is plausible, but is spread widely.** Split by the
   predicted probability of switching (known before the choice), the per-trial advantage over
   CTT rises from ~0.01 nats when a switch is unlikely (P < 0.1) to ~0.03–0.04 at P(switch) 0.4–0.5;
   trials with P(switch) > 0.3 (11%) carry about a quarter of the total advantage. It also
   falls with the GRU's own confidence (0.040 → 0.005 nats).
5. **The RL models' value variables are inside the GRU state, on r1's axes:** ΔQ decodes at
   R² 0.95–0.97 on PC1; total value ΣQ (R² 0.77–0.87) on PC2. The RL variables explain 94% of
   the variance of the GRU's log-odds.
6. **Where the RL models fail, against the animals (C4, added 2026-10-23).**
   - **The GRU's policy updates match the animals' in every condition** (C4b: RMS ≤ 0.05
     log-odds). The RL models deviate by 0.05–0.42, each in its own conditions.
   - **Calibration by the last three inputs** (C4c): mean error GRU 0.18 pp, CTT 1.10,
     Bari 2.13, Hattori 2.24. The RL errors fall into two opposite failure modes, both
     around losses.
     - **Hattori, which has no choice kernel, underestimates persistence after losses.**
       After R−, R−, L− (most recent first: t−1, t−2, t−3), the animals choose R 69% of the time
       and Hattori predicts 53%.
       After L−, R−, R−, Hattori sends too many back to R (+15 pp).
     - **Bari and CTT underestimate the return after a failed switch.** After switching to
       R and losing (R−, L+, L−), the animals go back to L more than predicted (+14 to
       +17 pp). Their persistence weights the last choice too heavily against the longer
       choice history that the GRU integrates (r6 I7).
   - **Where the GRU and CTT disagree, the animals side with the GRU**, in both directions
     (C4d: animals on the GRU's diagonal).
     - The 5% largest disagreements hold 32% of the total advantage.
     - 55% of them are on the first trial after a switch (13% overall). They are also more
       common after a loss and later in the session.
     - In two thirds of them CTT predicts staying (0.72) where the GRU and the animals
       split evenly (0.50 vs 0.51). In the other third the GRU and the animals stay (0.84)
       where CTT predicts 0.65.
   - **Two behavioural signatures make the failure modes concrete** (C4e). The GRU follows
     the animals on all four curves (mean error ≤ 0.005). The tables below the figure give
     every point.
     - **Return after a failed switch grows with the length of the run left behind.** It
       rises from 0.39 after a 1-trial run to about 0.53 after 5–9 trials.
       - Hattori returns too often (0.47 → 0.62).
       - Bari returns too rarely (0.31 → 0.43).
       - CTT is flat at 0.41: it has no memory of how long the old run was.
       - After a *rewarded* switch, the animals return less (0.22 → 0.30). Hattori matches
         this, while Bari and CTT miss by 0.02–0.03.
     - **Staying through losses depends on how the streak began.** If it began after a
       reward on that side, P(stay) falls slowly (0.90 → 0.85) and every model is close
       (Bari falls too fast, to 0.80).
       - If it began right after switching to that side, P(stay) *rises* with each loss
         (0.56 → 0.84). Part of this is selection, since the trials that reach a long streak
         come from persistent mice and states, but the models are scored on the same
         trials.
       - Hattori misses this badly (0.47 → 0.58; error 0.14), and Bari is too flat
         (0.64 → 0.74). CTT follows it (error 0.02).
     - Together: the animals carry a slowly decaying memory of which side they have been
       choosing, and how long. Losses on a newly chosen side are weighed against that
       memory. Hattori has no such memory, Bari's choice kernel fits it only roughly
       (too little return after a failed switch, too flat through losses), and CTT's
       doesn't track run length after a switch.

**Corrections after review (2026-10-19).** An earlier draft (i) used the Hattori2019 design,
whose Choice term equals RewC + UnrC — rank-deficient, so the separate weights were set by the
L2 penalty; it reported an "unrewarded-choice dip" and that no RL model shows a negative late
reward effect, neither of which holds (CTT's is slightly negative); (ii) split the advantage by
whether the animal then switched, which conditions on the outcome being scored and gave "two
thirds of the advantage on switch trials" — with pre-choice splits it is about a quarter; and
(iii) attributed B5's persistence to the model's own later choices, which the environment
confound in r1 rules out.

<!-- BEGIN result-2 -->
[regenerated by `analysis/report_r2.py` — do not edit by hand]

### RL baselines replayed on the same trials
| model | agent | replayed pooled eval likelihood | logged | max per-mouse diff |
|---|---|---:|---:|---:|
| bari | ForagerQLearning | 0.718088 | 0.718088 | 4.4e-16 |
| ctt | ForagerCompareThreshold | 0.721041 | 0.721041 | 1.1e-16 |
| hattori | ForagerQLearning | 0.715922 | 0.715922 | 2.2e-16 |

### C1 History kernels

Su2022 design, 10 trials back, 4,250,591 trials; models fit with their P(right) as soft labels on the animal's history.

Rewarded-choice (β_RewC) and unrewarded-choice (β_UnrC) effects, 1–4 trials back:

| source | RewC 1 | 2 | 3 | 4 | UnrC 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| animal | +1.807 | +0.833 | +0.446 | +0.266 | +0.876 | +0.321 | +0.156 | +0.110 |
| gru | +1.796 | +0.838 | +0.454 | +0.269 | +0.871 | +0.317 | +0.162 | +0.110 |
| bari | +1.716 | +0.631 | +0.426 | +0.300 | +1.010 | +0.133 | +0.068 | +0.042 |
| ctt | +1.732 | +0.764 | +0.404 | +0.237 | +0.942 | +0.340 | +0.155 | +0.088 |
| hattori | +1.610 | +0.836 | +0.485 | +0.304 | +0.722 | +0.309 | +0.152 | +0.086 |

Reward effect (β_RewC − β_UnrC) at 1–5 and 8–10 trials back:

| source | 1 | 2 | 3 | 4 | 5 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| animal | +0.931 | +0.512 | +0.290 | +0.156 | +0.097 | -0.033 | -0.074 | -0.111 |
| gru | +0.925 | +0.521 | +0.292 | +0.159 | +0.092 | -0.036 | -0.071 | -0.114 |
| bari | +0.706 | +0.499 | +0.358 | +0.258 | +0.186 | +0.071 | +0.048 | +0.014 |
| ctt | +0.790 | +0.423 | +0.249 | +0.149 | +0.088 | +0.007 | -0.007 | -0.025 |
| hattori | +0.889 | +0.527 | +0.333 | +0.218 | +0.146 | +0.048 | +0.032 | +0.009 |
| GRU counterfactual (flip) | +1.004 | +0.596 | +0.346 | +0.221 | +0.160 | +0.056 | +0.034 | +0.017 |

With 20 lags the late reward effect stays negative: animal -0.024, -0.059, -0.089 and GRU -0.026, -0.060, -0.091 at 10 / 15 / 20 back.
Counterfactual flips with a negative effect: k=1: 3%, k=3: 3%, k=5: 8%, k=8: 22%, k=10: 41% (median effect at k=10 +0.013).

Largest |reward-effect − animal's| over lags 1–10: gru 0.009, bari 0.225, ctt 0.141, hattori 0.119.
Counterfactual: 40,000 flips; unedited replay vs A1 logits median |err| 0.0009.

![C1](../fig_C1_kernels.png)

### C2 GRU vs RL, trial by trial

| | GRU | Bari | CTT | Hattori |
|---|---:|---:|---:|---:|
| all | 0.7339 | 0.7181 | 0.7210 | 0.7159 |
| Uncoupled Baiting | 0.7419 | 0.7270 | 0.7301 | 0.7252 |
| Coupled Baiting | 0.6844 | 0.6665 | 0.6688 | 0.6652 |
| Uncoupled Without Baiting | 0.8110 | 0.7971 | 0.8017 | 0.7928 |
| Coupled Without Baiting | 0.7551 | 0.7404 | 0.7415 | 0.7342 |

| vs | mice where GRU better | median per-mouse gain | r(P(right)) | mean \|ΔP\| | advantage (nats/trial) |
|---|---:|---:|---:|---:|---:|
| bari | 100.0% | +0.0145 | 0.980 | 0.055 | 0.0217 |
| ctt | 100.0% | +0.0117 | 0.983 | 0.049 | 0.0176 |
| hattori | 100.0% | +0.0169 | 0.974 | 0.058 | 0.0247 |

Advantage over CTT by pre-choice predicted P(switch) (bins of 0.05):

| binned by | P(switch) < 0.1 | 0.2–0.3 | 0.4–0.5 | share of total advantage from P(switch) > 0.3 |
|---|---:|---:|---:|---:|
| GRU | +0.0123 | +0.0233 | +0.0337 | 27% |
| CTT | +0.0110 | +0.0269 | +0.0409 | 24% |

By GRU confidence |logit|: +0.0397 (|logit| < 0.25) → +0.0054 (most confident bin).

![C2a](../fig_C2a_likelihood.png)
![C2b](../fig_C2b_agreement.png)
![C2c](../fig_C2c_advantage.png)

### C3 RL variables in the GRU state

| variable | R² | share on PC1 | PC2 | PC4 |
|---|---:|---:|---:|---:|
| bari logit(right) | 0.967 | 0.88 | 0.00 | 0.02 |
| bari Q_L | 0.926 | 0.54 | 0.20 | 0.03 |
| bari Q_R | 0.926 | 0.56 | 0.12 | 0.00 |
| bari Q_R − Q_L | 0.948 | 0.74 | 0.00 | 0.02 |
| bari Q_R + Q_L | 0.870 | 0.00 | 0.62 | 0.01 |
| ctt logit(right) | 0.976 | 0.91 | 0.00 | 0.02 |
| hattori logit(right) | 0.968 | 0.91 | 0.00 | 0.02 |
| hattori Q_L | 0.945 | 0.80 | 0.08 | 0.03 |
| hattori Q_R | 0.938 | 0.83 | 0.03 | 0.01 |
| hattori Q_R − Q_L | 0.967 | 0.91 | 0.00 | 0.02 |
| hattori Q_R + Q_L | 0.772 | 0.00 | 0.48 | 0.01 |

GRU logit(right) from the RL variables (cross-validated R²): all RL variables 0.943, bari 0.922, ctt 0.941, hattori 0.926.

![C3](../fig_C3_q_decoders.png)

### C4 Behaviour against the animals

![C4a](../fig_C4a_logit_updates.png)

Update calibration (C4b): n-weighted RMS deviation, in log-odds, between each model's predicted and the animals' empirical next-choice log-odds, binned by the model's own logit(t):

| group | GRU | Hattori (L2F1) | Bari (L1F1_CK1) | CTT |
|---|---:|---:|---:|---:|
| R rewarded (stay) | 0.02 | 0.31 | 0.28 | 0.18 |
| R rewarded (switch) | 0.03 | 0.08 | 0.14 | 0.21 |
| R unrewarded (stay) | 0.02 | 0.33 | 0.22 | 0.05 |
| R unrewarded (switch) | 0.03 | 0.40 | 0.35 | 0.28 |
| L rewarded (stay) | 0.05 | 0.42 | 0.27 | 0.18 |
| L rewarded (switch) | 0.03 | 0.09 | 0.16 | 0.19 |
| L unrewarded (stay) | 0.03 | 0.32 | 0.21 | 0.06 |
| L unrewarded (switch) | 0.04 | 0.39 | 0.36 | 0.26 |

![C4b](../fig_C4b_update_calibration.png)

History calibration (C4c), trial-weighted mean |predicted − observed P(right)| over the 64 three-trial histories: GRU 0.18 pp, Hattori (L2F1) 2.24 pp, Bari (L1F1_CK1) 2.13 pp, CTT 1.10 pp. Largest RL errors (pattern t−1,t−2,t−3: pp):
- Hattori (L2F1): L−,L−,R− +16.8 (animals 32%); R−,R−,L− -16.2 (animals 69%); R−,L−,L− -15.5 (animals 59%); L−,R−,R− +14.6 (animals 43%)
- Bari (L1F1_CK1): L−,R+,R− -14.5 (animals 58%); R−,L+,L− +14.2 (animals 43%); R−,L+,L+ +10.7 (animals 43%); L−,R+,R+ -10.0 (animals 57%)
- CTT: R−,L+,L+ +16.7 (animals 43%); R−,L+,L− +16.5 (animals 43%); L−,R+,R− -16.1 (animals 58%); L−,R+,R+ -15.8 (animals 57%)

![C4c](../fig_C4c_history_calibration.png)

Top 5% of |p_GRU(stay) − p_CTT(stay)| (> 0.161; 216,525 trials) carry 32% of the GRU's total advantage over CTT (mean 0.113 nats/trial within the top 5%). When the GRU predicts more staying (34%): GRU 0.84, CTT 0.65, animals 0.84; when it predicts less: GRU 0.50, CTT 0.72, animals 0.51.

| feature | top-disagreement mean | all trials |
|---|---:|---:|
| run length of previous side (capped 10) | 2.553 | 6.700 |
| trials since reward (capped 10) | 1.946 | 1.774 |
| \|P(rew\|R) − P(rew\|L)\| | 0.416 | 0.390 |
| trial position in session | 0.537 | 0.501 |
| previous input R rewarded (share) | 0.187 | 0.223 |
| previous input R unrewarded (share) | 0.292 | 0.282 |
| previous input L rewarded (share) | 0.203 | 0.225 |
| previous input L unrewarded (share) | 0.318 | 0.270 |

![C4d](../fig_C4d_disagreement.png)

### C4e Two behavioural signatures

Trial-weighted mean |model − animals| in each curve:

| curve | GRU | Hattori (L2F1) | Bari (L1F1_CK1) | CTT |
|---|---:|---:|---:|---:|
| return after switch: failed | 0.005 | 0.093 | 0.083 | 0.049 |
| return after switch: rewarded | 0.002 | 0.004 | 0.020 | 0.034 |
| stay through losses: after a reward on that side | 0.001 | 0.010 | 0.018 | 0.006 |
| stay through losses: right after switching there | 0.004 | 0.138 | 0.058 | 0.020 |

return after switch — failed (x = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10):

- animals: 0.39, 0.46, 0.50, 0.51, 0.53, 0.53, 0.52, 0.51, 0.51, 0.44
- GRU: 0.39, 0.46, 0.49, 0.51, 0.52, 0.53, 0.52, 0.51, 0.52, 0.46
- Hattori (L2F1): 0.47, 0.55, 0.58, 0.60, 0.61, 0.62, 0.62, 0.62, 0.62, 0.62
- Bari (L1F1_CK1): 0.31, 0.36, 0.39, 0.41, 0.42, 0.43, 0.43, 0.43, 0.43, 0.41
- CTT: 0.41, 0.41, 0.41, 0.41, 0.41, 0.41, 0.41, 0.40, 0.40, 0.40

return after switch — rewarded (x = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10):

- animals: 0.22, 0.26, 0.28, 0.30, 0.30, 0.29, 0.30, 0.29, 0.28, 0.27
- GRU: 0.22, 0.27, 0.29, 0.30, 0.30, 0.30, 0.30, 0.30, 0.29, 0.27
- Hattori (L2F1): 0.22, 0.27, 0.28, 0.29, 0.30, 0.30, 0.30, 0.30, 0.30, 0.28
- Bari (L1F1_CK1): 0.19, 0.24, 0.26, 0.27, 0.27, 0.27, 0.28, 0.28, 0.28, 0.26
- CTT: 0.28, 0.29, 0.28, 0.28, 0.27, 0.27, 0.27, 0.26, 0.26, 0.24

stay through losses — after a reward on that side (x = 1, 2, 3, 4, 5, 6, 7, 8):

- animals: 0.90, 0.88, 0.87, 0.86, 0.86, 0.86, 0.85, 0.87
- GRU: 0.90, 0.88, 0.87, 0.86, 0.86, 0.85, 0.85, 0.87
- Hattori (L2F1): 0.90, 0.89, 0.88, 0.87, 0.87, 0.86, 0.85, 0.83
- Bari (L1F1_CK1): 0.89, 0.87, 0.86, 0.84, 0.83, 0.82, 0.82, 0.80
- CTT: 0.89, 0.88, 0.87, 0.87, 0.86, 0.86, 0.86, 0.86

stay through losses — right after switching there (x = 1, 2, 3, 4, 5, 6, 7, 8):

- animals: 0.56, 0.67, 0.72, 0.75, 0.77, 0.79, 0.79, 0.84
- GRU: 0.56, 0.67, 0.72, 0.75, 0.77, 0.79, 0.80, 0.85
- Hattori (L2F1): 0.47, 0.53, 0.55, 0.56, 0.56, 0.57, 0.57, 0.58
- Bari (L1F1_CK1): 0.64, 0.69, 0.71, 0.72, 0.72, 0.73, 0.73, 0.74
- CTT: 0.59, 0.66, 0.71, 0.74, 0.77, 0.79, 0.80, 0.83

![C4e](../fig_C4e_signatures.png)
<!-- END result-2 -->

## Reading the figures

- **Logistic vs counterfactual (C1).** Both hold the later history fixed but answer different
  questions: the regression asks which past trials *predict* the choice given the others; the
  flip asks what the GRU *does* when one input changes. They agree in shape over the first
  ~4 lags, with the flip 9–39% larger (lags 1–4; correlated regressors share credit), and diverge
  after, least in Coupled Without Baiting.
- **Soft labels.** Model kernels are fit to the model's P(right) on the animal's actual
  history, so animal–model differences are differences in the decision rule, not in history.
- **Advantage splits (C2c).** Only pre-choice quantities are used. The outcome-conditioned
  stay/switch split is kept in `c2_compare.json` (`…_OUTCOME_CONDITIONED`) for reference.
- **CTT stripes (C2b).** CTT outputs a small set of discrete P(right) values.

## Caveats

- One seed of one model; RL models fit by Study 01 (DE per mouse on its train sessions),
  scored on the same eval sessions as the GRU — whose best checkpoint was itself chosen on
  that split (it picked the final step; a negligible optimistic bias).
- The counterfactual is open-loop: later choices and rewards stay as observed. Its averages
  are trial-weighted (flips sampled uniformly over eligible trials), so long sessions and
  high-volume mice weigh more; the logistic kernels drop each session's first 10 trials.
- C1 ran on CPU (no GPU in the container); the unedited replay matches the A1 GPU logits to
  median 0.0009 and every flip is compared with a same-device baseline.
- CTT exposes no Q values, so C3's Q decoders cover Bari and Hattori only.
- C4 is pooled over mice. Each cell compares mean predicted with observed choices on the same
  trials, so it is calibration in each condition, not per mouse. The GRU's near-perfect
  calibration is on held-out sessions, but it is a 55k-parameter model fitted to these mice
  (against 3–6 parameters per mouse for RL), so the informative part is where and in which
  direction each RL model fails.
- The library's `Hattori2019` design in `aind-dynamic-foraging-models` is rank-deficient as
  encoded (Choice = RewC + UnrC); worth fixing or documenting upstream.

## What this suggests for D–F

- Phase D: PC1 carries the value difference and PC2 the total value / last outcome, so a
  fixed-point analysis per input should be read in those coordinates; the open-loop kernel
  (+1.00, 0.60, 0.35, 0.22 …) is the target the slowest Jacobian modes should reproduce.
- Phase E: the 6% of GRU log-odds variance the RL variables miss is the natural first
  subspace to isolate and ablate — it should carry the advantage where switching is plausible.
- Phase F: per-mouse logistic kernels for animal and GRU are already in `c1_kernels.json`
  (`per_mouse_coef`) for relating kernel shape to the subject embedding.
