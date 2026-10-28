---
schema_version: 1
id: r8
slug: gru-vs-q-learning
status: draft
authors: [pck, claude]
wandb_groups: []
inputs:
  scripts:
    - analysis/h3_q_phase_portraits.py
    - analysis/h4_scenarios.py
    - analysis/h1_model_ladder.py
    - analysis/h2_learning_rate.py
    - analysis/report_r8.py
  data:
    - analysis/h3_q_phase_portraits.json
    - analysis/h4_scenarios.json
    - analysis/h1_model_ladder.json
    - analysis/h2_learning_rate.json
upstream:
  - study: 01-gru-scaling-law
    wandb_group: e4-e8-d614-source@20260906-195409
  - study: 01-gru-scaling-law
    wandb_group: rl-baseline-bari@20260713-010244
  - study: 01-gru-scaling-law
    wandb_group: rl-baseline-ctt@20260713-010305
  - study: 01-gru-scaling-law
    wandb_group: rl-baseline-hattori@20260713-010255
reproduce: make -C studies/50-gru-dynamics-embedding r8
---

# Result 8 — how the GRU's strategy differs from Q-learning (PLAN.md phase H)

**Question.** r2 (C4) showed where the RL models fail against the animals and r6 (I8) how
the GRU's carried value differs from ΔQ. Here the two are compared directly as dynamical
systems: in Q-learning's own coordinates (H3), and on identical scripted histories (H4).
H1 relaxes Q-learning one component at a time (model ladder); H2 measures each model's
effective learning rate as a function of state.

## Summary

1. **Read in Q-learning's coordinates, the GRU behaves like a delta rule after rewards and
   unlike any of the RL models after losses (H3).**
   - **Decoding.** Hattori's Q_L and Q_R decode from the GRU state at held-out R² 0.95 and
     0.94.
   - **After a reward.** The GRU's chosen value moves toward 1 at rate 0.42–0.43 and its
     unchosen value decays at 0.69–0.72 per trial. That is close to Hattori's 0.40–0.41
     and 0.75.
   - **After a loss, the RL models do two different things:**
     - Hattori leaves the chosen value almost untouched (slope −0.05, since α_unrew ≈ 0).
     - Bari pulls it toward about 0.05 at rate 0.25.
   - **The GRU does neither.** Its chosen value relaxes at rate 0.10–0.13 toward about 0.25–0.4,
     not toward 0, so a loss does not erase the value of the side just chosen. That update
     is only weakly linear (R² 0.39–0.47, against 0.86–0.93 for the GRU's other updates), so it depends on
     more than the two values.
   - **Unchosen side.** In all three models it is forgotten at similar rates (0.56–0.75).
2. **On identical histories the models diverge where C4 said they would (H4; fresh-session
   start, each mouse's context or fitted parameters, median across mice).**
   - **Return after a failed switch grows with the run left behind (sweep F).**
     - The GRU returns to the old side more the longer it stayed there: 0.46 after 1
       rewarded trial, saturating at 0.73 by about 7.
     - Hattori is high and flat (0.58 → 0.66).
     - Bari rises too slowly (0.36 → 0.63).
     - CTT is flat at 0.41.
   - **Staying through losses on a new side (sweep G).** After 6 R+ and then k losses on L,
     the GRU first goes back (P(stay on L) 0.28 after one loss). But each further
     unrewarded stay *raises* the chance of staying, to 0.70 by k = 3 and 0.80 by 10.
     Unrewarded choices build the GRU's choice trace (r6 I8).
     - Hattori barely rises (0.34 → 0.49): no choice trace.
     - Bari saturates lower (0.44 → 0.70).
     - CTT has the shape, but starts too high (0.59 → 0.80).
   - **The first trial after a switch.**
     - After a *failed* switch, P(R) is GRU 0.73, Hattori 0.66, Bari 0.59, CTT 0.41.
     - After a *rewarded* switch, P(R) is GRU 0.43, against 0.23–0.40 for the RL models.
     - So the GRU discounts a single outcome on a new side more than any RL model does.
   - **Alternating outcomes on one side (scenario D).** The GRU's policy is about 3× smoother
     than Bari's and CTT's (mean |ΔP| per trial 0.007, against 0.010 Hattori and 0.024
     Bari / CTT). Near saturation this is a small absolute effect.
3. **Synthesis.** The GRU behaves like Q-learning with two changes:
   - it holds a memory of which side it has been choosing, which losses don't erase;
   - it weighs a single new outcome against the run that preceded it.

   Both are what the RL models miss in C4c/C4e.
4. **The model ladder: a choice trace is the component that matters (H1).**
   - **Setup.** Each rung is fitted per mouse to the GRU's own predictions and scored on the
     animals.
   - **R0, plain Q-learning.** Fitted to the GRU, it recovers Hattori's fit to the animals:
     0.7169 against 0.7159.
   - **R1, a one-trial choice kernel.** This reaches CTT's level (0.7212), beating CTT in 54%
     of mice.
   - **R2, a decaying choice trace (the step that matters).**
     - Fitted retention: median 0.72 per trial (IQR 0.63–0.97; started at 0.5), a
       timescale of about 3 trials for the median mouse. That is close to v's choice
       weights in r6 I7. The wide spread means some mice use a much slower trace.
     - Likelihood reaches 0.7240, beating CTT in 88% of mice.
     - The "staying through losses after a switch" signature collapses from an error of
       0.136 (R0) to 0.006, the GRU's level.
     - *(Corrected 2026-10-26: the first fit started λ_T at 0.73, the value it then
       reported. The refit starts every new component neutral and saves per-mouse
       parameters in `_cache/h1_ladder_params_C-s0.parquet`. The results are unchanged
       within 0.0002.)*
   - **R3–R5 add little together** (+0.0011, up to 97% of mice). The components are:
     - an outcome-scaled trace;
     - a partial reset on a switch (median ρ 0.96, but IQR 0.2–1.0: strong for some mice);
     - an outcome-split last-choice kernel.
     Of these, R4 helps the return-after-failed-switch curve (error 0.035 → 0.027).
   - **The ladder plateaus at 0.7251.** That closes 32% of the CTT → GRU gap, and 51% of the
     Hattori → GRU gap. The remaining 0.009 matches what I7's two-variable distillation left
     (0.723). So the rest of the GRU's advantage is not one more additive component.
   - **The next candidate** is a non-saturating update (H2, item 5), not a state-dependent
     learning rate. H3 also showed the post-loss update is poorly described by a fixed
     linear rule in Q-coordinates.
   - **Identifiability.** From R3 on, the trace parameters trade off against Q-learning's: R3
     moves the trace to a slow timescale (λ_T 0.96) with large κ_unrew. Read rungs R3–R5
     functionally (what each adds to the fit), not parameter by parameter.
5. **The GRU's learning rate is *steadier* than Q-learning's, not more adaptive (H2).**
   - **Measurement.** Flip one reward and measure the next trial's log-odds. On average this
     moves the GRU by 1.00 (it matches C1), Hattori by 0.68 and Bari by 0.75.
   - **Much less saturation in the GRU.**
     - In Hattori, the effect falls as the chosen value saturates: with run length from
       0.96 to 0.47, with confidence from 1.23 to a minimum of 0.42, and with the recent reward rate from
       1.14 to 0.36.
     - Hattori's effect is also more than twice as large after a switch as after a stay
       (1.33 vs 0.59).
     - In the GRU the run-length, confidence and stay/switch curves are flat: 0.92–1.07 over
       run length, 0.86–1.06 over confidence, 1.00 vs 1.02 for stay vs switch.
     - Over the recent reward rate the GRU's effect does fall, from 1.12 to 0.83 (−26%), but
       far less than Hattori's (1.14 → 0.36, −68%).
     - Hattori's fall is the delta rule's α(1 − Q): reward matters less once Q is near 1. The
       GRU shows at most a weak version of this bound.
     - *(Corrected 2026-10-26: the reward-rate bins first split discrete rates on bin edges,
       which made the GRU's curve look flat with a spurious dip.)*
   - **Where the GRU does modulate:**
     - A reward after a dry spell counts more: 0.85 when the previous trial was rewarded,
       rising to 1.2 after two or more unrewarded trials.
     - A reward for an unexpected choice counts more: 1.28 when P(chosen) < 0.1, against a
       minimum of 0.84 around 0.4–0.5.
     - Effects are larger in Coupled Without Baiting (1.21, against 0.95–1.07 elsewhere).
   - **Bari** rises with confidence (0.54 → 1.03), because its fixed learning rate acts on an
     unbounded logit through β.
   - **Not meta-learning of the usual kind.** A learning rate that scales up with
     uncertainty or volatility is not what we see. The GRU's rate is flat over confidence
     and run length, and it falls only moderately as the recent reward rate rises. Rewards
     do count more after a dry spell, which is the one modulation in the expected direction.
   - **Implication for the ladder.** The missing piece is more likely an update that doesn't
     saturate (additive, in log-odds space, like v's affine update with distant targets, I4)
     than a state-dependent α.

<!-- BEGIN result-8 -->
[regenerated by `analysis/report_r8.py` — do not edit by hand]

### H3 Dynamics in Q-learning's coordinates

GRU Q̂ decoded from its state (target: Hattori (L2F1) per-mouse Q), held-out R² Q_L 0.946, Q_R 0.939.

One-trial change ΔQ = a + b·Q_chosen + c·Q_unchosen, per input and model (a / b / c, R²):

| input | side | GRU | Hattori | Bari |
|---|---|---|---|---|
| R rewarded | chosen | +0.434 / -0.434 / -0.002 (0.90) | +0.413 / -0.405 / +0.075 (0.84) | +0.135 / -0.095 / +0.151 (0.24) |
| R rewarded | unchosen | -0.004 / +0.009 / -0.721 (0.93) | +0.002 / -0.001 / -0.752 (0.96) | +0.000 / +0.001 / -0.585 (0.94) |
| R unrewarded | chosen | +0.051 / -0.129 / -0.025 (0.39) | -0.007 / -0.045 / +0.018 (0.19) | +0.012 / -0.247 / -0.044 (0.79) |
| R unrewarded | unchosen | -0.018 / +0.018 / -0.610 (0.89) | +0.002 / -0.001 / -0.726 (0.95) | +0.000 / +0.001 / -0.561 (0.93) |
| L rewarded | chosen | +0.426 / -0.419 / -0.006 (0.90) | +0.408 / -0.399 / +0.080 (0.84) | +0.134 / -0.094 / +0.152 (0.24) |
| L rewarded | unchosen | -0.023 / +0.034 / -0.691 (0.93) | +0.001 / -0.001 / -0.748 (0.96) | +0.000 / +0.000 / -0.585 (0.94) |
| L unrewarded | chosen | +0.026 / -0.103 / +0.096 (0.47) | -0.008 / -0.046 / +0.018 (0.19) | +0.013 / -0.248 / -0.047 (0.79) |
| L unrewarded | unchosen | -0.019 / +0.017 / -0.596 (0.86) | +0.002 / -0.001 / -0.728 (0.95) | +0.000 / +0.001 / -0.562 (0.93) |

![H3](../fig_H3_q_phase_portraits.png)

### H4 Canonical scenarios

614 mice; fresh session (GRU h0 = 0; Q-learning Q = 0; CTT at its initial state); context: per-mouse mean of its eval-session contexts. Median P(right) across mice:

- **A  8 R+, then 8 R−** — gru: 0.50, 0.81, 0.91, 0.95, 0.96, 0.97, 0.97, 0.97, 0.97, 0.96, 0.94, 0.93, 0.91, 0.90, 0.89, 0.88, 0.87; hattori: 0.51, 0.83, 0.92, 0.94, 0.95, 0.96, 0.96, 0.96, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.89, 0.88; bari: 0.51, 0.84, 0.91, 0.94, 0.96, 0.97, 0.98, 0.98, 0.98, 0.97, 0.95, 0.93, 0.90, 0.88, 0.86, 0.84, 0.83; ctt: 0.50, 0.78, 0.90, 0.94, 0.96, 0.97, 0.98, 0.98, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.87, 0.86, 0.86
- **B  8 R+, L−, then 6 R+** — gru: 0.50, 0.81, 0.91, 0.95, 0.96, 0.97, 0.97, 0.97, 0.97, 0.73, 0.92, 0.94, 0.94, 0.94, 0.95, 0.95; hattori: 0.51, 0.83, 0.92, 0.94, 0.95, 0.96, 0.96, 0.96, 0.96, 0.66, 0.88, 0.93, 0.95, 0.95, 0.96, 0.96; bari: 0.51, 0.84, 0.91, 0.94, 0.96, 0.97, 0.98, 0.98, 0.98, 0.59, 0.93, 0.95, 0.97, 0.97, 0.98, 0.98; ctt: 0.50, 0.78, 0.90, 0.94, 0.96, 0.97, 0.98, 0.98, 0.98, 0.41, 0.78, 0.90, 0.94, 0.96, 0.97, 0.98
- **C  8 R+, then 6 L+** — gru: 0.50, 0.81, 0.91, 0.95, 0.96, 0.97, 0.97, 0.97, 0.97, 0.43, 0.14, 0.07, 0.05, 0.04, 0.03; hattori: 0.51, 0.83, 0.92, 0.94, 0.95, 0.96, 0.96, 0.96, 0.96, 0.28, 0.10, 0.06, 0.05, 0.04, 0.04; bari: 0.51, 0.84, 0.91, 0.94, 0.96, 0.97, 0.98, 0.98, 0.98, 0.40, 0.15, 0.08, 0.05, 0.03, 0.03; ctt: 0.50, 0.78, 0.90, 0.94, 0.96, 0.97, 0.98, 0.98, 0.98, 0.23, 0.11, 0.06, 0.04, 0.03, 0.02
- **D  (R+, R−) × 8** — gru: 0.50, 0.81, 0.85, 0.93, 0.93, 0.96, 0.96, 0.97, 0.97, 0.98, 0.97, 0.98, 0.97, 0.98, 0.98, 0.98, 0.98; hattori: 0.51, 0.83, 0.82, 0.91, 0.90, 0.94, 0.92, 0.94, 0.93, 0.95, 0.94, 0.95, 0.94, 0.95, 0.94, 0.95, 0.94; bari: 0.51, 0.84, 0.82, 0.89, 0.87, 0.92, 0.89, 0.93, 0.91, 0.94, 0.92, 0.95, 0.92, 0.95, 0.93, 0.95, 0.93; ctt: 0.50, 0.78, 0.79, 0.90, 0.88, 0.93, 0.91, 0.95, 0.92, 0.95, 0.93, 0.96, 0.93, 0.96, 0.94, 0.96, 0.94
- **E  8 R+, 4 R−, then 8 L+** — gru: 0.50, 0.81, 0.91, 0.95, 0.96, 0.97, 0.97, 0.97, 0.97, 0.96, 0.94, 0.93, 0.91, 0.27, 0.10, 0.05, 0.04, 0.03, 0.03, 0.03, 0.03; hattori: 0.51, 0.83, 0.92, 0.94, 0.95, 0.96, 0.96, 0.96, 0.96, 0.95, 0.94, 0.93, 0.92, 0.25, 0.09, 0.06, 0.05, 0.04, 0.04, 0.04, 0.04; bari: 0.51, 0.84, 0.91, 0.94, 0.96, 0.97, 0.98, 0.98, 0.98, 0.97, 0.95, 0.93, 0.90, 0.24, 0.11, 0.07, 0.04, 0.03, 0.03, 0.02, 0.02; ctt: 0.50, 0.78, 0.90, 0.94, 0.96, 0.97, 0.98, 0.98, 0.98, 0.96, 0.94, 0.92, 0.90, 0.23, 0.11, 0.06, 0.04, 0.03, 0.02, 0.02, 0.02

| sweep | model | values |
|---|---|---|
| F: P(return to R) after n R+ then L−, n = 1…12 | gru | 0.46, 0.56, 0.61, 0.66, 0.70, 0.72, 0.73, 0.73, 0.73, 0.73, 0.73, 0.73 |
| F: P(return to R) after n R+ then L−, n = 1…12 | hattori | 0.58, 0.62, 0.64, 0.65, 0.65, 0.66, 0.66, 0.66, 0.66, 0.66, 0.66, 0.66 |
| F: P(return to R) after n R+ then L−, n = 1…12 | bari | 0.36, 0.42, 0.47, 0.51, 0.54, 0.56, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63 |
| F: P(return to R) after n R+ then L−, n = 1…12 | ctt | 0.41, 0.41, 0.41, 0.41, 0.41, 0.41, 0.41, 0.41, 0.41, 0.41, 0.41, 0.41 |
| G: P(stay on L) after 6 R+ then k L−, k = 1…10 | gru | 0.28, 0.58, 0.70, 0.75, 0.77, 0.78, 0.78, 0.78, 0.79, 0.80 |
| G: P(stay on L) after 6 R+ then k L−, k = 1…10 | hattori | 0.34, 0.45, 0.48, 0.49, 0.49, 0.49, 0.49, 0.49, 0.49, 0.49 |
| G: P(stay on L) after 6 R+ then k L−, k = 1…10 | bari | 0.44, 0.60, 0.66, 0.68, 0.69, 0.69, 0.70, 0.70, 0.70, 0.70 |
| G: P(stay on L) after 6 R+ then k L−, k = 1…10 | ctt | 0.59, 0.66, 0.71, 0.74, 0.76, 0.78, 0.79, 0.79, 0.80, 0.80 |

Scenario D, mean |ΔP(right)| per trial after trial 6 (median across mice): gru 0.007, hattori 0.010, bari 0.024, ctt 0.023.

![H4](../fig_H4_scenarios.png)

### H1 Model ladder (each rung fitted per mouse to the GRU's P(right), scored on the animals)

Reference likelihoods: GRU 0.7339, Hattori 0.7159, Bari 0.7181, CTT 0.7210.

| rung | animal likelihood | gap closed (CTT → GRU) | mice beating CTT | cross-entropy to GRU | median parameters |
|---|---:|---:|---:|---:|---|
| R0 Q-learning | 0.7169 | -32% | 14% | 0.33414 | a_rew 0.501, a_unrew 0.0597, forget 0.799, beta 3.24 |
| R1 + one-trial choice kernel | 0.7212 | +1% | 54% | 0.32835 | w_ck 0.793 |
| R2 + choice trace | 0.7240 | +23% | 88% | 0.32426 | lam_T 0.722, w_T 1.2 |
| R3 + outcome-scaled trace | 0.7247 | +29% | 94% | 0.32334 | lam_T 0.961, w_T 0.295, kappa_unrew 5.06 |
| R4 + reset on switch | 0.7250 | +31% | 97% | 0.32297 | rho 0.966 |
| R5 + outcome-split last-choice kernel | 0.7251 | +32% | 97% | 0.32283 | w_ck 0.339, w_ck_unrew 0.455 |

C4e signature error (mean |model − animals|):

| curve | GRU | R0 | R1 | R2 | R3 | R4 | R5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| return, failed | 0.005 | 0.093 | 0.052 | 0.036 | 0.035 | 0.027 | 0.026 |
| return, rewarded | 0.002 | 0.005 | 0.019 | 0.021 | 0.019 | 0.015 | 0.016 |
| stay, after a reward on that side | 0.001 | 0.011 | 0.008 | 0.004 | 0.004 | 0.003 | 0.002 |
| stay, right after switching there | 0.004 | 0.136 | 0.040 | 0.006 | 0.009 | 0.011 | 0.012 |

![H1](../fig_H1_model_ladder.png)

### H2 Effective learning rate by state (one reward flipped, next trial, log-odds)

200,000 probes; GRU one-step replay vs cached logits median |Δ| 0.0003. Mean effect / coefficient of variation across probes: gru 1.005 / 0.50, hattori 0.679 / 0.77, bari 0.752 / 0.47.

- **confidence |logit(t)|** — gru: 0.86, 0.93, 1.04, 1.06, 1.05, 1.03, 1.01, 1.00, 1.00, 0.99, 0.97, 0.89; hattori: 1.23, 0.94, 0.92, 0.85, 0.64, 0.47, 0.42, 0.45, 0.47, 0.61; bari: 0.54, 0.60, 0.66, 0.71, 0.75, 0.78, 0.82, 0.87, 0.93, 0.99, 1.03, 0.97
- **P(chosen side) at t** — gru: 1.28, 1.12, 1.01, 0.90, 0.84, 0.86, 0.92, 1.01, 1.05, 1.00; hattori: 1.67, 1.44, 1.22, 1.13, 1.29, 1.23, 0.92, 0.83, 0.71, 0.41; bari: 0.79, 0.72, 0.63, 0.57, 0.53, 0.55, 0.59, 0.65, 0.72, 0.84
- **run length of the chosen side (stays)** — gru: 0.92, 0.92, 0.93, 0.94, 0.93, 0.94, 0.96, 0.98, 1.01, 1.01, 1.07; hattori: 0.96, 0.83, 0.72, 0.65, 0.59, 0.55, 0.53, 0.50, 0.49, 0.47, 0.47; bari: 0.69, 0.70, 0.72, 0.74, 0.75, 0.76, 0.77, 0.78, 0.78, 0.78, 0.80
- **trials since the last reward** — gru: 0.85, 1.06, 1.16, 1.19, 1.20, 1.21, 1.21, 1.22, 1.20, 1.14; hattori: 0.51, 0.66, 0.77, 0.85, 0.91, 0.97, 1.01, 1.05, 1.06, 1.09; bari: 0.78, 0.75, 0.74, 0.72, 0.73, 0.73, 0.73, 0.73, 0.73, 0.69
- **reward rate, previous 10 trials** — gru: 1.12, 1.11, 1.07, 1.04, 1.01, 0.99, 0.98, 0.96, 0.92, 0.89, 0.83; hattori: 1.14, 0.96, 0.87, 0.78, 0.68, 0.61, 0.55, 0.49, 0.43, 0.40, 0.36; bari: 0.69, 0.71, 0.71, 0.72, 0.73, 0.75, 0.78, 0.80, 0.82, 0.85, 0.87
- **position in session** — gru: 0.88, 0.98, 1.01, 1.02, 1.03, 1.03, 1.03, 1.03, 1.03, 1.02; hattori: 0.69, 0.68, 0.67, 0.67, 0.68, 0.68, 0.68, 0.68, 0.67, 0.69; bari: 0.75, 0.76, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75
- **stay vs switch** — gru: stay 1.00, switch 1.02; hattori: stay 0.59, switch 1.33; bari: stay 0.77, switch 0.66
- **task** — gru: Uncoupled Baiting 1.01, Coupled Baiting 0.95, Uncoupled Without Baiting 1.07, Coupled Without Baiting 1.21; hattori: Uncoupled Baiting 0.68, Coupled Baiting 0.62, Uncoupled Without Baiting 0.74, Coupled Without Baiting 0.95; bari: Uncoupled Baiting 0.77, Coupled Baiting 0.60, Uncoupled Without Baiting 0.94, Coupled Without Baiting 1.04

![H2](../fig_H2_learning_rate.png)
<!-- END result-8 -->

## Method notes and caveats

- **H3 reads the GRU through Hattori's coordinates.** The decoded Q̂ is a linear readout of
  the state trained on Hattori's Q. Its dynamics are the GRU's state change projected on
  those readouts, so structure outside them is invisible.
- **H3's regressions are pooled over mice.** Pooling mice with different parameters creates
  small cross-terms even for the RL models (for example Hattori's chosen ← Q_unchosen of
  +0.08). GRU cross-terms of that size are therefore not evidence of coupling.
- **H4 starts from a fresh session** (GRU h0 = 0, Q = 0) with each mouse's mean context. Long
  scripted runs, for example 10 losses on a new side, are rare in real data, so they test
  extrapolation. Medians are across 614 mice.
- **H1 is fitted on the eval sessions,** to the GRU's predictions, not the animals' choices.
  The ladder is never fitted to the animals' choices (it takes their past choices and
  rewards as inputs, like every model here), but it is not cross-validated over sessions. The
  RL references were fitted to the animals on train sessions. A train-session refit would
  make the comparison strictly out of sample.
- **One seed of one model.**

## What this suggests next

- **H1 follow-up:** refit on train sessions and cross-validate; add a rung R6 with a
  non-saturating update (a learnable reward target, so Q is not bounded by 1), as H2 suggests.
