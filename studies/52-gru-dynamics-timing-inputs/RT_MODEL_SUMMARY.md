# What a reaction-time input changes in a mouse-foraging GRU

*Study 52 — GRU dynamics with previous-trial response inputs. Status 2026-10-26.
The RT model (R) versus the choice/reward model (C) on identical trials from 614 mice. These
are analysis results on trained networks from Study 07, checked twice by an independent code
review. Detailed reports: [r1](analysis/reports/r1-rt-model-descriptive.md), [r2](analysis/reports/r2-rt-mechanism.md),
[r3](analysis/reports/r3-rt-gain-and-outcomes.md), [r4](analysis/reports/r4-behaviour-embedding-attractors.md). Every number
traces to a JSON output in `analysis/`. The licks model is next.*

## High-level summary

1. **The fit gain is small but real.** Held-out likelihood rises from 0.7289 to 0.7297
   (+0.0008, ~0.1%), against a seed-to-seed spread of about 0.0001. Lick counts, by comparison,
   add about 10× as much.
2. **Mice switch more after a slow response, and C misses much of it.** After a loss, P(switch)
   rises from 0.10 to 0.23 across RT deciles. C flattens this curve (errors up to 3.5 percentage
   points); the RT model matches the mice within about 0.6 points.
3. **What C lacks is mainly the most recent RT, especially after losses.** In a joint fit, C
   reproduces the mice's dependence on RT 2–3 trials back through correlated history, but at
   lag 1 it reaches less than half the mice's slope after a loss.
4. **Mice discount outcomes after their fastest responses.** The win/loss effect on switching is
   about 24% smaller after the fastest decile of RTs. The RT model reproduces the full drop, C
   about half.
5. **Inside the model, RT is stored on its own large dimension**, nearly orthogonal to the choice
   readout, and is still decodable several trials back. It reaches the choice through a
   state-dependent nudge: slower → more switching, stronger after losses and when the model is
   committed, worth about a quarter of a reward flip, gone within ~3 trials.
6. **The model also counts an outcome less when that trial's response was very fast** (about 45%
   less at ~0.04 s). The previous trial's RT has no such effect.
7. **The value is purely trial-to-trial.** Replacing RT with any constant (including the mouse's
   own mean) makes the RT model worse than C. Mouse-level RT moves out of the subject embedding,
   which otherwise stays close to C's.

> **Bottom line:** RT adds little predictive power overall, but it captures a specific,
> interpretable piece of behaviour (switching and outcome-weighting tied to response speed) that
> a choice/reward model systematically gets wrong.

**Contents:** [Setup](#setup) · [Model fit](#does-rt-improve-the-fit) ·
[Behaviour](#what-in-the-mices-behaviour-does-rt-explain) ·
[Representation](#how-the-rt-model-represents-rt) ·
[Mechanism](#mechanism-what-one-rt-input-does) ·
[Embedding & attractors](#subject-embedding-and-attractors) ·
[Verification](#verification) · [Open questions](#open-questions)

---

## Setup

**Models.** Both are GRUs from Study 07 (W&B project `mice_rt_lick_scaling`), hidden size 128,
a 4-d subject embedding plus a scalar session context, trained on the same 614 mice and data
snapshot (20260603). The best checkpoint (step 90k) is used throughout.

- **C** (Study 50's model) receives the previous choice and reward.
- **R** also receives the previous trial's reaction time (go cue → first lick): log, clipped to
  [1 ms, 10 s], standardized with fixed population constants. 0 = 0.16 s, and 1 sd is a factor
  of about 1.8. Seeds 0 and 1 were analysed; seed 1 is the replicate.

**Data.** The eval split of the training mice: 4,339,402 trials in 8,903 sessions, identical for
both models.

- 24% of trials have no RT, as whole sessions: 1,659 sessions, and 108 mice entirely, mostly
  older Coupled Baiting data. There the RT input is 0. Unless noted, analyses use the 7,076
  sessions where every trial has an RT (3,259,324 trials, 506 mice).
- 3,045 trials have a raw RT below 1 ms (1,883 negative). The encoder clips them, so they reach
  the model at about −8.5 sd. Rank-based analyses are unaffected; regressions clip within-mouse z
  at ±4.

**By construction.** The output is still only the left/right choice, so any RT structure in the
model exists because RT helps predict choice. RT reaches the model only as the previous trial's
value; the first trial of each session gets a padding value of −1.

**Units.** Likelihood = geometric-mean probability of the chosen side. "Gain" =
log P<sub>R</sub>(chosen) − log P<sub>C</sub>(chosen) per trial, in millinats. Switch-signed
logits are positive toward the side *not* chosen on the trial being fed. Error bands are 95% CIs
from resampling mice unless stated.

---

## Does RT improve the fit?

### Overall likelihood and where the gain lives (A2)

**How.** Matched R and C trial by trial (identical trials, choices and rewards verified).
Compared pooled likelihoods overall and split by whether a session has RT. Held-out numbers are
from Study 07 (new mice, 3 seeds).

**Finding.**

| | C | R | gain |
|---|---:|---:|---:|
| Held-out mice (3 seeds) | 0.7289 | 0.7297 | +0.0008 |
| Eval split, all sessions | 0.73385 | 0.73472 | +0.00087 |
| Sessions with RT | 0.74765 | 0.74888 | +0.00123 |
| Sessions without RT | 0.69146 | 0.69131 | −0.00015 |

The whole gain sits in sessions with RT, so the model does not use "RT missing" as a session
flag. For scale: lick counts add +0.0077 held-out; growing the training set from 10 to 614 mice
added +0.0057; seed sd ≈ 0.0001.

### Trial-by-trial disagreement vs seed noise (B0b)

**How.** Distributions of per-trial Δlog P for R − C and for R seed 0 − seed 1, plus the summed
gains and losses.

**Finding.** R and C disagree a lot trial by trial (sd 0.084 nats) but nearly cancel: the net
gain (+5,358 nats) is 4.4% of the summed absolute differences, and 55% of trials favour R. Two
seeds of R disagree with sd 0.057, so ordinary training noise is about half (46%) of the R − C
variance.

![B0b](analysis/fig_B0b_disagreement.png)

---

## What in the mice's behaviour does RT explain?

### Switching after fast vs slow responses (B0c)

**How.** P(switch on trial t) by deciles of RT on t−1, pooled and ranked within each mouse,
split by whether t−1 was rewarded. Mice, R and C side by side (model P(switch) = probability of
the side not chosen on t−1), plus each model minus the mice, including R seed 1.

**Finding.**
- Mice switch much more after a slow response: after a loss 0.10 → 0.23, after a win
  0.05 → 0.14. The trend holds within each mouse.
- C flattens the curve: it over-predicts switching after fast responses and under-predicts after
  slow ones, by up to 0.035.
- Both R seeds match the mice within about 0.006 in every bin.

![B0c](analysis/fig_B0c_switch_by_prev_rt.png)
![B0c residuals](analysis/fig_B0c_switch_residuals.png)

### Does an older RT matter? Conditional fit (B0g; B0d is the marginal view)

**How.** B0d bins each lag (RT 1, 2, 3 back) on its own; that view is confounded because RT is
autocorrelated (0.32–0.39). B0g fits one logistic regression of switching on RT 1–3 back jointly
(log RT z-scored within session), rewards 1–3 back and RT × reward at lag 1. The same estimator
is applied to the mice (binary) and to each model's P(switch) (soft labels), with CIs from 100
mouse resamples.

**Finding.**

| log-odds per sd | 1 back, after loss | 1 back, after win | 2 back | 3 back |
|---|---:|---:|---:|---:|
| Mice | 0.25 (0.23–0.26) | 0.31 (0.29–0.33) | 0.09 (0.08–0.10) | 0.08 (0.07–0.09) |
| R (both seeds) | 0.25 | 0.29 | 0.10 | 0.07 |
| C | 0.11 | 0.22 | 0.09 | 0.07 |

C reproduces the lag-2/3 dependence through correlated choice/reward history, without seeing RT.
It falls short at lag 1, by more than half after losses and by about a third after wins. R's
lag-2 value sits just above the mice's interval.

![B0g](analysis/fig_B0g_conditional_lags.png)
![B0d (marginal)](analysis/fig_B0d_switch_by_lagged_rt.png)

*B0d, the marginal view, for reference. Its lag-2/3 trends partly reflect lag-1 RT.*

### Do outcomes count less after fast responses? (B0h)

**How.** For each within-mouse decile of RT on t−1: the reward effect on switching,
logit P(switch | t−1 unrewarded) − logit P(switch | t−1 rewarded), for mice, both R seeds and C.

**Finding.**
- Mice: 0.64 in the fastest decile vs 0.79–0.85 in deciles 2–6, a drop of 0.17.
- R reproduces the whole drop (0.64 on both seeds). C drops only 0.09 (0.88 → 0.79).
- At the slow end the mice decline to 0.61. C over-declines (0.54) and R stays closer
  (0.67 / 0.64).
- This is marginal (different deciles hold different states), and R − C compares two networks
  rather than splitting "RT" from "state". The model-internal test is F2 below.

![B0h](analysis/fig_B0h_outcome_by_rt.png)

### Where the RT model gains (B0a)

**How.** Mean per-trial gain by deciles of RT on t−1 (pooled and within-mouse), split by outcome
on t−1.

**Finding.** After losses the gain is U-shaped: +3.3 millinats in the fastest within-mouse
decile, +0.75 at mid range, +6.9 in the slowest. After wins it is small (0.4–1.0) except the
slowest decile (+2.2). Both ends after losses are where C's flattened switch curve is most wrong.

![B0a](analysis/fig_B0a_gain_by_prev_rt.png)

### Is RT just a proxy for C's uncertainty? (B0e)

**How.** RT on trial t (within-mouse z) against C's confidence |logit| on t, and R's gain by
previous RT split by C-confidence terciles.

**Finding.** Slow responses occur when C is less confident (Spearman ρ = −0.21), but R's point
estimate of the gain is positive at every confidence level. After the slowest RTs it is largest
at mid confidence (+7.2 millinats vs +3.4 low, +3.2 high). One bin's CI includes zero.

![B0e](analysis/fig_B0e_redundancy_with_C.png)

### Which mice gain (B0f)

**How.** Each mouse's mean gain against its median RT and RT variability (sd of log RT).

**Finding.** 85% of 506 mice gain. The gain is smaller in mice with slower (ρ = −0.20) or more
variable (ρ = −0.20) RTs.

![B0f](analysis/fig_B0f_per_mouse.png)

---

## How the RT model represents RT

*Each analysis uses a random sample of about 400k trials from sessions with RT.*

### State space (B3a, B3b)

**How.** PCA of each model's 128-unit hidden state (own fit per model), PCs 1–4, coloured by the
previous choice × reward and by the previous RT. Correlations of each PC with previous RT, reward
and choice.

**Finding.**
- The participation ratio rises from 3.9 (C) to 4.4 (R). PC1 is the previous choice in both
  (|r| ≈ 0.9).
- R's PC3 (11%) correlates 0.74 with RT and 0.33 with reward; PC2 correlates 0.73 with reward and
  −0.41 with RT. RT and reward share PCs 2–3 as a rotated mixture.
- In C, PC2 is the reward axis (r = 0.89) and no PC tracks RT.

![B3a](analysis/fig_B3a_pca_inputs.png)
![B3b](analysis/fig_B3b_pca_prev_rt.png)

### How long RT stays in the state (B3c)

**How.** Ridge decoders from all 128 units to RT 1–5 trials back, 5-fold cross-validated with
folds grouped by session. Three targets: raw; beyond choice/reward history (lags 1–6 regressed
out); and "unique" (also beyond more recent RTs).

**Finding.** R's unique R² is 0.85, 0.81, 0.47, 0.19, 0.12 at lags 1–5; C reaches at most 0.03.
RT's own autocorrelation is only 0.32–0.39, so R keeps a genuine multi-trial RT memory. Lag 1 is
guaranteed (it is R's input). Behaviourally, R mostly uses the latest RT (B0g, C1-RT), so most of
this memory is not expressed in choice.

![B3c](analysis/fig_B3c_rt_decoding.png)

### The RT direction in state space (B3d, observational)

**How.** Within each previous choice × reward cluster, the regression slope of the state on
previous RT; cosines between clusters' directions, the readout and a reward axis.

**Finding.**
- R's RT direction is shared across clusters (cosine 0.71–0.97), nearly orthogonal to the
  readout (0.10–0.12) and the reward axis (≤ 0.09), and large (≈1.2 state units per sd).
- This is correlational. C, which never sees RT, also covaries with it (0.19–0.31 logit per sd,
  vs R's 0.34–0.43). The causal effect of the RT input is measured directly in I-RT1.

![B3d](analysis/fig_B3d_rt_direction.png)

---

## Mechanism: what one RT input does

*The model is replayed with the wrapper's GRU step on the full eval split; the replays reproduce
the logged likelihoods (diff 4e-7 and 2e-7). Sweeps use 50,000 real states from sessions with
RT. Every result here was repeated on seed 1.*

### The update rule (I-RT1)

**How.** At each real state, one trial's RT input is swept from −2.5 to +2.5 sd with choice and
reward unchanged. Measured: the change in the switch-signed logit versus RT = 0, and the local
slope at the observed RT against the model's confidence (|logit| at RT = 0) and against the RT
itself.

**Finding.**
- A slow RT pushes toward switching: +0.54 to +0.63 logit at +2.5 sd (≈0.7 s), in all four input
  clusters and both seeds.
- A fast RT pushes toward staying, more after losses (down to −0.27 to −0.35) than after wins
  (−0.11 to −0.15).
- The local slope is larger after losses (median 0.30–0.31 vs 0.21 per sd) and rises with
  confidence: after wins 0.06 → 0.30, after losses 0.15 → 0.48, from least to most confident
  decile. 94–95% of states have a positive slope.
- At the very fastest inputs (< 0.06 s, under 2% of inputs) the curve after wins turns back
  toward switching on seed 0 (+0.12 to +0.16) but barely on seed 1 (+0.04). Seed-dependent.

| Seed 0 | Seed 1 |
|---|---|
| ![I-RT1 seed 0](analysis/fig_I_RT1_update_rule.png) | ![I-RT1 seed 1](analysis/fig_I_RT1_update_rule_R-s1.png) |

### Push or gain? (F1)

**How.** For each swept RT value v, regress each state's switch-logit at v on its switch-logit at
RT = 0: s<sub>v</sub> = a(v) + b(v)·s<sub>0</sub>, by outcome. A pure push changes a; a pure gain
changes b.

**Finding.**
- The fit is good (R² ≥ 0.94). At +2.5 sd: b = 0.83 / 0.82 after losses and 0.84 / 0.87 after
  wins (seed 0 / 1), with an intercept of +0.24 / +0.22 after losses.
- The effect scales with how strongly the model favours staying.
- 98–99% of states favour staying, so "shrink the preference toward neutral" and "push in
  proportion to commitment" cannot be told apart.

![F1](analysis/fig_F1_gain_vs_push.png)

### Impulse response (C1-RT)

**How.** Shift one trial's RT by ±1 sd, or flip its reward, and propagate 10 trials with all other
inputs as observed. The figure is the change in the switch-signed logit.

**Finding.** Effect on the next choice (seed 0 / seed 1):

| pulsed trial | RT +1 sd | RT −1 sd | reward flipped |
|---|---:|---:|---:|
| rewarded | +0.24 / +0.24 | −0.09 / −0.11 | +1.00 / +0.99 |
| unrewarded | +0.30 / +0.30 | −0.21 / −0.24 | −1.13 / −1.13 |

One sd of RT is worth about a quarter of a reward flip, and slow counts more than fast. Three
trials later the RT effect is ≈0.03; the reward effect is still 0.16–0.23.

![C1-RT](analysis/fig_C1_RT_impulse.png)

### Does RT change how much an outcome counts? (F2, F3)

**How.** The reward-flip effect on the next choice. F2: as a function of the same trial's RT
input. F3: as a function of the previous trial's RT input, with the current RT held at 0 (the two
correlate, ρ = 0.58).

**Finding.**
- F2: after a very fast response (−2.5 sd, ≈0.04 s) the outcome counts about 45% less than at
  +1 sd: 0.61 vs 1.12 after wins, −0.71 vs −1.24 after losses; seed 1 gives 0.63/1.11 and
  −0.76/−1.23.
- F3: no measurable effect (ρ −0.02 to −0.04, flat curves). The stored RT does not gate learning
  from the next outcome.

![F2/F3](analysis/fig_F2F3_rt_and_outcome.png)

### Clamping the RT input (I-RT2)

**How.** Full replays with every real RT input replaced: shuffled within session, set to the
session mean, the mouse mean, or 0 (population mean). Scored on sessions with RT, relative to C.

**Finding.**

| likelihood − C (×10⁻³) | seed 0 | seed 1 |
|---|---:|---:|
| intact RT | +1.23 | +1.27 |
| session mean | −0.57 | −0.43 |
| mouse mean | −0.57 | −0.44 |
| 0 (population mean) | −0.68 | −0.53 |
| shuffled within session | −1.94 | −1.93 |

The value is entirely trial-to-trial: with any constant, the RT model is worse than C. Why it
falls below C is open. The model may have shifted reliance from history to RT, or a constant input
may push its state off its trained range along the large RT dimension.

| Seed 0 | Seed 1 |
|---|---|
| ![I-RT2 seed 0](analysis/fig_I_RT2_clamps.png) | ![I-RT2 seed 1](analysis/fig_I_RT2_clamps_R-s1.png) |

---

## Subject embedding and attractors

### Mouse-level RT in the subject embedding (F-RT)

**How.** For 506 mice (≥ 500 RT trials), decode the mouse's median and sd of log RT from its 4-d
embedding: 5-fold CV rank R² against a 200-shuffle null, raw and beyond task mix. Then fit
cross-validated linear maps between the embeddings of C, R seed 0 and R seed 1.

**Finding.**
- C's embedding carries a little mouse-level RT: rank R² 0.085 (0.052 beyond task mix;
  p ≤ 0.005, the floor of the null).
- R's carries almost none: 0.011 / 0.012, above the null raw (p = 0.02 / 0.01) but gone beyond
  task mix.
- Otherwise R's embedding is close to C's up to a linear map: R predicts C with R² 0.87 / 0.85,
  C predicts R with 0.87; R's two seeds predict each other at 0.93–0.94.

![F-RT](analysis/fig_F_RT_embedding.png)

### Attractors at fixed RT (D-RT, negative result)

**How.** For each choice × reward input held constant, with RT fixed at −2.5 … +2.5 sd and the
mean session context, find stable fixed points from 300 real starting states (forward iteration
plus Newton, float64). Record each one's switch-logit and leading Jacobian eigenvalues.

**Finding.**
- Every case has one stable "stay" attractor found (switch-logit −2.4 to −5.2), as in Study 50.
- How RT moves it is inconsistent between left and right inputs and between seeds. The slowest
  modes are near-marginal (|λ₁| 0.98–0.9999) and vary irregularly.
- RT acts through the single-trial update and fades in about 3 trials, long before the state nears
  these endpoints. Convergence was incomplete in one condition (69% of starts), so other slow
  attractors cannot be excluded.

![D-RT](analysis/fig_D_RT_fixed_points.png)

---

## Verification

- **Pipelines.** Each trial table reproduces its model's logged likelihood (diff 4e-9 for seed 0,
  2e-8 for seed 1). The replays reproduce it as well. The RT input recorded per trial equals what
  the network received, exactly.
- **Seeds.** Headline results were repeated on R seed 1: B0c, B0g, B0h, I-RT1, C1-RT, I-RT2,
  F1–F3, F-RT, D-RT. All replicate except the fast-end reversal after wins.
- **Two independent code reviews.** These were given the code and specs, not the conclusions.
  Neither found a bug that changed a figure's data, and one metadata bug was fixed. They did catch
  over-stated or wrong claims, now corrected:
  - "C misses older RTs": that came from the marginal view; conditionally it does not (B0g).
  - F3's apparent effect was the current trial's RT leaking in; held fixed, it vanishes.
  - The B0g lag-1 coefficient was the after-loss slope only; both are now shown.
  - R − C is a comparison of two networks, not a decomposition into "RT" and "correlates".
  - PC3 is RT mixed with reward, not pure RT. Seed noise is about half, not most, of the R − C
    disagreement. C does drop at the fast end in B0h, by half as much as the mice.

> **Not yet checked independently:** the corrected report text and redrawn figures after the
> second review (checked against the JSON outputs).

---

## Open questions

- Why does a constant RT make the model worse than C? Is it reliance shifted from history, or an
  off-distribution state?
- What is the long RT memory (decodable 3–5 trials back) for, given it barely reaches the choice?
- Do mice discount outcomes after very fast responses because those responses are impulsive or
  anticipatory, and does that show up in task variables (e.g. early licks)?
- The licks model carries about 10× more predictive information. Does it reuse the same mechanism
  (other-port licking after losses as a "doubt" signal), and does RT add anything once licks are
  present (combined model)?

---

*Models: R = `h128-dscan-rt-only@20260823-093058`, C = Study 50's
`e4-e8-d614-source@20260906-195409` (W&B entity AIND-disRNN). An HTML version with the same
content is in `analysis/_cache/report/rt_model_report.html` (not tracked).*
