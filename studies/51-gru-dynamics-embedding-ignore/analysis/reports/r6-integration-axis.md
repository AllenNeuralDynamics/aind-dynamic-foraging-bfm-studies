---
schema_version: 1
id: r6
slug: integration-axis
status: draft
authors: [pck, claude]
wandb_groups:
  - nxd-3way@20260707-075401
  - e4-e8-d614-source@20260906-195409
inputs:
  scripts:
    - analysis/i1_cluster_axes.py
    - analysis/i2_tdr.py
    - analysis/i3_tdr_dynamics.py
    - analysis/i3c_value_causal.py
    - analysis/i4_distilled_model.py
    - analysis/report_r6.py
  data:
    - analysis/i1_cluster_axes.json
    - analysis/i2_tdr.json
    - analysis/i3_tdr_dynamics.json
    - analysis/i3c_value_causal.json
    - analysis/i4_distilled_model.json
upstream:
  - study: 50-gru-dynamics-embedding
    report: r6
reproduce: make -C studies/51-gru-dynamics-embedding-ignore r6
---

# r6 — Integration axis and a two-variable distilled model of C3

**Question.** Study 50 found that C carries one value coordinate through the jumps between
its four last-input clusters, along two mirror-tilted lobe axes, and that a distilled
one-variable update rule captures most of it. Does C3 keep that structure, where does the
fifth (post-ignore) cluster carry the side value, and how much of C3's ignore handling fits
in a simple rule?

## Summary — how C3 differs from C, and what the ignore input does

1. **The side value has C's two-lobe structure.** The side encoding axes of the L+ and L−
   clusters align (cosine 0.95), as do R+ and R− (0.96), with the left and right lobes
   nearly orthogonal (0.09–0.15) — C: 0.91 / 0.92 within, ~0.2 across (I1).
2. **After an ignore the side value is kept, between the lobes.** In the post-ignore cluster
   the side axis is a blend of both lobes (cosine 0.52–0.64 with each); the side value there
   is centred at neutral (mean +0.08) but as spread as in the engaged clusters (std 1.32 vs
   1.14–1.45). So the network holds a shrunken side belief through a bout rather than
   erasing it in one step (I1; r3 shows erasure over many ignores).
3. **Engagement rides one axis shared by all engaged clusters** (cosines 0.84–0.97), partly
   re-oriented in the post-ignore cluster (0.51–0.63) (I1).
3a. **The TDR value coordinate is carried through ignored trials, as through every engaged
   input (added 2026-10-26, I2–I3).** With "previous step ignored" and engagement added as
   regressors, C3's value axis is the same as when fitted Study 50's way on engaged steps only
   (cosine 0.997). Regressing value(t+1) on value(t): through an ignore the slope is 0.79
   (R² 0.97) from the R lobe, 0.62 (0.95) from the L lobe and 0.84 (0.99) from the
   post-ignore cluster — within or above the engaged inputs' range (slopes 0.58–0.79, R²
   0.90–0.96; C: 0.66–0.80, 0.93–0.98) and meeting Study 50's criterion for "carried". Engaged
   inputs arriving after an ignore carry it on too (slopes 0.81–0.86, R² 0.96–0.99).
3b. **What an ignore changes is the readout's offset, not the carried value.** Within every
   last-input cluster the side readout is ≈ a·v + b with a ≈ 1.0–1.2; b is the last input's push
   (R+ +1.66, R− +0.64, L+ −1.63, L− −0.57). In the post-ignore cluster b = −0.89, which puts the
   readout's neutral point (v = +0.85) at the cluster's mean value (+0.92): the last-choice push
   is gone, so the side readout is neutral on average while v itself is carried. **Correction
   of the earlier wording** ("each ignore shrinks the stored side preference by ~30% toward
   neutral", r3/r6 point 4, chat summaries): that holds for the side *readout*; on the TDR value
   coordinate the ignore carries v like any input (slope 0.62–0.84), and repeated ignores relax
   it toward the ignore attractor's value coordinate (+1.80), not toward zero. The attractor's
   readout is nonetheless side-neutral (+0.04, r3), which the within-cluster linear map does not
   reproduce (it predicts ≈ +1.0; R² 0.76), so directions other than v also set it.
3c. **Causally, the value is carried with one leak rate, whatever the input (added
   2026-10-26, I3c).** Nudging the state ±0.5 along the value axis and re-running the GRU on
   the observed inputs, 0.61–0.69 of the nudge is still on the value axis one step later in
   every input × source cell, ignore included (0.65–0.69), with SD 0.02–0.04 within a cell. I3's
   regression slopes (0.58–0.84) vary more and are higher for stays and post-ignore steps —
   the regression also picks up components that move with v. So the leak is a property of the
   value axis (~0.67 per step); inputs, including an ignore, add their push on top. Over
   steps the nudge decays fast then slowly (0.68, 0.51, 0.42, 0.36, … 0.19 after 10), slower
   than the product of I3 slopes predicts after ~4 steps (0.07 after 10): part of it settles
   into slower components. It is specific: random directions orthogonal to the value axis
   feed v little (−0.09 to +0.17 one step on) and an engage nudge not at all (|Δv| ≤ 0.03),
   while the value nudge moves the side readout (1.48 per unit, 0.78 one step on) and leaves
   engagement nearly unchanged (−0.04 to −0.06).
4. **One rule does most of the work: an ignore multiplies the side value by 0.71.** In a
   distilled model simulated from the animals' inputs alone, adding the rule v ← 0.71·v
   (+0.02) to the side variable lifts L/R likelihood after 6+ ignores from 0.426 (ignores
   skipped, as C does) to 0.505 — 83% of the C3-over-C gap there (GRUs: 0.525 vs 0.430).
   Without a gap the distilled models lose ~0.02 against their GRUs, as in Study 50 (I4).
5. **Engagement distils into a second variable with its own decay.** Per input, e(t+1) =
   A e(t) + c; after an ignore e ← 0.67·e − 0.55. Simulated from the animals' inputs it
   captures 87% of C3's engagement gain over a constant base rate (0.836 vs GRU 0.850,
   base 0.741). The two variables do not interact at ignore steps (cross terms ≤ 0.01) and
   only weakly, left/right-symmetrically, at engaged steps (|cross terms| ≤ 0.25) (I4).

## What is by construction vs what is a finding

- **By construction.** The teachers are C3's own readouts, so the distilled model can at best
  match C3; the teacher-forced fit of a linear map to them is a description, not a test.
  C's distilled model and the "ignores skipped" variant cannot use ignores by design.
- **Findings.** That a single multiplicative decay (0.71) per ignore — nearly side-symmetric
  and with ~0 offset — accounts for most of what C3 gains after ignores when *simulated*
  from inputs alone; that the post-ignore side value is held on a between-lobe axis; that
  side and engagement are nearly decoupled.

## Caveats

- The distilled models use pooled parameters (no per-mouse fit yet; Study 50's I5 did that).
- The side variable is the readout logit itself, not Study 50's TDR value axis; C's distilled
  model here is re-fitted the same way (0.7117) for a like-for-like comparison (Study 50's
  TDR version scored 0.718).
- One seed (C3-s0).

## Numbers

<!-- BEGIN result-1 -->
[regenerated by `analysis/report_r6.py` — do not edit by hand]

### I2 TDR with ignored trials

Terms (orthogonalised in this order): value, choice, reward, value×choice, choice×reward, ignored, engage; 1,000,000 steps; task + nuisance R² 0.767.

Value axis vs a refit of Study 50's design (5 terms, steps after an engaged step only): cosine 0.997. Lobe axes (β_v ± β_vc) cosine 0.28.

| raw axis | cosine with side readout | cosine with engage direction |
|---|---:|---:|
| value | +0.48 | -0.00 |
| choice | +0.00 | +0.00 |
| reward | -0.00 | -0.01 |
| value×choice | -0.00 | -0.34 |
| choice×reward | -0.00 | +0.00 |
| ignored | -0.00 | -0.05 |
| engage | -0.00 | +0.62 |

![I2](../fig_I2_tdr.png)

### I3 Value coordinate carried through each input (C3); C from Study 50

| input | source | n | slope | R² | intercept |
|---|---|---:|---:|---:|---:|
| R+ | R lobe (stay) | 160,408 | 0.751 | 0.957 | +0.53 |
| R+ | L lobe (switch) | 33,430 | 0.581 | 0.924 | +0.06 |
| R+ | post-ignore (after an ignore) | 9,545 | 0.807 | 0.984 | +0.59 |
| R− | R lobe (stay) | 227,822 | 0.785 | 0.964 | +0.38 |
| R− | L lobe (switch) | 19,464 | 0.614 | 0.932 | -0.15 |
| R− | post-ignore (after an ignore) | 9,751 | 0.842 | 0.986 | +0.41 |
| L+ | R lobe (switch) | 34,226 | 0.692 | 0.898 | -0.11 |
| L+ | L lobe (stay) | 161,455 | 0.762 | 0.950 | -0.69 |
| L+ | post-ignore (after an ignore) | 9,510 | 0.855 | 0.959 | -0.30 |
| L− | R lobe (switch) | 18,885 | 0.705 | 0.940 | +0.09 |
| L− | L lobe (stay) | 218,752 | 0.777 | 0.953 | -0.44 |
| L− | post-ignore (after an ignore) | 8,965 | 0.856 | 0.957 | -0.02 |
| ignore | R lobe (ignore) | 19,613 | 0.787 | 0.967 | +0.40 |
| ignore | L lobe (ignore) | 18,537 | 0.615 | 0.946 | -0.03 |
| ignore | post-ignore (ignore) | 49,637 | 0.840 | 0.987 | +0.30 |

C (Study 50, value axis of its own TDR): R rewarded from R lobe (stay) 0.79 / 0.97; R rewarded from L lobe (switch) 0.66 / 0.93; R unrewarded from R lobe (stay) 0.80 / 0.97; R unrewarded from L lobe (switch) 0.68 / 0.96; L rewarded from R lobe (switch) 0.73 / 0.95; L rewarded from L lobe (stay) 0.79 / 0.98; L unrewarded from R lobe (switch) 0.74 / 0.97; L unrewarded from L lobe (stay) 0.78 / 0.98.

Side readout ≈ a·v + b within each source cluster (state at t):

| cluster | a | b | v at neutral readout | mean v | R² |
|---|---:|---:|---:|---:|---:|
| R+ | 1.18 | +1.66 | -1.40 | +1.26 | 0.66 |
| R− | 1.09 | +0.64 | -0.58 | +1.34 | 0.62 |
| L+ | 1.01 | -1.62 | +1.60 | -1.55 | 0.78 |
| L− | 1.04 | -0.57 | +0.55 | -1.47 | 0.79 |
| ignore | 1.05 | -0.89 | +0.85 | +0.92 | 0.76 |

Value coordinate of each input's stable attractor (mean context): R+ +2.23; R− +2.99; L+ -3.09; L− -2.85; ignore +1.80.

![I3](../fig_I3_tdr_dynamics.png)

### I3c Causal test: nudge ±δ along the value axis, re-run on the observed inputs

20,000 starts (4,000 per input), δ = 0.5; unperturbed rollout matches the stored states to 4.2e-08. At the nudged state itself: side readout +1.48 per unit, engage +0.000.

| input | source | n | causal Δv(t+1)/δ (mean ± SD) | I3 slope |
|---|---|---:|---|---:|
| R+ | R lobe | 3,147 | 0.667 ± 0.023 | 0.751 |
| R+ | L lobe | 674 | 0.608 ± 0.034 | 0.581 |
| R− | R lobe | 3,533 | 0.693 ± 0.023 | 0.785 |
| R− | L lobe | 296 | 0.643 ± 0.032 | 0.614 |
| L+ | R lobe | 669 | 0.691 ± 0.032 | 0.692 |
| L+ | L lobe | 3,163 | 0.658 ± 0.025 | 0.762 |
| L− | R lobe | 331 | 0.685 ± 0.034 | 0.705 |
| L− | L lobe | 3,533 | 0.689 ± 0.030 | 0.777 |
| ignore | R lobe | 908 | 0.692 ± 0.020 | 0.787 |
| ignore | L lobe | 927 | 0.647 ± 0.036 | 0.615 |
| ignore | post-ignore | 2,165 | 0.692 ± 0.018 | 0.840 |

| steps after the nudge | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| value nudge → Δv/δ | 0.68 | 0.51 | 0.42 | 0.36 | 0.31 | 0.28 | 0.25 | 0.23 | 0.21 | 0.19 |
| product of I3 slopes | 0.76 | 0.59 | 0.45 | 0.35 | 0.26 | 0.20 | 0.16 | 0.12 | 0.09 | 0.07 |
| value nudge → Δside | 0.78 | 0.53 | 0.42 | 0.35 | 0.30 | 0.26 | 0.23 | 0.21 | 0.19 | 0.18 |
| value nudge → Δengage | -0.04 | -0.05 | -0.06 | -0.06 | -0.06 | -0.06 | -0.06 | -0.06 | -0.06 | -0.05 |
| engage nudge → Δv/δ | +0.00 | -0.00 | -0.02 | -0.03 | -0.03 | -0.03 | -0.03 | -0.03 | -0.03 | -0.03 |
| engage nudge → Δengage | 1.03 | 0.92 | 0.85 | 0.80 | 0.76 | 0.73 | 0.70 | 0.68 | 0.65 | 0.63 |
| random ⊥ value 1 → Δv/δ | -0.05 | -0.07 | -0.08 | -0.10 | -0.12 | -0.13 | -0.14 | -0.15 | -0.15 | -0.15 |
| random ⊥ value 2 → Δv/δ | +0.17 | +0.23 | +0.22 | +0.20 | +0.18 | +0.16 | +0.14 | +0.12 | +0.10 | +0.09 |
| random ⊥ value 3 → Δv/δ | -0.01 | -0.03 | -0.05 | -0.07 | -0.08 | -0.09 | -0.09 | -0.09 | -0.09 | -0.09 |
| random ⊥ value 4 → Δv/δ | -0.09 | -0.13 | -0.15 | -0.15 | -0.14 | -0.13 | -0.12 | -0.11 | -0.10 | -0.09 |
| random ⊥ value 5 → Δv/δ | -0.06 | -0.05 | -0.03 | -0.02 | -0.03 | -0.04 | -0.05 | -0.05 | -0.06 | -0.07 |

Correlation across starts of observed vs chain-predicted Δv/δ, per step: 0.51, 0.63, 0.70, 0.72, 0.72, 0.71, 0.69, 0.67, 0.64, 0.62.

![I3c](../fig_I3c_value_causal.png)

### I1 Encoding axes per last-input cluster (cosines; shared sessions)

C3, side readout:

| | L+ | L− | R+ | R− | ignore |
|---|---:|---:|---:|---:|---:|
| L+ | 1.00 | 0.95 | 0.15 | 0.14 | 0.52 |
| L− | 0.95 | 1.00 | 0.09 | 0.10 | 0.55 |
| R+ | 0.15 | 0.09 | 1.00 | 0.96 | 0.60 |
| R− | 0.14 | 0.10 | 0.96 | 1.00 | 0.64 |
| ignore | 0.52 | 0.55 | 0.60 | 0.64 | 1.00 |

C (Study 50), side readout:

| | L+ | L− | R+ | R− |
|---|---:|---:|---:|---:|
| L+ | 1.00 | 0.91 | 0.21 | 0.22 |
| L− | 0.91 | 1.00 | 0.18 | 0.19 |
| R+ | 0.21 | 0.18 | 1.00 | 0.92 |
| R− | 0.22 | 0.19 | 0.92 | 1.00 |

C3, engage readout:

| | L+ | L− | R+ | R− | ignore |
|---|---:|---:|---:|---:|---:|
| L+ | 1.00 | 0.96 | 0.84 | 0.85 | 0.63 |
| L− | 0.96 | 1.00 | 0.87 | 0.91 | 0.62 |
| R+ | 0.84 | 0.87 | 1.00 | 0.97 | 0.51 |
| R− | 0.85 | 0.91 | 0.97 | 1.00 | 0.58 |
| ignore | 0.63 | 0.62 | 0.51 | 0.58 | 1.00 |

| cluster | C3 side mean / std | C side mean / std | C3 engage mean / std |
|---|---|---|---|
| L+ | -3.19 / 1.45 | -3.11 / 1.44 | +5.01 / 1.70 |
| L− | -2.11 / 1.17 | -2.04 / 1.15 | +4.06 / 1.67 |
| R+ | +3.14 / 1.41 | +3.10 / 1.43 | +5.00 / 1.72 |
| R− | +2.10 / 1.14 | +2.07 / 1.15 | +4.01 / 1.69 |
| ignore | +0.08 / 1.32 | — | -0.34 / 1.25 |

![I1](../fig_I1_cluster_axes.png)

### I4 Distilled models, simulated from the animals' inputs (shared sessions)

| model | L/R likelihood | gap 0 | gap 1 | gap 2 | gap 3–5 | gap 6+ |
|---|---:|---:|---:|---:|---:|---:|
| C3 (GRU) | 0.7351 | 0.742 | 0.614 | 0.574 | 0.552 | 0.525 |
| distilled 2-var (v, e) | 0.7176 | 0.725 | 0.594 | 0.555 | 0.532 | 0.506 |
| distilled v, with ignore rule | 0.7137 | 0.721 | 0.592 | 0.552 | 0.528 | 0.505 |
| distilled v, ignores skipped | 0.7117 | 0.720 | 0.578 | 0.520 | 0.482 | 0.426 |
| C (GRU) | 0.7338 | 0.742 | 0.597 | 0.535 | 0.496 | 0.430 |
| C distilled v | 0.7117 | 0.720 | 0.578 | 0.520 | 0.483 | 0.427 |

Trials per gap bin: 0 4,135,637; 1 113,192; 2 28,830; 3–5 23,981; 6+ 12,700.

Engage-vs-ignore likelihood on every step: C3 (GRU) 0.8503; distilled 2-var (v, e) 0.8360; constant base rate 0.7411.
Simulated vs GRU: corr(v) 0.958, corr(e) 0.852.

2-var update per group, x(t+1) = A x(t) + c, x = (v, e):

| group | n | A (v←v, v←e; e←v, e←e) | c (v, e) |
|---|---:|---|---|
| L− (stay) | 1,065,671 | 0.80, +0.040; +0.095, 0.94 | -0.42, +0.26 |
| L− (switch) | 97,119 | 0.31, +0.093; -0.225, 0.95 | -0.95, +0.35 |
| L− (first) | 2,223 | 0.00, +0.000; +0.000, 0.00 | -0.86, +2.71 |
| L+ (stay) | 790,894 | 0.81, +0.068; +0.105, 0.92 | -1.46, +1.13 |
| L+ (switch) | 176,097 | 0.13, +0.151; -0.252, 0.87 | -1.98, +1.38 |
| L+ (first) | 2,311 | 0.00, +0.000; +0.000, 0.00 | -1.66, +3.41 |
| R− (stay) | 1,110,811 | 0.81, -0.046; -0.081, 0.93 | +0.46, +0.26 |
| R− (switch) | 101,790 | 0.37, -0.070; +0.226, 0.84 | +0.92, +0.48 |
| R− (first) | 2,126 | 0.00, +0.000; +0.000, 0.00 | +0.86, +1.66 |
| R+ (stay) | 786,165 | 0.83, -0.077; -0.135, 0.93 | +1.42, +1.18 |
| R+ (switch) | 171,567 | 0.18, -0.091; +0.194, 0.81 | +1.83, +1.38 |
| R+ (first) | 2,184 | 0.00, +0.000; +0.000, 0.00 | +1.80, +2.78 |
| ignore | 416,729 | 0.71, +0.008; +0.001, 0.67 | +0.02, -0.55 |

1-var ignore rule: v ← 0.710·v +0.024.

![I4](../fig_I4_distilled_model.png)
<!-- END result-1 -->
