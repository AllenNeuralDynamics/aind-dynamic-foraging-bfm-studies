---
schema_version: 1
id: r2
slug: rt-mechanism
status: draft
authors: [pck, claude]
wandb_groups:
  - h128-dscan-rt-only@20260823-093058
inputs:
  scripts:
    - analysis/i_rt_mechanism.py
  data:
    - analysis/i_rt_mechanism.json
    - analysis/i_rt_mechanism_R-s1.json
reproduce: make -C studies/52-gru-dynamics-timing-inputs r2
---
# r2 — How the RT input acts in R's dynamics (tier 3, two seeds)

*Producer: `analysis/i_rt_mechanism.py [key]` (stage A, GPU; numbers in
`i_rt_mechanism.json` for R-s0 and `i_rt_mechanism_R-s1.json`). Each seed is replayed with
the wrapper's `GruStep` on the full eval split. The replays reproduce the logged
likelihoods (diff 4e-7 and 2e-7), and the RT column of `xs` equals the A1 table's
`prev_log_rt_input` exactly. RT is in the model's input units: log RT, globally
standardized, 0 = population mean (≈0.16 s), 1 sd ≈ ×1.8 in seconds. "Toward switching"
= toward the side not chosen on the trial whose outcome is fed. Sample: 50,000 real states
from RT-present sessions. First written 2026-10-24; revised the same day after an
independent code review and the seed-1 repeat (see "Corrections"). Values are given as
"seed 0 / seed 1" where both are shown.*

## Summary

<!-- BEGIN result-2 -->
1. **The RT update rule** (I-RT1, left panel). Relative to an average RT:
   - **Slow RT** pushes toward switching: +0.54 to +0.63 logit at +2.5 sd (≈0.7 s), on
     both seeds and all four input clusters.
   - **Fast RT** pushes toward staying:
     - after losses down to −0.27 to −0.29 (seed 0) / −0.34 to −0.35 (seed 1), with the
       minimum at −1.25 to −1.75 sd;
     - after wins only down to −0.11 to −0.15, with the minimum at −0.75 to −1.0 sd.
   - **At the very fastest inputs** (below about −1.75 sd, < 0.06 s, under 2% of real
     inputs), the curve after wins turns back toward switching. This is clear on seed 0
     (+0.12 to +0.16 at −2.5 sd) but small on seed 1 (+0.04 to +0.05). Treat it as
     seed-dependent.
   - The rule is mirror-symmetric for left and right choices.
2. **The effect is state-dependent: larger after losses and when R is committed**
   (I-RT1, middle; F1 in r3).
   - The local slope at the observed RT is larger after unrewarded trials: median 0.30 /
     0.31 vs 0.21 / 0.21 per sd.
   - It rises with R's confidence, measured as |logit| at RT input 0 so that the RT push
     itself is not included:
     - after wins, 0.06 → 0.30 / 0.05 → 0.27;
     - after losses, 0.15 → 0.48 / 0.14 → 0.47, from the least to the most confident decile.
   - 94% / 95% of states have a positive slope.
   - r3's F1 shows the effect scales with how strongly R favours staying. Because almost
     no states favour switching, "shrink the preference toward neutral" and "push in
     proportion to commitment" cannot be told apart.
3. **One RT shifts the next choice by about a quarter of a reward, and fades faster** (C1-RT).

   | pulsed trial | RT +1 sd | RT −1 sd | reward flipped |
   |---|---|---|---|
   | rewarded | +0.24 / +0.24 | −0.09 / −0.11 | +1.00 / +0.99 |
   | unrewarded | +0.30 / +0.30 | −0.21 / −0.24 | −1.13 / −1.13 |

   - Values are the change in logit toward switching on the next choice.
   - The response is asymmetric: slow counts more than fast.
   - Three trials later the RT effect is about 0.03; the reward-flip effect is still
     0.16–0.23.
4. **R's gain over C depends on trial-level RT; a constant RT is worse than no RT model**
   (I-RT2).
   - Intact R beats C by +1.23 / +1.27 ×10⁻³ (RT-present sessions).
   - Replacing the real RT inputs with a constant drops R below C:

     | constant | seed 0 | seed 1 (×10⁻³) |
     |---|---|---|
     | session mean | −0.57 | −0.43 |
     | mouse mean | −0.57 | −0.44 |
     | 0 | −0.68 | −0.53 |

   - The mouse's own mean is slightly better than the population constant: the difference
     (0.11 / 0.09 ×10⁻³) is 7–9% of intact R's advantage over C, or about 5–6% of the gap
     between the constant-0 replay and intact R.
   - Shuffling RT within each session is worst: −1.94 / −1.93.
   - Why a constant falls below C is not determined. R may have shifted reliance from
     history to RT. Or a constant RT input may push R's state off its trained
     distribution along a large RT dimension (r1 item 7). These replays cannot tell the
     two apart.
<!-- END result-2 -->

## Reading this together with r1 and r3

- **Causal vs observational.** r1's B3d slope (0.34–0.43 logit/sd) is observational: it mixes
  the RT input's effect with everything that covaries with RT. C shows 0.19–0.31 without
  seeing RT, so correlates are clearly part of it. But R and C are different networks, so
  subtracting C's slope does not isolate the causal part. The causal effect is measured
  here directly (median local slope 0.21–0.31/sd), and that is the number for R's use of
  RT.
- **The RT memory outlasts its behavioural effect.** RT stays decodable 3–5 trials back
  (r1 B3c), but its effect on choice is nearly gone after about 3 trials. Behaviourally,
  the most recent RT carries most of what C misses (r1 B0g). The stored RT does not
  measurably change how the next outcome counts (r3 F3, corrected).
- **The U-shaped gain in r1 B0a** (after unrewarded trials, at both RT extremes) is *not*
  the fast-end reversal in item 1, which is after rewarded trials and affects under 2% of
  inputs. It follows from the monotone part of the rule (fast → stay) plus C
  over-predicting switching after fast losses (r1 B0c: C − mice ≈ +0.018).

## By construction vs finding

- **By construction:**
  - RT reaches the GRU only as the previous trial's value.
  - "1 sd" is a population sd of log RT (the within-mouse sd is nearly the same,
    0.59–0.62).
  - The first row of a session is −1 on every channel and is never edited. Inputs for
    trials without an RT stay 0 in every clamp.
- **Findings:**
  - the direction (slow → switch);
  - the state dependence (losses, commitment);
  - the short behavioural memory;
  - that a constant RT is worse than C;
  - all replicated on two seeds, except the fast-end reversal after wins.

## Corrections (2026-10-24)

- **Item 1 previously said** the fast-end reversal "matches B0a's U-shaped gain". It does
  not: the reversal is after *rewarded* trials, B0a's U is after *unrewarded* ones, and the
  reversal is seed-dependent and confined to < 2% of inputs. The minimum is not "around
  −1.3 sd" for both outcomes: it is −0.75 to −1.0 sd after wins and −1.25 to −1.75 sd after
  losses.
- **Item 2's confidence axis was** |logit| at the observed RT, which includes the RT push.
  It is now |logit| at RT input 0. The numbers barely changed (0.06 → 0.30 vs 0.06 → 0.28
  after wins).
- **Item 4 previously said** "the mouse's typical RT is worth nothing beyond a constant" and
  "R learned to use RT instead of some of the history information C relies on". The first
  is slightly strong (the mouse mean recovers 7–9% of the intact gain more than 0 does). The second is one of two
  explanations the replays cannot distinguish.
  - The mouse and session means are now computed over real RT inputs only; they used to
    include the zero placeholders. Seed-0 numbers were unchanged at the reported precision.
- **The link to B0d's "decline with lag"** is replaced by B0g's conditional fit (r1 item 3).
- **Second review (2026-10-26):**
  - The "Reading together" paragraph previously said B3d "mostly reflects correlates" and
    that only R − C is attributable. Subtracting a different network's slope is not a valid
    decomposition (the causal slope here, 0.21–0.31, is larger than R − C, 0.09–0.19).
    Reworded.
  - The 7–9% for the mouse mean now states its denominator.
  - The JSON `key` field was overwritten by a loop variable ("reward flipped"); fixed and
    rerun. The figures' titles now name the seed.

## Figures

| | |
|---|---|
| I-RT1, seed 0 / seed 1 | ![](../fig_I_RT1_update_rule.png) ![](../fig_I_RT1_update_rule_R-s1.png) |
| C1-RT impulse, seed 0 / seed 1 | ![](../fig_C1_RT_impulse.png) ![](../fig_C1_RT_impulse_R-s1.png) |
| I-RT2 clamps, seed 0 / seed 1 | ![](../fig_I_RT2_clamps.png) ![](../fig_I_RT2_clamps_R-s1.png) |

## Next

- D-RT: fixed points and slow modes at fixed RT inputs.
- F: does C's subject embedding encode mouse-level RT, and does R's?
- Behavioural test of r3's interpretation (outcomes after fast responses count less).
- Then the licks model (L), with the same tooling.
