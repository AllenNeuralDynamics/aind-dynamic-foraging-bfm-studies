# r3 — RT as a gain, and RT × outcome (tier-3 follow-ups, two seeds)

*Producers: `analysis/i_rt_followups.py` (stage A, run per seed) and
`analysis/fig_i_rt_followups.py` (figures; numbers in `i_rt_followups_<key>.json` and
`i_rt_followups_summary.json`). R-s0 and R-s1, 50,000 real states each from RT-present
sessions. These are the same states as r2's I-RT1, except that the previous step must
also carry a real RT. All logits are in switch coordinates: s = logit toward the side not
chosen on the trial whose outcome is fed. Written 2026-10-24.*

## Summary

1. **RT's effect scales with how strongly R favours staying** (F1).
   - For each RT input v, s_v = a(v) + b(v)·s₀ fits well across states (R² ≥ 0.94), where
     s₀ is the same state's switch-logit with RT input 0. Seed values come in pairs (s0,
     s1):

     | | slope b at +2.5 sd | intercept a at +2.5 sd | b at −2.5 sd |
     |---|---|---|---|
     | **after a loss** | 0.83, 0.82 | +0.24, +0.22 | 1.07, 1.12 |
     | **after a win** | 0.84, 0.87 | +0.03, +0.15 | 0.99, 1.03 |

   - The raw change (right panels) grows with commitment: at +2 sd, from about +0.15–0.25
     in the least committed decile to +0.8 in the most committed.
   - **Limit:** 98–99% of states favour staying (s₀ < 0; median −2.1 after losses, −3.4
     after wins). The data cannot separate "slow RT shrinks the preference toward neutral"
     from "slow RT pushes toward switching in proportion to commitment". a(v) is an
     extrapolation to s₀ = 0. The robust statement is the scaling with commitment.
2. **A trial's own RT changes how much its outcome counts** (F2).
   - The effect of flipping a trial's reward on the next choice is about 45% smaller when
     that trial's RT input is very fast (−2.5 sd, ≈0.04 s) than at +1 sd: 0.61 vs 1.12 after
     wins, −0.71 vs −1.24 after losses (seed 0). Seed 1: 0.63 / 1.11 and −0.76 / −1.23. The
     peak (1.14 after wins, seed 0) is close to the +1 sd value.
   - The effect is near its maximum from about +0.5 sd upward. So outcomes after very
     fast responses are discounted; slow ones are not boosted beyond typical.
3. **The previous trial's RT has no measurable effect on this** (F3).
   - Measured with the current trial's RT input held at 0: Spearman ρ between the previous
     RT input and the flip effect is −0.02 / −0.02 after wins and losses (seed 0), −0.02 /
     −0.04 (seed 1). The medians by decile are flat.
   - The stored RT memory (r1 item 8) does not gate learning from the next outcome.

## By construction vs finding

- **By construction:**
  - In F2, the reward and the RT of the same trial enter the GRU together in one input
    vector, so an interaction is possible by architecture. Its form is the finding.
  - RT inputs below −2 sd are rare (1.6% of RT-present inputs). The fast end of F2 rests
    on few real trials, though the model's response there is well defined.
- **Findings:**
  - scaling with commitment;
  - discounting of outcomes after very fast responses;
  - the weak carry-over;
  - all replicated on two seeds.

## Interpretation (not yet tested)

A very fast response (an anticipatory or impulsive lick) is treated by the model as a less
informative trial: its outcome counts less (F2), and it pushes toward staying (r2). A slow
response loosens the current commitment. Both fit the mice's behaviour in r1, but they are
statements about the model. Testing them in behaviour directly, e.g. the win-stay /
lose-switch rate after fast vs slow trials conditioned on the model's state, is a natural
next step.

## Corrections

- **2026-10-26 (second review): F3 was confounded.** It was previously measured at the
  current trial's *observed* RT, which correlates with the previous RT (ρ = 0.58). So the
  reported "slight rise" (|ρ| ≈ 0.12–0.15) was F2's same-trial effect leaking in. With the
  current RT held at 0 the effect is gone. The conclusion (no strong gating) stands and is
  stronger.
- F2's 45% now names its reference point (+1 sd).

## Figures

| | |
|---|---|
| F1 push vs gain, both seeds | ![](../fig_F1_gain_vs_push.png) |
| F2/F3 RT × outcome, both seeds | ![](../fig_F2F3_rt_and_outcome.png) |
