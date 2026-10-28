# r5 — The licks model (L): what lick-count inputs change in the GRU

*Producers:
- `analysis/extract_trial_states.py`, `analysis/a2_align_with_c.py`;
- `analysis/b0_licks.py` (behaviour), `analysis/b3_state_licks.py` (representation);
- `analysis/i_licks_mechanism.py [key]` (update rule, impulse, clamps);
- `analysis/i_licks_followups.py <key>` + `analysis/fig_i_licks_followups.py` (outcome
  decomposition, push vs gain);
- `analysis/f_embedding_licks.py` (embedding).

Numbers are in the matching JSONs. Models: L-s0 and L-s1 (Study 07 `h128-dscan-licks-only`),
compared with Study 50's C-s0 on identical trials. Written 2026-10-26; revised the same day after
an independent code review (see "Corrections").*

## Setup and what is by construction

- **The inputs.** L receives, besides the previous choice and reward, the previous trial's left
  and right lick counts in [go cue, go cue + 2 s), standardized as (count − 3.139) / 4.159.
  Counts were recovered exactly from the trial table, and the network's lick inputs equal the
  table's derived inputs (max error 0).
- **Choice-relative coordinates.** Analyses re-express the counts relative to the choice on that
  trial:
  - **chosen-side licks** — on rewarded trials these include the licks that consume the
    reward;
  - **other-side licks** — licks at the port the mouse did *not* choose.
- **Typical counts.** After a loss: chosen 2.8, other 0.35 on average (median 2 / 0). After a
  win: chosen 8.8, other 0.14 (median 9 / 0). 88% of trials have no other-side licks.
- **By construction:**
  - The chosen-side count largely restates the choice, and after a reward it partly restates
    the reward.
  - Missing lick data become 0, indistinguishable from a true zero.
  - Lick data exist in almost every session (only 7 of 8,903 eval sessions have none). Unlike
    RT, the analyses use all 8,896 sessions with licks (4,336,941 trials, 614 mice).

## Summary

1. **Licks improve the fit about 10× more than RT, and the gain is real, not seed noise.**
   - Eval likelihood 0.7422 vs C's 0.7339 (+0.0084); held-out (Study 07) 0.7365–0.7367 vs
     0.7289 (+0.0077).
   - 61% of trials and 99% of mice favour L.
   - Seed noise is only ~11% of the L − C per-trial variance (sd 0.057 vs 0.172), against ~46%
     for the RT model.
2. **Licking the other port is the strongest signal.**
   - After a loss, mice switch with probability 0.13 if they did not lick the other port and
     0.56 if they licked it 6+ times. After a win: 0.08 → 0.42.
   - L reproduces this within ±0.03. C stays nearly flat (0.15 → 0.23), under-predicting
     switching by up to 0.33.
   - The ~12% of trials with any other-side licks carry most of the gain: +20 to +426 millinats
     per trial, vs +11 on average.
3. **Licking the chosen port more predicts staying.** In a joint fit at lags 1–3 (log-odds per
   sd of log(1 + count), within-session z, 100 mouse resamples):

   | | other side, 1 back (loss / win) | chosen side, 1 back (loss / win) |
   |---|---|---|
   | Mice | +0.39 / +0.37 | −0.26 / −0.22 |
   | L (both seeds) | +0.36 / +0.36–0.37 | −0.24 to −0.25 / −0.22 to −0.23 |
   | C | +0.10 / +0.16 | −0.04 / −0.04 |

   As with RT, C reproduces the lag-2/3 dependence through history (e.g. other side 2 back:
   C 0.10, mice 0.10); it misses lag 1. Unlike Study 07's logistic probe, the effect is not
   confined to losses.
4. **In the model, one trial's other-side licks move the next choice more than flipping that
   trial's reward input.**
   - +2 other-side licks: +0.87 / +0.80 logit toward switching after a win (seed 0 / 1),
     +1.06 / +1.06 after a loss.
   - Flipping only the reward input: +0.51 / +0.57 and −0.69 / −0.64.
   - The full update rule (2-D sweep at 50k real states), relative to a typical trial:
     - other-side licks push toward switching: up to +2.8 logit at 8 licks after a loss,
       +2.5 after a win (the after-win slice is drawn at 8 chosen licks, the grid value
       nearest the reference of 9, so its zero point is +0.01);
     - fewer chosen-side licks than typical push toward switching (+0.9 at 0 licks after a
       win), more push toward staying (−0.45 to −0.76 at 16 after a loss, seed 1 / seed 0).
   - Lick effects fade within ~3 trials (other +2: 0.87 → 0.30 → 0.14 → 0.08).
5. **L reads about 40% of the outcome from the licks** (FL1).
   - Flipping a trial's outcome consistently (reward input *and* the typical lick pattern)
     moves the next choice by about 1.0 logit, the same as the RT model's reward-flip effect.
   - Flipping only the reward input gives 0.50–0.70; changing only the lick pattern gives
     0.38–0.47 (seed 0; seed 1 similar).
   - So in L the reward input and the consumption-lick pattern share the outcome signal,
     roughly additively. This is consistent with L's plain reward-flip effect being smaller
     than the RT model's (+0.51 / −0.69 vs +1.00 / −1.13), though the two are different
     networks.
6. **Other-side licks mostly *erase* the current commitment, rather than add a fixed push**
   (FL3).
   - Fitting s_k = a(k) + b(k)·s₀ across states, the slope on the prior preference falls with
     k other-side licks: after a loss b = 1 → 0.74 → 0.48 → 0.16 for k = 0, 2, 4, 8.
   - The intercept rises to about +0.9. After a win: b → 0.35–0.41, a ≈ +0.25 (seed 0) to
     +0.47 (seed 1).
   - With many other-side licks, the next choice barely depends on the switch-logit the same
     state would have had with no other-side licks (s₀).
   - Caveats:
     - 97–98.5% of states favour staying, so a(k) is extrapolated.
     - The linear fit degrades where the effect is largest: after a loss R² = 0.20 at k = 6
       and 0.05 at k = 8 (after a win 0.73 and 0.50). So a(8), b(8) summarise a weak,
       saturating relationship rather than a clean line.
     - The robust statement is that the switch-logit after many other-side licks varies
       little with s₀.
7. **L also weights the outcome more when the mouse licked the other port** (FL2). A
   consistent outcome flip moves the next choice by 1.0 with no other-side licks and by ~1.5
   with 4–6.
8. **The value is in the trial-specific lick pattern, mostly at the other port** (clamps, × 10⁻³
   likelihood vs C):

   | lick inputs replaced by… | seed 0 | seed 1 |
   |---|---|---|
   | nothing (intact) | +8.35 | +8.22 |
   | their mean for that choice × outcome, other side only | +1.89 | +1.93 |
   | their mean for that choice × outcome, chosen side only | +5.71 | +5.59 |
   | their mean for that choice × outcome, both sides | −1.09 | −1.07 |
   | 0 (population mean) | −35 | −51 |
   | shuffled within session | −95 | −108 |

   - Removing the trial-specific part of the other-side licks costs about 6.5 of the 8.4. The
     chosen side costs about 2.6, and the two interact.
   - The last two rows create impossible inputs (e.g. many licks at the unchosen port on every
     trial, or licks contradicting the choice). They show L depends heavily on licks matching
     the choice, not how much information licks carry.
9. **Representation** (B3-L, ~395k trials).
   - L's state is *lower*-dimensional than C's (participation ratio 3.3 vs 4.1; PC1 = choice,
     53%). In L, chosen-side licks and reward share PC2 (r = −0.78 and −0.76).
   - Trials after other-side licking sit between the two choice lobes on PC1, i.e. less
     committed. Because the shift is toward the centre from both sides, the linear correlation
     with PC1 is ~0.
   - Decoding (unique R², beyond history and more recent licks): other-side licks 0.65, 0.35,
     0.14, 0.09, 0.06 at lags 1–5; chosen-side 0.76, 0.59, 0.21, 0.12, 0.08. C ≤ 0.04.
   - The lick memory is shorter than the RT model's RT memory (0.85, 0.81, 0.47, …).
   - Observational directions (seed 0 only) suggest that L organizes licks partly *per port*.
     The state direction for other-side licks after choosing right aligns with chosen-side
     licks after choosing left (cosine 0.76–0.80; both are licks at the left port). The
     mirror pairs for the right port are weaker (0.59–0.71).
   - Caveat: C, which never receives licks, shows similar alignments in some cells (e.g.
     +0.71 and +0.61), so these regressions partly reflect things that covary with licking.
     They cannot show where the choice-relative effect is computed.
   - The other-side directions have larger readout components (|cos| 0.26–0.31) than RT's
     (0.10–0.12).
10. **The subject embedding keeps mouse-level other-port licking** (F-L, 614 mice).
    - Decoding a mouse's mean other-side licking (CV rank R²):
      - C: 0.28 (0.23 beyond task mix);
      - L: 0.22 / 0.23 (0.17 / 0.19 beyond task mix), all p ≤ 0.005.
    - So, unlike RT, the mouse-level signal stays in L's embedding, slightly reduced.
    - Chosen-side licking after a loss is weakly decodable before the task-mix control (C
      0.038, L 0.025 / 0.028, p ≤ 0.005) but not beyond it. After a win it is not decodable
      in either. Mice with ≥ 500 trials (all 614 qualify).
    - L's embedding is less similar to C's (alignment R² 0.79–0.82) than the RT model's was
      (0.85–0.87); L's two seeds give 0.94.

## By construction vs finding

- **By construction:**
  - lag-1 lick decodability (L's input);
  - the chosen-side/reward overlap;
  - that a constant or shuffled lick input is off-distribution;
  - that a reward flip with unchanged licks is an input combination the data rarely contain.
- **Findings:**
  - the mice's switching after other-port licking and its size;
  - C's specific failure;
  - the split of the outcome signal between reward input and licks;
  - other-side licks acting mainly as a reset of commitment;
  - the retained mouse-level signal in the embedding;
  - Seeds:
    - I-L, FL, the B0L-b error figure and B0L-d were run on both seeds.
    - B0L-a, B0L-c, B0L-e, the B0L-b main figure and B3-L are seed 0 only.
    - F-L includes both seeds' embeddings.
- **Suggestive only (seed 0, observational):** a partly port-based organisation of the lick
  inputs (item 9).

## Caveats and open items

- **B0L-d units:** the lag-1 slopes are per session sd of log(1 + count), pooled over
  outcomes. Chosen-side counts are bimodal by outcome (2.8 after a loss, 8.8 after a win), so
  the per-outcome slopes are not per within-outcome sd.
- **Not run for L:** fixed points (D) and an RT-style F2 sweep.
- **The combined model (RT + licks):** does RT add anything once licks are present, and is
  "other-port licking after a loss" the same doubt signal as a slow RT?
## Corrections (2026-10-26, independent review)

- **Chosen-side decile bins (B0L-a, B0L-b).** Ranking with tied ranks piled whole blocks of
  equal counts into single bins. That caused the bump at decile 5 after losses (seen even in
  C): a bin filled with the count-2 block from high-switching mice. Ties are now broken at
  random, so the bins are true deciles, and the figures were redrawn. The earlier caveat
  ("most likely a binning artifact") is now confirmed and resolved.
- **Wording:**
  - The "per port" representation is downgraded to suggestive, with C's alignments and the
    weaker mirror pair reported.
  - "Chosen-side vigour not encoded" was partly wrong; it is "not beyond task mix".
  - FL3's fit quality is now stated.
  - "Explains" became "consistent with" for a comparison across networks.
  - The I-L1 after-win slice is drawn at 8, not 9.
  - The seed coverage of B0-L is stated.
- **Figures:** the F-L tick labels were unreadable (fixed). The FL1/FL2 panels now state that
  their bars and bands are IQRs across states.
- **Not changed:** no alignment, sign or side-assignment bug was found. The clamp arrays, the
  cell means, the permutation, the soft-label regression and the decoding sets were verified
  against the data.

## Figures

| | |
|---|---|
| B0L-a gain by previous licking | ![](../fig_B0La_gain_by_licks.png) |
| B0L-b switching by licking: mice, L, C | ![](../fig_B0Lb_switch_by_licks.png) |
| B0L-b each model's error (incl. seed 1) | ![](../fig_B0Lb_switch_residuals.png) |
| B0L-c disagreement vs seed noise | ![](../fig_B0Lc_disagreement.png) |
| B0L-d conditional lags | ![](../fig_B0Ld_conditional_lags.png) |
| B0L-e per mouse | ![](../fig_B0Le_per_mouse.png) |
| B3L-a PCA by other-side licks | ![](../fig_B3La_pca_other_licks.png) |
| B3L-b lick decoding | ![](../fig_B3Lb_lick_decoding.png) |
| B3L-c lick directions (observational) | ![](../fig_B3Lc_lick_directions.png) |
| I-L1 2-D update rule, seed 0 / seed 1 | ![](../fig_I_L1_update_rule.png) ![](../fig_I_L1_update_rule_L-s1.png) |
| C1-L impulse, seed 0 / seed 1 | ![](../fig_C1_L_impulse.png) ![](../fig_C1_L_impulse_L-s1.png) |
| I-L2 clamps, seed 0 / seed 1 | ![](../fig_I_L2_clamps.png) ![](../fig_I_L2_clamps_L-s1.png) |
| FL1–FL3 follow-ups, both seeds | ![](../fig_FL_licks_followups.png) |
| F-L embedding | ![](../fig_F_L_embedding.png) |
