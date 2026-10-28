# r1 — RT model (R-s0) vs choice/reward model (C-s0): where RT helps and how it is stored

*Producers: `analysis/a2_align_with_c.py`, `analysis/b0_where_rt_helps.py`,
`analysis/b3_state_rt.py` (numbers in the matching JSONs). Seed 0 of each model unless
noted; R-s1 is the seed-noise baseline in B0b and a replicate in B0c/B0g. Same 4.34M eval
trials for both models. Unless noted, only the 7,076 sessions where every trial has an RT
(3.26M trials, 506 mice). First written 2026-10-24; revised the same day after an
independent code review (see "Corrections").*

## Summary

1. **Mice switch more after a slow response, and C misses much of it.**
   - After an unrewarded trial, P(switch) rises from 0.10 to 0.23 across the previous
     trial's RT deciles; after a rewarded trial, from 0.05 to 0.14 (B0c).
   - The trend holds within each mouse.
   - C captures part of the trend through correlates but flattens it: it over-predicts
     switching after fast responses and under-predicts after slow ones, by up to 0.035.
   - Both seeds of R match the mice within about 0.006 in every bin.
2. **The gain is largest after unrewarded trials, at both RT extremes** (B0a).
   - Within-mouse deciles, after unrewarded trials: +3.3 millinats in the fastest decile,
     +0.75 mid-range, +6.9 in the slowest.
   - After rewarded trials the gain is small and flat (0.4–1.0) except in the slowest
     decile (+2.2).
   - Both ends of the unrewarded curve are where C's flattened switch curve is most wrong.
3. **The most recent RT carries most of what C lacks, especially after losses** (B0g; B0d
   is the marginal view).
   - The fit: a joint logistic regression of switching on RT 1, 2 and 3 back (within-session
     z), rewards 1–3 back and RT × reward 1 back, applied identically to the mice and the
     models.
   - Results, in log-odds per sd, with 95% CIs from 100 mouse resamples:

     | | RT 1 back, after a loss | RT 1 back, after a win | RT 2 back | RT 3 back |
     |---|---|---|---|---|
     | Mice | 0.25 (0.23–0.26) | 0.31 (0.29–0.33) | 0.09 (0.08–0.10) | 0.08 (0.07–0.09) |
     | R | 0.25 | 0.29 | 0.10 | 0.07 |
     | R seed 1 | 0.25 | 0.29 | 0.10 | 0.07 |
     | C | 0.11 | 0.22 | 0.09 | 0.07 |

   - C reproduces the lag-2/3 dependence through correlated choice/reward history without
     seeing RT.
   - C falls short at lag 1: by more than half after losses, by about a third after wins.
   - R matches the mice at lag 1. At lag 2 R is slightly above the mice's interval (0.103 vs
     0.081–0.101), while C is inside it.
4. **RT is related to C's uncertainty but not redundant with it** (B0e).
   - Slow responses come on trials where C is less confident (Spearman ρ = −0.21).
   - R's point estimate of the gain is positive in every confidence tercile and RT decile.
     One bin's CI includes 0 (decile 5, high confidence).
   - After the slowest RTs the gain is largest at *mid* confidence (+7.2 millinats, vs
     +3.4 low and +3.2 high).
5. **R and C disagree trial by trial much more than their net difference, and about half
   of that is seed noise** (B0b).
   - Per-trial Δlog P has sd 0.084 for R − C against 0.057 for two seeds of R. Seed noise
     is about 46% of the R − C variance (68% of the mean |Δ|).
   - The net gain (+5,358 nats) is 4.4% of the summed absolute differences.
6. **Per mouse** (B0f): 85% of mice gain. The gain is smaller in mice with slower or more
   variable RTs (ρ = −0.20 for each).
7. **R holds RT on a large dimension of its state, nearly orthogonal to the readout.**
   - PCA (B3a/b): the participation ratio rises from 3.9 (C) to 4.4 (R).
     - R's PC3 (11%) correlates 0.74 with the previous RT and 0.33 with the previous
       reward.
     - R's PC2 (12%) correlates 0.73 with reward and −0.41 with RT. RT and reward share
       PCs 2–3 as a rotated mixture.
     - In C, PC2 is the reward axis (r = 0.89) and no PC tracks RT.
     - PC1 is the previous choice in both models (|r| ≈ 0.9).
   - Within each input cluster (B3d, observational), R's state covaries with the previous
     RT along a direction:
     - shared across clusters (cosine 0.71–0.97);
     - nearly orthogonal to the readout (|cos| 0.10–0.12) and the reward axis (≤ 0.09);
     - of size about 1.2 state units per sd.
   - This is observational: it mixes the RT input's effect with everything that covaries
     with RT. C, which never sees RT, also covaries with it (0.19–0.31 logit per sd along
     its readout, vs R's 0.34–0.43), so correlates are clearly involved.
   - Because R and C are different networks, subtracting C's slope does not isolate the
     input's part. The causal effect is measured directly in r2 (I-RT1, median 0.21–0.31
     per sd).
8. **R keeps a multi-trial RT memory** (B3c).
   - Unique decodable variance, beyond choice/reward history 1–6 back and more recent
     RTs: R² 0.85, 0.81, 0.47, 0.19, 0.12 at lags 1–5.
   - C carries ≤ 0.03. RT's own autocorrelation (lag 1 vs lags 2–5) is 0.39, 0.35, 0.33,
     0.32.
   - Caveat: "unique" can still contain the part of RT k back predicted by older RTs (a
     running average); the reviewer estimated ~5% of the lag-5 residual.
   - Behaviourally, R uses mainly the most recent RT (item 3). Most of this memory is not
     expressed in choice.

## By construction vs finding

- **By construction:**
  - RT 1 back is R's input, so its lag-1 decodability is guaranteed.
  - Within-mouse and within-session z-scoring remove mouse and session averages from the
    *analysis*. The model itself sees globally standardized RT.
  - 24% of trials (whole sessions) have no RT and are excluded here; on them R gains
    nothing (A2).
  - 3,045 eval trials sit at the encoder's 1 ms floor (1,883 negative, 26 zero, 1,136 between
    0 and 1 ms; 2,921 of them in RT-present sessions) and enter the model at about −8.5
    population sd. Deciles are unaffected; B3's regressions winsorize RT z at ±4 (0.54% of
    trials).
- **Findings:**
  - the switch–RT relationship in the mice;
  - C's flattened curve, and that C nonetheless reproduces the lag-2/3 dependence;
  - R's multi-trial RT memory;
  - R's shared, readout-orthogonal RT dimension.

## Corrections (2026-10-24, after an independent code review)

The review found no bug that changed a figure's data. It did find these claims wrong or
overstated:

- **Item 3 previously said** "older RTs matter too… C's errors persist at those lags; R's
  stay near zero", from the marginal B0d.
  - Each lag there is binned alone, and RT is autocorrelated, so lag-2/3 trends partly
    reflect lag-1 RT and session drift.
  - The new conditional fit (B0g) shows C already matches the mice at lags 2–3; R's
    advantage is at lag 1.
- **Item 5 previously said** "most trial-by-trial disagreement is seed-level noise". Seed
  noise is about 46% of the variance: about half, not most.
- **Item 7 previously said:**
  - "PC3 is almost entirely the previous RT" — it is a mixture with reward (r = 0.74 RT,
    0.33 reward).
  - "the four input clusters keep their layout on PCs 1–2" — R's PC2 mixes reward and RT.
  - that the 0.29–0.39 logit-per-sd effect was R's RT effect.
    - That number is an observational regression slope, which C reproduces at 0.19–0.31
      without seeing RT.
    - It was also pulled down by RT ≤ 0 trials at about −15 within-mouse sd. With
      winsorizing it is 0.34–0.43, and only the R − C part (≈0.1–0.2) is attributable to
      the input.
  - The earlier note that this slope "averaged over the state dependence" of I-RT1 was
    also wrong: the two are different quantities.
- **Item 8:** B3c's nuisance regressors previously stopped at lag 5, which omits the
  outcome that drives RT 5 back. They now include lag 6. The lag-5 numbers changed
  slightly (C's apparent rise at lag 5 was this artifact).
- **Item 4 previously said** "largest at low/mid confidence". It is mid only.

## Corrections (2026-10-26, second review)

- **B0g:** with reward coded 0/1, the plotted "RT 1 back" coefficient was the slope after
  *unrewarded* trials only. The figure now shows the lag-1 slope after losses and after
  wins separately, and item 3 gives both. The CIs now come from 100 mouse resamples (was 20,
  too few for 95% intervals). R's lag-2 value lies just outside the mice's interval; this
  was not mentioned before.
- **Item 7:** "only the R − C difference can be credited to the input" was not a valid
  decomposition (different networks; the causal slope 0.21–0.31 exceeds R − C 0.09–0.19).
  Reworded.
- **Smaller fixes:**
  - RT autocorrelation values;
  - the count of trials at the 1 ms floor (3,045, not 1,909);
  - item 4's "in every tercile" qualified by its CIs.

## Figures

| | |
|---|---|
| B0a gain by previous RT | ![](../fig_B0a_gain_by_prev_rt.png) |
| B0b trial-by-trial disagreement vs seed noise | ![](../fig_B0b_disagreement.png) |
| B0c switching vs previous RT: mice, R, C | ![](../fig_B0c_switch_by_prev_rt.png) |
| B0c each model's error (incl. R seed 1) | ![](../fig_B0c_switch_residuals.png) |
| B0d RT at lags 1–3, marginal | ![](../fig_B0d_switch_by_lagged_rt.png) |
| B0g RT at lags 1–3, conditional (joint fit) | ![](../fig_B0g_conditional_lags.png) |
| B0e redundancy with C's confidence | ![](../fig_B0e_redundancy_with_C.png) |
| B0f per mouse | ![](../fig_B0f_per_mouse.png) |
| B3a PCA, input clusters | ![](../fig_B3a_pca_inputs.png) |
| B3b PCA, previous RT | ![](../fig_B3b_pca_prev_rt.png) |
| B3c RT decoding by lag | ![](../fig_B3c_rt_decoding.png) |
| B3d state covariation with RT (observational) | ![](../fig_B3d_rt_direction.png) |
