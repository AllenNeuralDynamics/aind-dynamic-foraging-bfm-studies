# r4 — Outcomes after fast responses in behaviour, the subject embedding, and attractors

*Producers:
- `analysis/b0h_outcome_by_rt.py` (behaviour; `b0h_outcome_by_rt.json`);
- `analysis/f_embedding_rt.py` (embeddings; `f_embedding_rt.json`);
- `analysis/d_rt_fixed_points.py <key>` + `analysis/fig_d_rt_fixed_points.py` (attractors;
  `d_rt_fixed_points_<key>.json`, `d_rt_fixed_points_summary.json`).

Models: R-s0, R-s1 and C-s0. Written 2026-10-24.*

## Summary

1. **In the mice, an outcome counts less after their fastest responses. R reproduces this;
   C shows only half the drop** (B0h).
   - Measure: the reward effect on switching, logit P(switch | t−1 unrewarded) −
     logit P(switch | t−1 rewarded), by within-mouse decile of RT on t−1.
   - **Mice:** 0.64 in the fastest decile vs 0.79–0.85 in deciles 2–6.
   - **R:** 0.64 / 0.64 (seed 0 / seed 1) in the fastest decile.
   - **C:** 0.79 in the fastest decile, a drop of 0.09 from its decile-2 value of 0.88. The
     mice drop 0.17 (0.81 → 0.64). So part of the mice's fast-end drop comes with the state
     those trials occur in (C reproduces it without RT), and part is missing from C. R
     captures the whole drop.
   - At the slow end the mice decline to 0.61. C over-declines to 0.54 and R stays closer
     (0.67 / 0.64). R − C there (+0.13) is as large as at the fast end (−0.14).
   - Both ends are where R improves on C. The fast end is the behavioural counterpart of
     r3's F2, where the model counts outcomes after very fast responses about 45% less.
   - **Caveat:** this is marginal (different deciles hold different states). R and C are
     different networks, so R − C is a comparison, not a decomposition into "RT" and
     "state".
2. **C's embedding carries a little of each mouse's typical RT; R's carries almost none
   beyond task mix, and is otherwise close to C's embedding up to a linear map** (F-RT, 506
   mice).
   - Decoding a mouse's median log RT from its 4-d embedding (CV rank R²):
     - C: 0.085 (0.052 beyond task mix; p ≤ 0.005, the floor of a 200-shuffle null);
     - R: 0.011 / 0.012, small but above the null (p = 0.02 / 0.01), and vanishing beyond
       task mix (p ≈ 0.4).
   - The sd of log RT behaves the same way (C 0.041; R ≈ 0, raw p = 0.02 for R-s0).
   - Linear alignment (5-fold CV R²):
     - R predicts C: 0.868 / 0.853 (seed 0 / seed 1);
     - C predicts R: 0.872 / 0.867;
     - R's two seeds predict each other: 0.93–0.94.
   - Interpretation (not tested directly): the RT input takes over the small amount of
     mouse-level RT information C had packed into its embedding. Otherwise the embedding
     is largely unchanged.
3. **RT does not move the attractors in a consistent way** (D-RT).
   - The setup: each (choice, reward) input is held constant at a fixed RT input (−2.5 to
     +2.5 sd), at the mean session context, starting from 300 real states.
   - **Same as Study 50:** at every RT and on both seeds the search *found* exactly one
     stable fixed point per input. It is a deep "stay" attractor (switch-logit −2.4 to −5.2).
     - Not every start converged (as low as 69% for one condition).
     - Forward iteration runs 600 steps against relaxation times of up to about 8,000
       trials.
     - So other slow attractors cannot be excluded.
   - **What does not replicate:** how it moves with RT.
     - It is not mirror-symmetric between left and right. On seed 0, going from RT 0 to
       +1.5 sd deepens the R-rewarded attractor (−3.4 → −5.2) while the L-rewarded one
       barely changes (−3.5 → −3.7). Over the fast range (−2.5 → 0) the L-rewarded
       attractor gets shallower (−4.9 → −3.5) while the R-rewarded one barely changes.
     - On seed 1, three of four inputs become shallower with slower RT and the fourth does
       not.
   - The slowest Jacobian modes are near-marginal (|λ₁| 0.98–0.9999, τ ≈ 50–8,000 trials)
     and change irregularly with RT.
   - **Conclusion:** the asymptote of a constant input is a poor summary of what RT does.
     Its effect lives in the single-step update (r2 I-RT1, r3 F1–F2) and fades within
     about 3 trials (r2 C1-RT), long before the state approaches these attractors. This
     is a negative result, kept for completeness.

## By construction vs finding

- **By construction:**
  - C cannot use RT, so it is the control in B0h and F-RT.
  - A constant input (the same outcome and RT on every trial) is an idealisation; the
    animals never produce it.
  - Embeddings were taken from the `best_eval` checkpoint, the same one used everywhere
    else.
- **Findings:**
  - the discount of outcomes after very fast responses in the mice (half of it
    reproduced by C, all of it by R);
  - C's embedding carrying mouse-level RT and R's not;
  - the lack of a robust RT effect on attractors.
- **Interpretation:** the embedding "handing" RT over to the input.

## Corrections (2026-10-26, second review)

- **B0h:** "C: 0.79 with no drop" was false. C drops 0.09, the mice 0.17. The slow-end
  decline is not "shared by C": C over-declines, and R − C is as large there as at the fast
  end. The "attributable to RT" wording is replaced (R − C is a comparison of networks, not
  a decomposition).
- **F-RT:**
  - "R's carries none" was too strong: the raw decoding is small but above the null.
  - The alignment directions were swapped.
  - p = 0.005 is the permutation floor.
  - The hand-over reading is an interpretation.
- **D-RT:**
  - The L-rewarded example compared a different RT range than the R-rewarded one.
  - "Exactly one stable fixed point" is "one found", given incomplete convergence and
    long relaxation times.

## Figures

| | |
|---|---|
| B0h reward effect by RT decile: mice, R, R-s1, C | ![](../fig_B0h_outcome_by_rt.png) |
| F-RT embedding: mouse-level RT and alignment | ![](../fig_F_RT_embedding.png) |
| D-RT attractors vs RT, both seeds | ![](../fig_D_RT_fixed_points.png) |
