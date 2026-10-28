# Study 50 — GRU dynamics and subject embedding: summary of all analyses

What a GRU trained on 614 mice learned about dynamic foraging, how its strategy differs from
Q-learning, and what its per-mouse embedding encodes. Every analysis so far, with methods and
statistics. Status: draft, 2026-10-26. Details and full tables are in reports r1–r6 and r8
(this folder); every figure is described in [`../FIGURES.md`](analysis/FIGURES.md). Figures are shown
inline from the committed PNGs in `analysis/`.

**Scope:** model C, seed 0 (GRU, 128 units, 4-d subject embedding); 614 mice, 4.34M eval
trials, 8,903 sessions.

---

## High-level summary

1. **The GRU fits every mouse better than every RL model.**
   - Eval likelihood: GRU 0.734, CTT 0.721, Bari 0.718, Hattori 0.716.
   - Better in 100% of mice, with a median gain of +0.012 over CTT.
   - Its choice-history kernels match the animals' (largest gap 0.009 log-odds); the RL
     models miss by 0.12–0.23.
2. **Its state is low-dimensional and organised by the last trial.**
   - PC1 holds 43% of variance (participation ratio 4.3).
   - States fall into four clusters by the previous choice × outcome, grouped into two
     lobes by the previous choice.
   - There is one stable attractor per constant input, on the chosen side.
3. **One coordinate, v, carries the memory.**
   - A value-like coordinate is carried smoothly through every jump between clusters (slope
     0.66–0.80, R² 0.93–0.98).
   - It is a leaky trace over past choices: every choice enters and fades at about 0.65–0.72 per
     trial, rewarded choices with about 1.6–1.8× the weight. Unrewarded choices also leave a
     small push towards their side that hardly fades over 20 trials (I9, I9c, I9d).
   - A one-variable model of v tracks the GRU at correlation 0.985.
4. **Against the animals, Q-learning fails around losses and switches.**
   - Hattori under-persists after losses.
   - Bari and CTT under-return after a failed switch, and CTT ignores how long the previous
     run was.
   - The GRU is calibrated within 0.18 percentage points over all 64 three-trial histories.
5. **The GRU computes something different from Q-learning.**
   - In Q-learning's coordinates it behaves like a delta rule after rewards, but keeps the
     chosen side's value after a loss.
   - Its learning rate barely saturates.
   - Adding a decaying choice trace to Q-learning is the one component that matters
     (0.7169 → 0.7240; beats CTT in 88% of mice). The model ladder plateaus at 0.7251, a
     third of the way to the GRU.
6. **The embedding is a smooth map of strategy.**
   - It has three named, causally verified axes (determinism / stickiness, side bias, slow
     learning / long choice memory) plus one dynamics-only axis.
   - It is a continuum, not discrete types, and carries no setup or cohort signature beyond
     what task exposure and behaviour explain.
   - Each mouse's context moves along a shared, front-loaded path during training.
7. **Seeds 1 and 2 replicate it (r7, added 2026-10-27).**
   - The three seeds are the same model up to a change of basis: per-trial logit r 0.998,
     v r 0.99, hidden-state CCA ≥ 0.88 over 9 components, embedding CCA 0.84–0.985.
   - 14 of 18 headline claims pass a fixed test in all three seeds.
   - Seed 1 carries v less cleanly through lobe switches, which fails four claims there.
   - Seed-specific, and corrected here: the numbering of the embedding PCs (the directions
     replicate), seed 0's extra L-rewarded fixed points and bistable mice, and the exact
     slow-mode timescales.

---

## 0. Model, data and conventions

- **Model C.**
  - GRU with 128 units and a linear readout: logit(right) = r·h + const, with
    r = w_R − w_L.
  - Input each trial: the previous choice and reward, plus a 4-d **context**. The context is
    the mouse's learned embedding plus a small session-dependent shift computed from the
    embedding and the training phase.
  - The weights are the same for all mice. Best checkpoint: step 90,000.
- **Data.** The eval sessions (every other session) of the 614 training mice: 4.34M trials,
  8,903 sessions, four task types.
- **RL baselines.** Per-mouse fits from Study 01, all replayed on the same trials:
  - Hattori: Q-learning with separate rewarded / unrewarded learning rates and forgetting;
  - Bari: Q-learning plus a one-trial choice kernel;
  - CTT: compare-to-threshold.
- **Timing.** The state on trial t has seen trial t−1 and predicts trial t.
- **Notation.** Histories are written most recent first (t−1, t−2, t−3); + is rewarded,
  − unrewarded; R is right, L left.

---

## 1. What the hidden state looks like (r1)

### B3a · B2 · B3b — low-dimensional; PC1 is choice preference
- **How.**
  - PCA of the state over all eval trials.
  - Unit heatmaps.
  - PCs coloured by the GRU's P(right).
- **Result.**
  - **Variance by PC:** PC1 42.7%, PC2 18.0%, PC3 7.6%, PC4 6.8%, PC5 5.6%; participation
    ratio 4.3; 9 PCs reach 90%.
  - **Activity is dense:** 64 of 128 units correlate |r| > 0.5 with the logit.
  - **PC1 runs with P(right).**
  - **PC5 is a slow within-session drift** (r = 0.51 with trial position, 0.53 with the
    cumulative reward count). It also carries the previous outcome (r = 0.33).

![fig_B3a_pca_variance.png](analysis/fig_B3a_pca_variance.png)
*`fig_B3a_pca_variance.png`*

![fig_B3b_state_space_p_right.png](analysis/fig_B3b_state_space_p_right.png)
*`fig_B3b_state_space_p_right.png`*

![fig_B3_pc56_trial_position.png](analysis/fig_B3_pc56_trial_position.png)
*`fig_B3_pc56_trial_position.png`*

### B3c — four clusters set by the last trial; earlier trials set the position inside
- **How.** States in PC1–PC2 (and PC1–PC4), split by the previous choice × outcome, then by
  the last two inputs (16 panels) and the last three (64 panels).
- **Result.**
  - There is one cluster per last input; PC2 separates rewarded from unrewarded.
  - Within a cluster, a same-side t−2 input pushes the state to the committed end, and an
    opposite-side one to the inner end near P(right) 0.5.
  - t−3 shifts it again, by less: leaky integration visible in the raw state.

![fig_B3c_prev_outcome.png](analysis/fig_B3c_prev_outcome.png)
*`fig_B3c_prev_outcome.png`*

![fig_B3c_prev_outcome_t2.png](analysis/fig_B3c_prev_outcome_t2.png)
*`fig_B3c_prev_outcome_t2.png`*

![fig_B3c_prev_outcome_t3.png](analysis/fig_B3c_prev_outcome_t3.png)
*`fig_B3c_prev_outcome_t3.png`*

### B4 · B5 · B6 · B7 — other descriptive results
- **B4, update maps:** strongly nonlinear logit updates.
  - A reward pulls the logit toward the chosen side, most when the model leaned away.
  - No reward pulls it back toward 0.
- **B5, outcome-triggered:** after a reward the log-odds sit about 0.85 higher toward the
  chosen side, falling to about 0.7 at 10 trials. This is confounded by the environment,
  since rewarded trials come from richer blocks. The causal one-reward effect is C1's
  (1.0 → 0.02).
- **B6, tracking:** the GRU and the animals agree against the true reward probabilities
  (gap ≤ 0.008), but this is a weak test.
- **B7, decoders:**
  - The previous choice and reward decode at R² ≥ 0.998.
  - Rewards 2–5 back are still present (unique R² 0.98 → 0.30).
  - The true reward-probability difference decodes only moderately (R² 0.43).

![fig_B4_update_maps.png](analysis/fig_B4_update_maps.png)
*`fig_B4_update_maps.png`*

![fig_B5_outcome_triggered.png](analysis/fig_B5_outcome_triggered.png)
*`fig_B5_outcome_triggered.png`*

![fig_B6_tracking.png](analysis/fig_B6_tracking.png)
*`fig_B6_tracking.png`*

![fig_B7_decoders.png](analysis/fig_B7_decoders.png)
*`fig_B7_decoders.png`*

---

## 2. The GRU against the animals and the RL models (r2)

All comparisons use the same 4.34M trials; the animals' choices are the reference.

### C2a · C2b — the GRU is better for every mouse
- **Pooled likelihood:** GRU 0.7339, CTT 0.7210, Bari 0.7181, Hattori 0.7159.
- **Coverage:** better than all three RL fits in 100% of mice and in every task.
- **Agreement:** trial by trial the models agree closely (r = 0.97–0.98 on P(right)), so
  the advantage is concentrated in specific situations.

![fig_C2a_likelihood.png](analysis/fig_C2a_likelihood.png)
*`fig_C2a_likelihood.png`*

![fig_C2b_agreement.png](analysis/fig_C2b_agreement.png)
*`fig_C2b_agreement.png`*

### C1 — the GRU's history kernels match the animals'
- **How.**
  - The same logistic regression on rewarded and unrewarded choices 10 trials back (Su2022
    design), fitted to the animals and to each model's predictions.
  - A counterfactual: flip one past reward in the GRU's input.
- **Result.**
  - **Animal vs GRU:** rewarded choice 1 back 1.807 vs 1.796, unrewarded 0.876 vs 0.871;
    largest reward-effect gap 0.009.
  - **The RL models** miss by 0.12–0.23.
  - **Counterfactual:** flipping one reward moves the GRU's next log-odds by +1.00,
    decaying to +0.017 at 10 trials.

![fig_C1_kernels.png](analysis/fig_C1_kernels.png)
*`fig_C1_kernels.png`*

### C2c · C3 — where the advantage comes from
- **Per trial,** the GRU gains most in uncertain, post-switch states.
- **In total,** most of its advantage comes from the many ordinary trials after an
  unrewarded choice.
- **Against CTT,** the advantage rises with the predicted chance of switching (about 0.01
  nats at P(switch) < 0.1, rising to 0.03–0.04 at 0.4–0.5) and falls with the GRU's
  confidence.
- **The RL value variables are inside the GRU state:**
  - ΔQ decodes at R² 0.95–0.97 (on PC1) and ΣQ at 0.77–0.87 (on PC2);
  - together they explain 94% of the GRU's logit.

![fig_C2c_advantage.png](analysis/fig_C2c_advantage.png)
*`fig_C2c_advantage.png`*

![fig_C2c_advantage_pc_pairs.png](analysis/fig_C2c_advantage_pc_pairs.png)
*`fig_C2c_advantage_pc_pairs.png`*

![fig_C3_q_decoders.png](analysis/fig_C3_q_decoders.png)
*`fig_C3_q_decoders.png`*

### C4a · C4b · C4c — the RL models fail in specific, opposite ways around losses
- **How.**
  - **C4b:** bin trials by each model's own logit within each input, and compare its
    predicted next-trial log-odds with the animals' empirical rate.
  - **C4c:** calibration over all 64 three-trial histories.
- **Result.**
  - **Updates:** the GRU's sit on the animals' (RMS ≤ 0.05 log-odds); the RL models deviate
    by 0.05–0.42.
  - **History calibration, mean |error|:** GRU 0.18 pp, CTT 1.10, Bari 2.13, Hattori 2.24.
  - **Hattori under-persists after losses:** after R−, R−, L− the animals choose R 69%;
    Hattori predicts 53%.
  - **Bari and CTT under-return after a failed switch** (errors ±14–17 pp).

![fig_C4a_logit_updates.png](analysis/fig_C4a_logit_updates.png)
*`fig_C4a_logit_updates.png`*

![fig_C4b_update_calibration.png](analysis/fig_C4b_update_calibration.png)
*`fig_C4b_update_calibration.png`*

![fig_C4c_history_calibration.png](analysis/fig_C4c_history_calibration.png)
*`fig_C4c_history_calibration.png`*

### C4d · C4e — when the GRU and RL disagree, the animals side with the GRU
- **Disagreements:** the 5% of trials where the GRU and CTT disagree most carry 32% of the
  total advantage.
  - 55% of them are the first trial after a switch (13% overall).
  - The animals follow the GRU in both directions.
- **Return after a failed switch** grows with the run left behind: animals 0.39 → 0.53.
  - Hattori overshoots (0.47 → 0.62), Bari undershoots (0.31 → 0.43), CTT is flat (0.41).
  - The GRU's error is ≤ 0.005.
- **Staying through losses after a switch** rises with each loss: animals 0.56 → 0.84.
  Hattori stays nearly flat (0.47 → 0.58).

![fig_C4d_disagreement.png](analysis/fig_C4d_disagreement.png)
*`fig_C4d_disagreement.png`*

![fig_C4e_signatures.png](analysis/fig_C4e_signatures.png)
*`fig_C4e_signatures.png`*

---

## 3. Dynamics and attractors (r3; per-mouse version in r5)

### D1 · D2 · F2 — one attractor per input; slow modes off the readout
- **How.**
  - Fixed points of each constant input (choice × reward held fixed), found by forward
    iteration and damped Newton in float64, with Jacobian spectra at each.
  - D2 used 5 group contexts; F2 used every mouse's context (96 starts per mouse and input).
- **Result.**
  - **Attractors:** one stable attractor per input on the chosen side in 612 of 614 mice.
    Median logits: R+ +4.36, R− +3.08, L+ −3.72, L− −3.07. "L rewarded" is bistable in at
    least 78 mice.
  - **Slow modes:** very slow (τ ≈ 120–650 trials) but barely touching the readout (overlap
    ~0.11).
    - They don't explain the 2–3-trial behavioural timescale.
    - The embedding barely sets them (rank R² 0.14).
  - **Across mice:**
    - **bias shifts the attractors** (ρ +0.57);
    - **reward gain is a speed, not a depth:** it tracks v's one-step retention (ρ −0.46),
      not attractor depth (ρ +0.28);
    - **persistence tracks depth** (ρ +0.50).
  - **How the state reaches the attractor (D1b):** 300 real states were iterated under each
    constant input for 600 steps.
    - **The fixed points are correct:** every path ends on D's stable fixed point, matching
      it in PCs 1–6.
    - **One step covers most of the distance:** median |Δh| about 2.5 at step 1, about 0.1
      by step 10.
    - **The rest is a slow drift** over 30–600 steps. It is longest for unrewarded inputs,
      where PC2 falls from 1.5 to 1.0 in a region the data rarely visits.
    - **D1's one-step arrows therefore converge where one step lands, not on the attractor.**
    - **For "L rewarded",** every real start reaches the shallow attractor (logit −4.04).
      The deep one (−4.98) is a separate basin that no real start enters.

![fig_D1_flow_fields.png](analysis/fig_D1_flow_fields.png)
*`fig_D1_flow_fields.png` — one-step displacements (arrows drawn at full length), not trajectories.*

![fig_D1b_convergence.png](analysis/fig_D1b_convergence.png)
*`fig_D1b_convergence.png` — repeating each input from 300 real states; colour = number of steps.*

![fig_D1e_flow_own_context.png](analysis/fig_D1e_flow_own_context.png)
*`fig_D1e_flow_own_context.png` — each state stepped with its own session context. Each dot
is one mouse's attractor; the 614 attractors line up along each input's depth direction.*

![fig_D1f_flow_prototypes.png](analysis/fig_D1f_flow_prototypes.png)
*`fig_D1f_flow_prototypes.png` — prototype mice from the named axes. The main difference is
where each mouse's states sit; the attractors shift only modestly.*

![fig_D1d_flow_scaled.png](analysis/fig_D1d_flow_scaled.png)
*`fig_D1d_flow_scaled.png` — arrow length ∝ step size, linear (top) and log10 (bottom); the log version flows onto the fixed points.*

![fig_D1c_flow_two_scales.png](analysis/fig_D1c_flow_two_scales.png)
*`fig_D1c_flow_two_scales.png` — the flow at two scales. Top: one-step direction, coloured by
step size, whose minimum is at the fixed point. Bottom: zoomed slow drift onto the fixed point
after ≥ 5 repeats.*

![fig_D2_fixed_points.png](analysis/fig_D2_fixed_points.png)
*`fig_D2_fixed_points.png`*

![fig_F2_mouse_dynamics.png](analysis/fig_F2_mouse_dynamics.png)
*`fig_F2_mouse_dynamics.png`*

---

## 4. The value coordinate v (r6)

### I1 · I2 — how v is built, and what is by construction
- **How (targeted dimensionality reduction).**
  - Regress the state on: previous choice, previous reward and their product; the GRU's own
    logit and logit × choice; and nuisances (context, trial position).
  - The value axis is the state's encoding of the logit, averaged over the two lobes; v is
    each trial's projection on it.
- **Result.**
  - **Geometry:** two integration axes, set by the previous choice, not by the outcome.
    - Cosine 0.91–0.92 within a lobe, 0.35–0.41 across lobes.
    - All four cluster axes have cosine 0.80–0.84 with the shared axis.
  - **Value-like, not task-state-like:** the within-cluster value axis is parallel to the
    axis encoding Hattori's ΔQ (cosine 0.94–0.96), not to the true reward-probability
    difference (0.12–0.26).
  - **v is not the logit:**
    - logit ≈ 1.02·v + an offset set by the last input, R² 0.944;
    - the value axis has cosine 0.50 with the readout.
- **By construction:** v tracks the choice, is shared across clusters and is linear, and the
  other TDR axes are orthogonal to the readout.
- **Findings:**
  - the two-lobe geometry;
  - the alignment with ΔQ;
  - carry-through across switches;
  - one update rule;
  - causal use (E2b);
  - what v integrates (I7, I8).

![fig_I2b_value_axis_in_pcs.png](analysis/fig_I2b_value_axis_in_pcs.png)
*`fig_I2b_value_axis_in_pcs.png` — states coloured by v, with v's axis, the two lobe axes,
the readout and the choice and reward axes projected into PC planes. v's axis is 83% in PCs
1–4, mostly −PC1 with a PC4 part; the lobe axes tilt apart along PC2.*

![fig_I1_cluster_axes.png](analysis/fig_I1_cluster_axes.png)
*`fig_I1_cluster_axes.png`*

![fig_I2_tdr.png](analysis/fig_I2_tdr.png)
*`fig_I2_tdr.png`*

### I3 · I4 — v is carried through every jump; one simple rule describes it
- **How.**
  - **I3:** regress v(t+1) on v(t) per input and stay/switch.
  - **I4:** fit v(t+1) = v* + λ(v(t) − v*) per group, and a readout logit ≈ a·v + b[last
    input]. Then simulate v from the animals' choices and rewards alone, with no GRU.
- **Result.**
  - **Carry-through:** slope 0.78–0.80 on stays, 0.66–0.74 on switches; R² 0.93–0.98
    throughout. Inputs mostly move the state along the choice and reward axes.
  - **Targets:** rewarded stay ±2.7–2.9, unrewarded stay ±1.7–2.0, switch ≈ 0. Unrewarded
    stays still push v toward the chosen side.
  - **Simulation:** the simulated v tracks the GRU's (corr 0.985) and scores 0.718 on the
    animals (GRU 0.734). Of the gap, about 0.010 is the readout and 0.006 the dynamics.

![fig_I3_tdr_dynamics.png](analysis/fig_I3_tdr_dynamics.png)
*`fig_I3_tdr_dynamics.png`*

![fig_I4_distilled_model.png](analysis/fig_I4_distilled_model.png)
*`fig_I4_distilled_model.png`*

![fig_I5_mouse_dynamics.png](analysis/fig_I5_mouse_dynamics.png)
*`fig_I5_mouse_dynamics.png`*

### I6 · I7 · I8 — what v integrates, compared with ΔQ
- **How.**
  - **I6:** the same transition fits on RL ΔQ.
  - **I7:** a second carried coordinate, and a history regression of v.
  - **I8, causal tests:** closure, a perturbation along v, and flips of one past choice or
    reward.
- **Result.**
  - **I6:** v follows one affine rule even after switches (R² ≥ 0.93); RL's ΔQ does not
    (R² 0.12–0.86). After a switch, ΔQ depends on the two Q values separately.
  - **I7:** a second coordinate, the choice × reward axis ("lose → avoid", a memory of
    unrewarded choices), closes 34% of the remaining gap (simulated 0.7234).
  - **I8, closure:** v(t) alone predicts v(t+1) at held-out R² 0.93–0.98, against 0.9995
    for the full state. So v is approximately self-contained.
  - **I8, perturbation:** a push along v decays 0.69, 0.48, 0.36 … 0.15 at k = 10, and half
    of it leaves v's axis after one step. v is a view of a multi-timescale system, not a
    single mode.
  - **I8, flips:**
    - **In v,** past choices matter far more than their rewards (k = 2: unrewarded choice
      0.85, rewarded 1.35, reward alone 0.26).
    - **At the policy level,** the GRU's unrewarded / rewarded ratio is flat across lags
      (0.39–0.48); Hattori's rises (0.39 → 1.31).
    - **Reward vs choice memory:** the GRU forgets rewards faster than choices (reward share
      0.31 → 0.06); in Q-learning both persist in Q.

![fig_I6_rl_transitions.png](analysis/fig_I6_rl_transitions.png)
*`fig_I6_rl_transitions.png`*

![fig_I7_second_variable.png](analysis/fig_I7_second_variable.png)
*`fig_I7_second_variable.png`*

![fig_I8_v_tests.png](analysis/fig_I8_v_tests.png)
*`fig_I8_v_tests.png`*

### I9 — what v integrates, as kernels; does "choice trace × outcome" suffice?
- **How.**
  - v regressed on the last 10 choices (±1) and rewarded choices, scored by held-out R².
  - Three descriptions: free (21 parameters); one choice trace scaled by outcome (4); and a
    choice trace plus a separate reward trace (5).
  - Hattori and Bari ΔQ fitted the same way for contrast.
- **Result.**
  - **Kernels:** v's rewarded-choice weights are 0.48, 0.64, 0.45, 0.29 (lags 1–4); its
    unrewarded-choice weights are 0.26, 0.41, 0.32, 0.23. Both peak at lag 2.
  - **Fit:** free R² 0.953; one-trace-scaled 0.945 (λ 0.80, reward ×1.29); two-rate 0.947
    (choice trace λ 0.82, reward trace μ 0.62).
  - **So v is a choice memory decaying about 0.8 per trial, plus a roughly 30% reward boost
    that fades faster.** The unrewarded/rewarded weight ratio rises with lag (0.55 → > 1).
  - **R² does not separate the models:** ΔQ fits these descriptions as well (Hattori
    0.940–0.943, Bari 0.914–0.915).
  - **The kernel shapes do:**
    - Hattori's ratio falls (0.58 → 0.25);
    - Bari's ΔQ is almost reward-only (ratio about 0.1);
    - v's rises.
  - **Note:** this regression is marginal over correlated histories. I8's causal flips
    instead give a flat ratio of choice effects (about 0.65). Both agree the reward part is
    shorter-lived.

![fig_I9_v_kernel.png](analysis/fig_I9_v_kernel.png)
*`fig_I9_v_kernel.png`*

### I9b — the same kernels per mouse
- **How.** I9's three fits per mouse (610 mice) for v and each mouse's own Hattori and Bari
  ΔQ; odd sessions held out; split-half reliability.
- **Result.**
  - **Not a pooling artefact.** In v, the reward trace fades faster than the choice trace in
    99% of mice, and the unrewarded/rewarded ratio rises in 99.8%. Hattori shows the
    opposite in about 98–99% of mice.
  - **v's choice-memory timescale** λ: median 0.79, 10th–90th percentile 0.75–0.83,
    reliability 0.95.
  - **Reward boost** κ: 0.34 (0.18–0.53).
  - **λ is the slow-learning embedding axis** (ρ +0.70). Mice with a longer choice memory
    in v show longer animal choice memory (+0.74), less win-stay (−0.79) and more switching
    (+0.69).
  - **κ tracks win-stay** (+0.56).
  - **Side bias is unrelated to the kernel.**

![fig_I9b_v_kernel_per_mouse.png](analysis/fig_I9b_v_kernel_per_mouse.png)
*`fig_I9b_v_kernel_per_mouse.png`*

### I9c — kernel variants: lag 1 absorbed, and two outcome traces
- **How.**
  - The last trial was moved into one offset per last-trial cluster, with kernels fitted
    on lags 2–10 (within-cluster R²).
  - A two-outcome family (separate amplitude and decay for rewarded and unrewarded
    choices) was compared with "choice + reward". Pooled and per mouse.
- **Result.**
  - **Lag 1 doesn't distort the rest.** v's kernel on lags 2–10 is unchanged.
  - **Older history explains more of v than of ΔQ:** 86% of v's within-cluster variance,
    against 67% (Hattori ΔQ) and 74% (Bari ΔQ). v is closer to a linear filter of past
    choices and outcomes than Q-learning is.
  - **Two traces:** rewarded choices weigh 0.64 and decay 0.68 per trial; unrewarded
    choices weigh 0.36 but decay 0.83. Unrewarded choices decay more slowly in 100% of
    mice (Hattori: 7%). I9d refines this: the 0.83 averages a fast trace and a slow residue.
  - **Rewarded-choice amplitude is the determinism axis** (ρ +0.83).
  - **The timescales and the unrewarded weight are the slow-learning axis** (λ_rew +0.65,
    A_unr +0.53).

![fig_I9c_v_kernel_variants.png](analysis/fig_I9c_v_kernel_variants.png)
*`fig_I9c_v_kernel_variants.png`*

### I9d — longer history and three traces

- **How.** I9c's offsets fits at K = 20 lags (K = 10 refitted on the same rows), adding a
  six-parameter "three traces" model: the two outcome traces plus a shared choice trace.
  Held-out within-cluster R² and per-mouse BIC.
- **Result.**
  - **Three traces nearly matches the free fit** (gap 0.0006 against 0.011 for two outcomes),
    and has the lowest BIC in 99.5% of mice.
  - **The components:** a choice trace on every choice (λ ≈ 0.67), a reward boost at a
    similar timescale (λ ≈ 0.71), and a small unrewarded residue (≈ 0.045) that does not decay
    within 20 trials.
  - **Correction to I9c.** "Unrewarded choices have a longer memory" (λ_unr 0.83) was a
    compromise between the fast trace and the slow residue. Both outcomes share the fast
    timescale.
  - **Caveat.** The residue is flat over the window and its decay sits at the fit bound, so it
    may be a session-level side-preference signal rather than a trace; untested.

![fig_I9d_v_kernel_long.png](analysis/fig_I9d_v_kernel_long.png)
*`fig_I9d_v_kernel_long.png`*

### I9e — the three-trace model as an update rule, fitted to the mice

- **How.** Choice, reward and unrewarded-choice traces (T ← λT + input) feeding a logistic
  choice rule. Fitted per mouse to the animals' choices and scored on held-out sessions
  (2-fold), against Q-learning refitted the same way. The GRU and the wrapper's RL fits are a
  reference.
- **Result.**
  - **The best simple model we have**, at a held-out likelihood of 0.7231. That compares with
    0.7145 for refitted Q-learning, 0.7219 for Q + choice kernel + last trial, and 0.7210 for
    the wrapper's CTT. It closes 45% of the gap from Q-learning to the GRU (0.7339). It beats
    Q + kernel in 68% of mice and never beats the GRU.
  - **The mice show v's slow unrewarded residue.** An unrewarded trace with λ_u ≈ 0.98, slower
    than the reward trace in 99% of mice, pushing towards the chosen side in 85%.
  - **But the behavioural choice trace is fast** (λ_c 0.39 against v's 0.65). The reward-trace
    parameters agree across mice with v's own fit (ρ +0.64); the residue doesn't.
  - **It does not fix the switch signatures.** On the C4e return and stay curves it does no
    better than Q + kernel, and the GRU is about 10× closer. The GRU's remaining advantage is
    around switches, not in the kernel.

![fig_I9e_trace_models.png](analysis/fig_I9e_trace_models.png)
*`fig_I9e_trace_models.png`*

### I10 — how the GRU holds v
- **How.** v's exact one-step retention (eᵀJe) at 20,000 real transitions per input group,
  split into three terms: the update gate's leak (1 − z), the gate's own change, and the
  recurrent candidate that re-writes the state.
- **Result.**
  - **The update gate is mostly open along v** (z̄ 0.65–0.69), so the leak keeps only
    0.31–0.35 per trial.
  - **The recurrent candidate term adds 0.32–0.40,** for a total of 0.66–0.72, which matches
    I8's perturbation decay (0.69). v is maintained by recurrent re-computation, not by a
    closed gate.
  - **Local retention is the same on stays and switches.** The higher fitted slopes on
    stays come from other state components that covary with v.
  - **v is distributed:** participation ratio 32 of 128 units; the top 10 units hold 48%.

![fig_I10_gates.png](analysis/fig_I10_gates.png)
*`fig_I10_gates.png`*

---

## 5. Subspaces and ablations (r4)

### E1 · E2b — which directions carry which memory
- **How.**
  - **E1:** the geometry of the input, decoder and readout directions.
  - **E2b:** at every step, replace the state's component along an axis with the mean of
    its own last-input cluster (keeping the state on the manifold). Memory only, against 10
    random 1-d and 2-d controls.
- **Result.**
  - **Readout:** the readout direction carries only 9.9% of state variance.
  - **Reward axis:** removing it costs −0.081 likelihood (random 1-d: −0.013) and halves the
    one-trial reward effect (0.93 → 0.40).
  - **Memory beyond two trials:** removing the readout coordinate or both lobe value axes
    abolishes it (lag 3: 0.29 → 0.09 / 0.04; random ≥ 0.17).
  - **v alone:** removing it costs more likelihood than random directions (−0.028), but its
    kernel stays in the random range.
  - **A 1-d carried state** keeps the model at 0.713.

![fig_E1_subspaces.png](analysis/fig_E1_subspaces.png)
*`fig_E1_subspaces.png`*

![fig_E2b_conditional_ablations.png](analysis/fig_E2b_conditional_ablations.png)
*`fig_E2b_conditional_ablations.png`*

---

## 6. How the GRU's strategy differs from Q-learning (r8)

### H3 — in Q-learning's coordinates: a delta rule after rewards, not after losses
- **How.**
  - Decode Hattori's Q_L and Q_R from the GRU state (held-out R² 0.95 / 0.94).
  - Draw the one-trial change after each input for the GRU, Hattori and Bari.
  - Regress each value's change on both values.
- **Result.**
  - **After a reward** the GRU moves the chosen value at rate 0.42–0.43 and forgets the
    unchosen at 0.69–0.72, close to Hattori (0.40–0.41, 0.75).
  - **After a loss:**
    - Hattori leaves the chosen value almost untouched (0.05);
    - Bari drives it toward 0 (0.25);
    - the GRU relaxes it slowly (0.10–0.13) toward about 0.25–0.4, so a loss doesn't erase
      the chosen side's value.

![fig_H3_q_phase_portraits.png](analysis/fig_H3_q_phase_portraits.png)
*`fig_H3_q_phase_portraits.png`*

### H4 — identical scripted histories through every model
- **How.** Each mouse starts a fresh session, with its context (GRU) or its fitted agents
  (RL), and runs through the same scripted choices and outcomes. Medians across mice.

| probe | GRU | Hattori | Bari | CTT |
|---|---|---|---|---|
| Return after one failed switch, after 1 → 12 rewarded trials on the old side | 0.46 → 0.73 | 0.58 → 0.66 | 0.36 → 0.63 | 0.41 flat |
| Stay on the new side through 1 → 10 losses | 0.28 → 0.80 | 0.34 → 0.49 | 0.44 → 0.70 | 0.59 → 0.80 |
| P(right) after an unrewarded switch to the left | 0.73 | 0.66 | 0.59 | 0.41 |
| P(right) after a rewarded switch to the left | 0.43 | 0.28 | 0.40 | 0.23 |

- **Result.**
  - The GRU weighs one new outcome against the run that preceded it.
  - Unrewarded stays build its choice memory.

![fig_H4_scenarios.png](analysis/fig_H4_scenarios.png)
*`fig_H4_scenarios.png`*

### H2 — the GRU's learning rate barely saturates
- **How.** Flip one reward on 200,000 trials and measure the next trial's log-odds (the
  effective learning rate), binned by state.
- **Result.**
  - **Mean effect:** GRU 1.00, Hattori 0.68, Bari 0.75.
  - **Hattori's effect saturates** as Q nears 1:
    - it falls with run length (0.96 → 0.47) and with the recent reward rate
      (1.14 → 0.36);
    - it is twice as large after a switch (1.33 vs 0.59).
  - **The GRU's effect is flat** over run length (0.92–1.07), confidence and stay/switch,
    and falls only moderately with the reward rate (1.12 → 0.83).
  - **Where the GRU does modulate:** rewards count more after a dry spell (0.85 → 1.2) and
    for unexpected choices (1.28).
  - **Interpretation:** this is not the meta-learned "learning rate scales with
    uncertainty" pattern. The GRU's update is mainly non-saturating.

![fig_H2_learning_rate.png](analysis/fig_H2_learning_rate.png)
*`fig_H2_learning_rate.png`*

### H1 — the model ladder: a choice trace is the component that matters
- **How.**
  - Relax Q-learning one component at a time.
  - Fit each rung per mouse to the GRU's predictions (GPU, Adam, soft labels).
  - Score on the animals.

| rung | animal likelihood | mice beating CTT |
|---|---|---|
| R0 Q-learning | 0.7169 | 14% |
| R1 + one-trial choice kernel | 0.7212 | 54% |
| R2 + decaying choice trace | 0.7240 | 88% |
| R3–R5 + outcome scaling, switch reset, split kernel | 0.7251 | 97% |
| GRU (ceiling) | 0.7339 | 100% |

- **Result.**
  - **R2's trace** keeps a median 0.72 per trial (IQR 0.63–0.97) and fixes the "stay
    through losses after a switch" signature (error 0.136 → 0.006).
  - **The plateau** closes 32% of the CTT → GRU gap.
  - **The remainder** is not one more additive component; H2 points to a non-saturating
    update.

![fig_H1_model_ladder.png](analysis/fig_H1_model_ladder.png)
*`fig_H1_model_ladder.png`*

---

## 7. The subject embedding (r5)

### F3 · F1 — the embedding predicts each mouse's behaviour and RL parameters
- **How.** Decode per-mouse features from the embedding: 5-fold CV rank R², a
  200-permutation shuffle null, and a task-mix control.
- **Result.**
  - **Study 01 replicates:** RL bias 0.73, learning rates 0.65–0.67.
  - **The GRU's own features:** bias, gain and persistence decode at 0.78–0.80. They track
    the animals' per mouse (Spearman 0.95, 0.80, 0.96).
  - **Not a task-mix effect:** task mix alone gives ≤ 0.26.
  - **Directions:** bias and gain are separate (cosine −0.05); gain, persistence and the RL
    learning rates share one (0.92–0.94).
  - **F1, the extremes:** for high-gain mice the GRU jumps toward a rewarded side; for
    low-gain mice it updates gradually. The animals' reward effect at the two ends: 1.33 vs
    0.53.

![fig_F3_embedding_features.png](analysis/fig_F3_embedding_features.png)
*`fig_F3_embedding_features.png`*

![fig_F1_embedding_extremes.png](analysis/fig_F1_embedding_extremes.png)
*`fig_F1_embedding_extremes.png`*

### F7 — a behavioural feature library, with reliability ceilings
- **How.**
  - 13 features per mouse, from the animals' choices and from the GRU's predictions:
    - determinism (the overall scale of history dependence);
    - stickiness ratio (unrewarded ÷ rewarded weight);
    - win-stay and lose-stay;
    - reward and choice memory spans;
    - bias;
    - matching sensitivity and matching bias;
    - poorer-side rate and switch rate;
    - return after a failed switch, and stay through losses.
  - Reliability from odd vs even sessions (Spearman–Brown).
- **Result.**
  - **Gain, stickiness and persistence were one dimension:** they correlate 0.80–0.95 across
    mice in the GRU, all scaling with determinism (like an RL inverse temperature).
  - **The scale-free version separates them:** the stickiness ratio is nearly independent
    of determinism in the animals (+0.1).
  - **Near their reliability ceiling in the embedding** (rank R² 0.6–0.74): animal
    determinism, win-stay, lose-stay, switch rate, bias and matching bias.
  - **Reliable but weakly carried:** the animals' stickiness ratio (reliability 0.73, R²
    0.10) and matching sensitivity (0.67, 0.13).

![fig_F7a_feature_library.png](analysis/fig_F7a_feature_library.png)
*`fig_F7a_feature_library.png`*

### F7 · F8 — named axes, verified causally
- **How.**
  - **F7:** varimax rotation of the GRU features' loadings on the whitened embedding.
  - **F8:** synthetic mice at −2 to +2 SD along each axis. The GRU is run on 600 real
    sessions, features are measured from its predictions, and the causal slopes are
    compared with the loadings.

| axis | what moves together | causal agreement |
|---|---|---|
| 1 · determinism / stickiness | determinism +0.91, lose-stay +0.90, win-stay +0.85, switch rate −0.89; GRU likelihood +0.88 | 0.94 |
| 2 · side bias | bias +0.92, matching bias +0.88; nothing else | 0.92 |
| 3 · slow learning / long choice memory | choice memory span +0.66, poorer-side rate +0.58, return after failed switch +0.45; RL learning rates −0.70 to −0.74 | 0.78 |
| 4 · dynamics-only | v retention −0.62, attractor asymmetry +0.54; almost no behavioural loading | 0.71 |

![fig_F7b_named_axes.png](analysis/fig_F7b_named_axes.png)
*`fig_F7b_named_axes.png`*

![fig_F8_embedding_traversal.png](analysis/fig_F8_embedding_traversal.png)
*`fig_F8_embedding_traversal.png`*

### F9 — a strategy profile for every mouse
- **Profiles table:** [`../f9_mouse_strategy_profiles.csv`](analysis/f9_mouse_strategy_profiles.csv)
  gives each mouse's coordinates and percentiles on the four axes, with its own behavioural
  features.
- **Prototypes** at each end of each axis. For example:
  - mouse 802252 (99th percentile on determinism): strong rewarded and unrewarded weights,
    switch rate 0.06;
  - mouse 455526 (7th percentile): weak history dependence, frequent switching.
- **Per-mouse strategy cards:** in `analysis/_cache/explore/cards/` (gitignored). Regenerate
  any mouse with `python analysis/f9_strategy_cards.py --mouse <id>`.

![fig_F9_strategy_prototypes.png](analysis/fig_F9_strategy_prototypes.png)
*`fig_F9_strategy_prototypes.png`*

### F5 — atlas and setup labels: no setup signature beyond behaviour
- **How.**
  - The embedding's PCs are coloured by, and correlated with, every variable.
  - Setup labels (institute, data source, room, trainer, PI, curriculum, first year) are
    decoded four ways: from the embedding, from task mix, and from the animals' kernels;
    and as the embedding's incremental gain over animal behaviour + task mix, with an
    embedding-shuffle null.
- **Result.**
  - **Decodable:** strategy and dynamics (rank R² 0.5–0.85).
  - **Not decodable:** non-choice traits such as reaction time, weight, water and ignore
    rate (≤ 0.08).
  - **The embedding's PCs:**
    - PC3 = learning rate / gain (Bari α −0.83);
    - PC4 = bias;
    - PC2 = attractor depth (+0.92);
    - PC1 mixes these with predictability.
  - **Setup labels:**
    - The embedding alone decodes institute above chance (0.79, against a null of 0.62).
    - Task mix does better (0.93).
    - Adding the embedding to behaviour + task mix gains nothing for any label (−0.05 to
      +0.028).

![fig_F5a_embedding_atlas.png](analysis/fig_F5a_embedding_atlas.png)
*`fig_F5a_embedding_atlas.png`*

![fig_F5b_embedding_directions.png](analysis/fig_F5b_embedding_directions.png)
*`fig_F5b_embedding_directions.png`*

![fig_F5c_embedding_batch.png](analysis/fig_F5c_embedding_batch.png)
*`fig_F5c_embedding_batch.png`*

### F6 — a continuum, not discrete strategy types
- **How.**
  - Gaussian mixtures (k = 1–8), against a single-Gaussian null and a heavy-tailed
    (multivariate t) null.
  - Mantel tests of embedding distance against behavioural distance.
- **Result.**
  - **Clusters:** BIC prefers 2 components (gain 74). They are a core (65%) plus a broad
    halo, with means about 1 SD apart: what heavy tails alone produce (t-null 95th
    percentile 74).
  - **Distances correspond:** mice close in the embedding behave alike (Spearman 0.49 with
    the animals' kernels, 0.57 GRU, 0.55 RL parameters; null about 0.03).

![fig_F6_embedding_types.png](analysis/fig_F6_embedding_types.png)
*`fig_F6_embedding_types.png`*

### F4 · F4b · F4c — how each mouse's context moves with training
- **How.**
  - **F4:** early vs late halves of each mouse's sessions; GRU vs animal kernel changes.
  - **F4b:** rerun late sessions with the early context (counterfactual).
  - **F4c:** the network's context for every training session of every mouse (17k
    sessions).
- **Result.**
  - **Per-mouse change:** the GRU follows each mouse's early→late change to the noise
    ceiling (bias 0.92, gain 0.59, persistence 0.93). But its average gain falls (−0.035)
    while the animals' rises (+0.047).
  - **Size of the movement:** only 3% of context variance is within mice, but the distance
    isn't small.
    - Median first→last change: +0.80 between-mouse SD along PC2, in 99.5% of mice.
    - The movement is front-loaded and non-monotonic: bias swings up, then partly back.
  - **Effect of the drift:** it explains about a third of the GRU's average gain drop, and
    ≤ 6% of the per-mouse changes. The rest comes from each mouse's own history.

![fig_F4_early_late.png](analysis/fig_F4_early_late.png)
*`fig_F4_early_late.png`*

![fig_F4b_context_drift.png](analysis/fig_F4b_context_drift.png)
*`fig_F4b_context_drift.png`*

![fig_F4c_context_trajectories.png](analysis/fig_F4c_context_trajectories.png)
*`fig_F4c_context_trajectories.png`*

---

## 8. Caveats and corrections

- **One seed of one model — partly lifted (r7, 2026-10-27).**
  - Seeds 1–2 of model C replicate 14 of 18 headline claims; see r7 for the four that fail
    in seed 1.
  - Three claims are seed-specific, so read them as seed 0's: "embedding PC3 = learning
    rate, PC4 = bias" (the *directions* replicate, the PC numbering does not), the
    L-rewarded multiple fixed points and bistable mice (D, F2), and the slow-mode τ values.
  - Model B (H256) is still untested.
- **The models are fitted to these mice.**
  - The GRU has about 55k parameters, against 3–6 per mouse for the RL fits. So the
    informative part is where and how each RL model fails, not that the GRU passes.
  - The ladder (H1) was fitted on eval sessions, though only to the GRU's predictions.
- **Many results are correlational across mice.** The causal ones are all in the model:
  - the counterfactual flips (C1, I8, H2);
  - the ablations (E2b);
  - the fixed points (D, F2);
  - the context counterfactual (F4b);
  - the embedding traversal (F8).
- **Corrections made during the study** (all in [`../../CHANGELOG.md`](CHANGELOG.md)):
  - **B5:** the long-lasting reward effect is environment-confounded, not causal.
  - **"Choice kernel vs ΔQ" was overstated:** ΔQ with fitted forgetting is also
    choice-dominated. The GRU-specific properties are a separable choice trace and a short
    reward memory.
  - **D2's "reward gain deepens attractors"** was revised by F2: gain is a speed, and depth
    tracks persistence.
  - **The F5c setup-label test** was redesigned after it turned out circular; the
    conclusion held.
  - **"The context barely moves" was wrong:** a 3% variance share hides a drift of up to
    about 1 between-mouse SD.
  - **A four-reviewer correctness pass** fixed two binning bugs (B6, H2), an incomplete
    fixed-point search (F2) and an initialisation artefact (H1), plus about 40 smaller
    wording and number fixes.

---

## 9. Open questions

- **Close the GRU–ladder gap:** a ladder rung with a non-saturating value update, refitted on
  training sessions.
- **Validate the named axes on the animals' raw behaviour,** not only on the GRU's.
- **Why is the animals' stickiness ratio reliable but barely encoded** (0.73 vs 0.10)? It
  could be a model limitation or a sampling one.
- **Loose ends:**
  - the slow off-axis tail of v;
  - whether v plus the readout reproduces B4's nonlinear maps.
- **Robustness (phase G):** seeds 1–2 of model C, and model B; align the embedding axes
  across seeds.
