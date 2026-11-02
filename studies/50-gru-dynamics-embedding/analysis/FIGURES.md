# Study 50 — figure guide

How each committed figure is made and what it shows, for model **C seed 0** (GRU, 128 units,
4-d subject embedding + scalar session conditioning, trained on 614 mice; best_eval step
90 000). Status as of 2026-10-26; the reports in `reports/` (r1–r6, r8) carry the full numbers,
provenance and caveats. Every figure has one producer script in this folder, and most write a JSON with the numbers
they show (`b3c_history.py`, `c2c_pc_pairs.py` and `c1_figure.py` don't); `make -C studies/50-gru-dynamics-embedding <target>` regenerates it.

## Data and conventions used throughout

- **Trials.** The 614 training mice's eval sessions (every second session): 4,339,402
  trials in 8,903 sessions, from one forward pass of the model (A1, `extract_trial_states.py`).
  The recomputed likelihood (0.7338507) reproduces the logged eval likelihood (0.7338506).
- **Timing.** Row t's hidden state has seen trial t−1's choice and reward and produces the
  prediction for trial t. "1 back" = trial t−1.
- **Model architecture.** Input (previous choice, previous reward, 4-d context) → GRU (128) →
  linear readout (2 logits). The context = the mouse's embedding + a small session-dependent
  correction. Because the readout is linear, the choice depends on one direction of the
  state, the **readout direction** r = w_R − w_L; logit(right) = r·h + (b_R − b_L).
- **Task.** Each session's task comes from the snapshot session table (Uncoupled Baiting 4,733
  sessions, Coupled Baiting 2,446, Uncoupled Without Baiting 1,433, Coupled Without Baiting 285).
- **PCA.** One PCA of the 128-d state over all trials (B3a); every state-space figure uses
  these axes. Scatter plots draw a fixed random subsample (150k trials) for legibility.
- **"Clusters".** The four groups of states defined by the previous trial's choice × outcome
  (R rewarded, R unrewarded, L rewarded, L unrewarded).
- **RL baselines.** Study 01's per-mouse fits of Bari (Q-learning + choice kernel), CTT
  (compare-to-threshold) and Hattori (asymmetric learning rates), replayed on the same trials;
  each replay reproduces its run's logged likelihood for every mouse.
- **Compute.** The container lost GPU access during phase C; C1, D, I3, E2, E2b and I8 ran on CPU; F2, H1, H2, H4 and F4b ran on GPU after the 2026-10-23 restart.
  Their unedited replays match the GPU logits to median 0.0009.

---

## Phase B — what the model and its state look like (report r1)

### `fig_B1_example_sessions.png` — example sessions
- **How.** One session per task: the session whose length is closest to that task's median.
  It plots the true reward probability of each side, the animal's choices (ticks: right above,
  left below; tall = rewarded), a 10-trial running mean of the animal's choices, and the
  model's P(right). `b1_example_sessions.py`.
- **Result.** The model's P(right) tracks the animal closely in all four tasks. Per-session
  likelihoods: 0.72 (Uncoupled Baiting), 0.87 (Uncoupled Without Baiting), 0.82 (Coupled
  Without Baiting) and 0.56 (Coupled Baiting). The Coupled Baiting example is a noisy 2019
  session, representative of its task's session length but not of the model's fit.

### `fig_B2_unit_heatmap.png` — all 128 units over one session
- **How.** The hidden state of every unit over the Uncoupled Baiting example session. Units are
  sorted by their correlation with the model's logit. `b2_unit_heatmap.py`.
- **Result.** Activity is dense, not sparse:
  - 64 units have |corr| > 0.5 with the logit;
  - 9% of values are saturated (|h| > 0.95);
  - three kinds of unit are visible: units that flip with choice blocks, units that are nearly
    constant within the session, and units that alternate trial to trial (outcome responses).

### `fig_B3a_pca_variance.png` — dimensionality
- **How.** PCA of the state over all 4.3M trials (float64 covariance, streamed). It shows the
  variance explained by each of the first 20 PCs and the cumulative curve. `b3_pca.py`.
- **Result.**
  - PC1 explains 42.7% of the variance, PC2 18.0%, PC3 7.6%, PC4 6.8% and PC5 5.6%.
  - The participation ratio is 4.3.
  - 9, 14 and 39 PCs reach 90%, 95% and 99% of the variance.
  - So the state is effectively low-dimensional.

- **Truncated version `fig_B3a_pca_variance_10.png`:** the same spectrum for PCs 1–10 only
  (percentages still of the total). The first 10 PCs hold 92.5%. Made by
  `b3a_pca_variance_10.py` from the cached PCA, without refitting.

### `fig_B3b_state_space_p_right.png` — PCs 1–4 coloured by P(right)
- **How.** Every pair of PCs 1–4 (6 panels), with points coloured by the model's P(right).
  Uses the wrapper's state-space plotting helper. `b3_state_space.py`.
- **Result.** **PC1 is choice preference:** P(right) runs smoothly from PC1 < 0 (right) to
  PC1 > 0 (left). The PC1–PC2 plane shows two arches, and the PC1–PC4 plane two lobes.

### `fig_B3c_prev_outcome.png` — the four last-input clusters (PC1–PC2)
- **How.** Four panels, one per previous choice × outcome. The trials in that condition are
  coloured by P(right); all others are grey. `b3_state_space.py`.
- **Result.** The previous trial splits the state into **four separate regions**:
  - PC1 follows the previous choice;
  - PC2 follows the previous outcome (rewarded → lower arch, unrewarded → upper arch);
  - within each region P(right) still varies, and after an unrewarded trial it spreads toward
    the other side.
- **Caveat.** The condition columns are blank for each session's first 10 trials, which are
  therefore excluded.

### `fig_B3c_prev_outcome_pc1_pc4.png` — the same clusters in PC1–PC4
- **How.** As above, plotted in PC1 vs PC4.
- **Result.**
  - In this plane the **previous choice** separates the two lobes; rewarded and unrewarded
    states overlap within a lobe.
  - Each lobe is tilted: committed states sit at the outer, lower end.
  - After a rewarded choice there are two bands. The second band is states swinging back
    after having leaned the other way.
  - B7 finds PC4 holds the largest single share of the true reward-probability signal (22%;
    most of that signal lies outside PCs 1–4), so PC4 may partly track which side is
    currently richer. That is a hypothesis.

### `fig_B3c_prev_outcome_t2.png`, `_t3.png` (+ `_pc1_pc4`) — split by deeper history
- **How.** As `fig_B3c_prev_outcome`, but conditioned on the inputs at t−1 and t−2 (16
  panels: row = t−1, column = t−2) and at t−1 to t−3 (64 panels in four 4 × 4 blocks, one
  per t−1; within a block row = t−2, column = t−3).
  - Each panel shows up to 4,000 trials with that history, drawn from all eval trials, over
    a grey 150k subsample. Panel n is the full count. Every pattern has at least 3,500
    trials.
  - Made in PC1–PC2 and PC1–PC4. `b3c_history.py`.
- **Result.**
  - **The last input sets the cluster; earlier inputs set the position inside it.** A
    same-side t−2 input pushes the state to the cluster's outer, committed end. An
    opposite-side t−2 input puts it at the inner end, near P(right) 0.5, where some states
    already favour the other side.
  - **t−3 shifts the state again, but less**, so the sub-clusters overlap more. This is the
    leaky integration of I3/I4 seen directly.
  - After a switch (e.g. t−1 L+, t−2 R+), the state lies between the lobes and P(right) is
    mixed.

### `fig_B3e_trajectories.png` — example trajectories
- **How.** 60 consecutive trials of each task's example session, centred on its first choice
  switch after trial 30, in PC1–PC2. Points are coloured by time; large outlined dots are
  rewarded trials. `b3_state_space.py`.
- **Result.**
  - The state sits in tight clusters at the PC1 extremes and crosses between them in a few
    trials.
  - The noisy Coupled Baiting session crosses PC1 repeatedly (−2.5 to +2.4), mostly in the
    upper-right region.

### `fig_B3_pc56_trial_position.png` — PCs 5–6 coloured by trial position
- **How.** PC5 and PC6 against PC1 and PC2, coloured by the trial's position within its
  session. This figure was drawn because PC5 explains more than 5% of the variance.
  `b3_state_space.py`.
- **Result.** **PC5 is a slow within-session drift.** It rises from the first trials to the
  last (r = 0.51 with trial position, 0.53 with the cumulative reward count), and also carries
  the previous outcome (r = 0.33). The session-conditioning input is constant within a session,
  so it cannot provide this.

### `fig_B3h_pc_overview.png` — PC1–PC2 and PC1–PC4 by P(right) and by the t−1 input
- **How.**
  - B3's 150,000-trial subsample (seed 0) in PC1 vs PC2 (left) and PC1 vs PC4 (right).
  - Top row: coloured by the GRU's P(right). Bottom row: coloured by the four t−1 inputs
    (previous choice × outcome; the study's standard colours), drawn in shuffled order.
  - `b3h_pc_overview.py`.
- **Result.**
  - **PC1–PC2:** the four inputs occupy four distinct regions. PC1 is the chosen side; PC2
    separates rewarded (low) from unrewarded (high) inputs on both sides.
  - **PC1–PC4:** this plane mainly splits the two choice lobes; rewarded and unrewarded inputs
    separate only partly within each.
- **Companion `fig_B3h_pc_overview_value.png`:** the same planes coloured by I2's value
  coordinate v (coolwarm centred on 0, full range). A clean version of I2b's first two panels.
  - **v is graded smoothly along −PC1** across the whole arch, reaching |v| ≈ 3–4 only at the
    outer ends. P(right) is nearly two-toned instead, saturating across most of each lobe.
  - **The two disagree in one region:** PC1–PC4's lower-middle cluster has v ≈ +1 (favours
    right) where P(right) favours left. These are the just-switched states of I11b.

### `fig_B3f_mice_over_embedding.png` — nine mice across the embedding
- **How.** Nine mice chosen nearest to the 10/50/90% grid of the subject embedding's first two
  PCs. Each mouse's trials are shown in PC1–PC2 over the grey cloud of all trials, coloured by
  P(right). `b3_state_space.py`.
- **Result.** Mice occupy **different parts of the same state space**. They share the same
  weights but differ in their embedding and in their own choice / reward histories, so this
  alone does not show the embedding causes it. Some mice never reach the PC1 extremes. This was
  the lead for phase F.

### `fig_B4_update_maps.png` — update maps
- **How.** The model's logit(right) on trial t+1 against trial t, for each trial-t choice ×
  outcome. Grey shows the density of trial pairs; lines show the binned mean, pooled and per
  task. `b4_update_maps.py`.
- **Result.** The updates are **strongly nonlinear**:
  - after a reward the logit moves toward the chosen side, and most strongly when the model
    leaned away (left rewarded maps −3 / 0 / +3 to −3.5 / −1.2 / −0.9);
  - no reward mostly pulls the logit back toward 0 rather than past it; near logit 0 an
    unrewarded choice still moves toward the chosen side (+0.47 right, −0.44 left).

  Phase I suggests most of this nonlinearity is in the readout of a nearly linearly carried
  value (r6). That is not yet tested.

### `fig_B5_outcome_triggered.png` — what follows one outcome
- **How.**
  - Bottom panel: P(choose right) on trials t+1…t+10 after each trial-t choice × outcome, for
    the model and for the animal.
  - Top panel: rewarded − unrewarded Δlogit for the same choice, matched on logit(t).
  - `b5_outcome_triggered.py`.
- **Result.**
  - The model and the animal agree to within 0.001.
  - The top panel starts near +0.85 and falls to about +0.7 at 10 trials, **but this is not the
    effect of one reward.**
    Matching on logit(t) does not match the environment: at the same logit, rewarded trials
    come from richer blocks (P(R) − P(L) ≈ +0.24 vs −0.20).
  - The causal effect of one reward is in C1: 1.0 → 0.02 over 10 trials.
- **Corrected after review.** An earlier draft read the top panel as a long-lasting reward
  effect.

### `fig_B6_tracking.png` — choice vs the true reward probabilities
- **How.** P(choose right), model and animal, binned by P(rew | R) − P(rew | L), one panel per
  task. Thin lines use all trials; bold lines use trials at least 10 trials after the last
  probability change. `b6_tracking.py`.
- **Result.**
  - The model and the animal agree in every bin: mean gap ≤ 0.008 per task at steady state.
  - The curves are jagged because each bin mixes different probability pairs and baiting
    histories. The figure is about the agreement, not the shape.
- **Caveat.** This is a weak test: averaged P(right) curves are easy to match.

### `fig_B7_decoders.png` — what the state linearly encodes
- **How.** One ridge decoder per variable from all 128 units, on 400k trials, with 5-fold
  cross-validation grouped by session.
  - Where a variable lives is read from the decoder's **activation pattern** Σw, not its
    weights. Ridge weights load on low-variance directions.
  - For history lags, a **unique R²** is computed after regressing out all more recent lags.
  - `b7_decoders.py`.
- **Result.**
  - The previous choice and reward decode at R² ≥ 0.998.
  - Rewards 2–5 back are still present beyond what more recent trials explain (unique R²
    0.98 → 0.30), but only 10–20% of their signal is in PCs 1–4.
  - Choice signal sits on PC1 (81%), reward signal on PC2 (74%), and the model's logit on PC1
    (90%).
  - The true reward-probability difference decodes only moderately (R² 0.43); its largest
    single share is on PC4 (22%), and most of it lies outside PCs 1–4.

---

## Phase C — compared with the animals and the RL models (report r2)

### `fig_C1_kernels.png` — history kernels
- **How.**
  - **Logistic regression (Su2022 design):** rewarded choice and unrewarded choice, 10 trials
    back, with a 20-lag check. It is fitted to the animals' choices and to each model's
    P(right) as soft labels, always on the animal's real history. Pooled, per task and per
    mouse; the shading is the animal fit ± 1.96 SEM across mice. `c1_kernels.py`.
  - **Counterfactual (square markers):** 40,000 trials whose reward was flipped in the GRU's
    input, with every later input kept as observed. The marker is the change in log-odds on
    each later trial, signed by choice. `c1_impulse.py`. Figure: `c1_figure.py`.
- **Result.**
  - **The GRU's kernels match the animals'** (rewarded choice 1.807 vs 1.796, unrewarded
    0.876 vs 0.871, largest reward-effect gap 0.009). The RL models deviate by 0.12–0.23.
  - The late reward effect turns slightly negative after about 9 trials and reaches −0.05 to
    −0.09 from about lag 12 out to 20, in both animals and GRU.
  - In the counterfactual, one reward moves the GRU by +1.00 next trial, decaying to +0.017 at
    10. It is positive on average and in the median at every lag, but negative in 41% of flips
    at lag 10.
  - So the logistic weights are not the GRU's causal kernel at long lags.
- **Corrected after review.** The first version used the Hattori2019 design. Its choice term
  is exactly the sum of the other two, so the weights were set by the regularizer. That
  version's "unrewarded-choice dip" is gone.

### `fig_C2a_likelihood.png` — GRU vs RL likelihood
- **How.** Pooled eval likelihood per task for the GRU and the three RL models, plus a per-mouse
  scatter of GRU vs CTT. `c2_compare.py`.
- **Result.**
  - Pooled likelihood: GRU 0.7339, CTT 0.7210, Bari 0.7181, Hattori 0.7159.
  - **The GRU is better for 100% of mice,** with a median per-mouse gain of +0.012 over CTT.
    It is better in every task.

### `fig_C2b_agreement.png` — trial-by-trial agreement
- **How.** 2-D histograms of GRU P(right) against each RL model's P(right). `c2_compare.py`.
- **Result.**
  - The GRU and RL models agree closely: r = 0.974–0.983, mean |ΔP| 0.049–0.058.
  - The vertical stripes for CTT are its discrete output levels.

### `fig_C2c_advantage.png` — where the GRU wins
- **How.** Per-trial advantage = log p_GRU(choice) − log p_CTT(choice), shown four ways:
  - its mean and its sum in PC1–PC2 bins;
  - binned by each model's **pre-choice** predicted P(switch);
  - binned by the GRU's confidence.

  `c2_compare.py`.
- **Result.**
  - The advantage is positive almost everywhere.
  - Per trial, it is largest where the model is uncertain (middle of PC1).
  - It rises with predicted switch probability, from about 0.01 nats when a switch is unlikely
    to 0.03–0.04 at P(switch) 0.4–0.5.
  - Trials with P(switch) > 0.3 (11%) carry about a quarter of the total advantage.
  - It falls with GRU confidence (0.040 → 0.005 nats).
- **Corrected after review.** The first version split by whether the animal *then* switched.
  That conditions on the outcome being scored and gave "two-thirds on switch trials".

### `fig_C2c_advantage_pc_pairs.png` — the same maps in every pair of PCs 1–4
- **How.**
  - The same per-trial advantage over CTT as C2c's first two panels, for all six PC pairs.
    Each pair uses a 48 × 48 grid over the 0.01th–99.99th percentile range plus a 5% margin.
  - Top row: mean per trial (cells with < 200 trials blank).
  - Bottom row: each cell's sum as % of the total advantage, on one scale for all pairs.
  - `c2c_pc_pairs.py`.
- **Result.**
  - **Per trial,** the GRU wins most where the state is uncertain: the centre of PC1, and
    the extremes of PC4 (PC1–PC4, PC2–PC4, PC3–PC4).
  - **In total,** the advantage sits where trials are dense: the post-unrewarded arms
    (PC2 ≈ +1.5) and the inner part of each tilted lobe (PC1–PC4).
  - PC3–PC4 shows no localised total. The total is spread over the cloud, so PC3 adds no
    region of its own.

### `fig_C4a_logit_updates.png` — how each model's policy changes after each input
- **How.** Δlogit = logit(t+1) − logit(t) against logit(t) for the GRU, Hattori, Bari and
  CTT (columns), one row per input on trial t. Stay is solid, switch dashed; binned median
  with an interquartile band. `c4_behaviour.py`.
- **Result.**
  - The broad shapes are shared: little change after a rewarded stay at high confidence,
    and a near reset after a switch.
  - **Hattori barely updates after a loss on the preferred side** (flat at Δ ≈ 0). The GRU,
    Bari and CTT reduce their preference in proportion to confidence.

### `fig_C4b_update_calibration.png` — do the updates match the animals'?
- **How.** Within each model × group, bin trials by the model's own logit(t). Plot the
  model's mean predicted log-odds at t+1 against the animals' empirical log-odds at t+1.
  The diagonal means a match; the legend gives the n-weighted RMS deviation.
- **Result.**
  - **The GRU is on the diagonal everywhere** (RMS ≤ 0.05).
  - **Hattori is S-shaped** (0.08–0.42). Near-neutral losses leave the animals more
    persistent than predicted.
  - **Bari misses after failed switches** (0.35–0.36): the animals go back more.
  - **CTT is vertical after unrewarded switches.** It predicts one value regardless of
    history while the animals vary (0.26–0.28).

### `fig_C4c_history_calibration.png` — calibration by the last three inputs
- **How.** The 64 histories of the inputs at t−1…t−3, laid out as in
  `fig_B3c_prev_outcome_t3`. The first panel is the animals' P(right). The others show each
  model's mean predicted minus observed P(right) in percentage points (orange = predicts
  more right). The last panel is the trial-weighted mean |error|.
- **Result.**
  - Mean error: GRU 0.18 pp, CTT 1.10, Bari 2.13, Hattori 2.24.
  - Histories are written most recent first (t−1, t−2, t−3).
  - **Hattori underestimates persistence after losses.** After R−, R−, L−, the animals
    choose R 69% of the time and Hattori predicts 53%.
  - **Bari and CTT underestimate the return after a failed switch** (±14–17 pp).

### `fig_C4d_disagreement.png` — where the GRU and the RL models disagree
- **How.**
  - Top row: trials binned by p_GRU(stay) − p_RL(stay), with the animals' P(stay) − p_RL(stay)
    against it (on the diagonal the GRU is right, on zero the RL model is right). Then the
    previous input of the 5% of trials with the largest |GRU − CTT|.
  - Bottom row: those trials profiled against all trials.
- **Result.**
  - **The animals follow the GRU in every comparison, in both directions.**
  - The top 5% hold 32% of the advantage over CTT.
  - 55% of them are on the first trial after a switch (13% overall). They are also more
    common after a loss and later in the session.

### `fig_C4e_signatures.png` — returning after a switch, staying through losses
- **How.** Columns: each model against the animals (thick grey, ± SEM), on the same trials.
  - **Top row: P(return to the old side) after trial t−1 switched**, against the length of
    the run on the old side. Solid: the switch was unrewarded; dashed: rewarded.
  - **Bottom row: P(stay) after n consecutive unrewarded choices on one side.** Solid: the
    streak began after a reward there; dashed: right after switching there.
  - `c4e_signatures.py`.
- **Result.** The GRU matches all four curves.
  - **After a failed switch:**
    - The animals return more the longer the old run was (0.39 → 0.53).
    - Hattori overshoots (0.47 → 0.62).
    - Bari undershoots (0.31 → 0.43).
    - CTT is flat (0.41).
  - **Losses right after a switch:**
    - The animals stay more with each loss (0.56 → 0.84).
    - Hattori stays nearly flat (0.47 → 0.58).
    - Bari is too flat (0.64 → 0.74).
    - CTT follows.

### `fig_C3_q_decoders.png` — RL variables inside the GRU state
- **How.** The B7 method applied to Bari and Hattori Q-values (CTT has none) and to each RL
  model's log-odds. Also the reverse: the GRU's log-odds predicted from all RL variables.
  `c3_decoders.py`.
- **Result.**
  - ΔQ decodes at R² 0.95–0.97 and sits on **PC1**.
  - Total value ΣQ decodes at R² 0.77–0.87 and sits on **PC2**.
  - The RL variables explain 94% of the variance of the GRU's log-odds. The remaining 6% is
    where its advantage should live (a hypothesis; E2b/I7 follow it up).

---

## Phase D — dynamics (report r3)

### `fig_D1_flow_fields.png` — flow fields
- **How.**
  - Uses the free-context step function (wrapper `gru_step`), which reproduces the stored
    forward pass (median |Δlogit| 0.0009).
  - For each of the four inputs, one step's state change is computed from 20,000 real states,
    at the all-sessions mean context.
  - The changes are averaged in PC1–PC2 bins and drawn at full length.
  - Overlays show the fixed points of each input held constant. These are found from 2,000 real
    starting states by Adam and by forward iteration, each refined by damped Newton in float64.
  - `d_dynamics.py`.
- **Result.** **One trial moves the state a long way.** Each input carries the state most of the
  way into its own cluster region, and each input's attractor sits inside that region.
- **Reading note.** The arrows are one-step displacements, not trajectories. Drawn at full
  length, they converge where a single step lands (roughly steps 1–5), not on the attractor.
  After that, repeating the input moves the state slowly, over tens to hundreds of trials,
  until it reaches the fixed point. This is most visible for the unrewarded inputs, whose
  stars sit beyond where the arrows meet. D1b shows the full convergence.

### `fig_D1b_convergence.png` — where repeated inputs actually take real states
- **How.**
  - 300 real states are iterated for 600 steps under each constant input (all-sessions mean
    context, float64).
  - Paths are drawn in PC1–PC2, with the positions after 1, 3, 10, 30, 100 and 600 steps,
    and D's fixed points.
  - `d1b_convergence.py`.
- **Result.**
  - **Every path ends on D's stable fixed point.** The endpoints match it in PCs 1–6.
  - **One step covers most of the distance** (median |Δh| about 2.5, falling to about 0.1 by
    step 10).
  - **The rest is a slow drift** over 30–600 steps, longest for unrewarded inputs (PC2
    1.5 → 1.0).
  - **So D1's full-length one-step arrows** converge where one step lands, not on the
    attractor.
  - **For "L rewarded",** every real start reaches the shallow attractor (logit −4.04). The
    deep one (−4.98) is a separate basin no real start enters.

### `fig_D1c_flow_two_scales.png` — the flow field at two scales (replaces D1 for reading)
- **How.**
  - Top row, the fast jump: one-step Δh from 20,000 real states, binned in PC1–PC2. Arrows
    show direction only, and the background colour is the median step size (log scale).
  - Bottom row, the slow drift, zoomed: states that already had the input repeated ≥ 5 times
    (300 real starts, steps 5–600), with the next step's direction and size, the median path
    and D's fixed points.
  - `d1c_flow_two_scales.py`.
- **Result.**
  - **Top row:** the step-size map has its minimum at each stable fixed point.
  - **Bottom row:** the direction field and the median path end on the star. The slow
    drift is visible, especially for unrewarded inputs.

### `fig_D1d_flow_scaled.png` — D1 with arrow length scaled to the step size
- **How.** D1's one-step Δh (binned in PC1–PC2), with arrows shrunk so none leaves its bin.
  Two scalings: top row, length ∝ step size (linear); bottom row, length ∝ log10 step size
  (0.001–4). `d1d_flow_scaled.py`.
- **Result.**
  - **Linear scaling** removes D1's false convergence point, but every arrow in the input's
    own region shrinks to a dot. The attractor isn't singled out, and the slow drift is
    invisible.
  - **Log scaling** keeps the arrows readable everywhere. They flow along the arch onto the
    star, so this is the best single-panel version.

### `fig_D1f_fixed_points_flow.png` — the four attractors together, then each input's flow
- **How.**
  - All-sessions mean context, in PC1–PC2, with B3h's input colours.
  - **Only behaviourally relevant fixed points are drawn.** These are the ones real states
    reach: 300 real starts iterated 600 steps under each constant input (as D1b), with end
    states assigned to the nearest fixed point.
  - Panel 1: the kept fixed points of all four inputs.
  - Panels 2–5: one input each, with D1's full-length arrows (mean one-step Δh from 20,000 real
    states per PC1–PC2 bin) and that input's kept fixed point.
  - `d1f_fixed_points_flow.py`; reach counts in `d1f_fixed_points_flow.json`.
  - `fig_D1f_flow_2x2.png`: panels 2–5 alone in a 2 × 2 grid (rows = side R / L, columns =
    rewarded / unrewarded).
- **Result.**
  - **One stable attractor per input,** at the four corners of the arch: R rewarded lower left,
    R unrewarded upper left, L rewarded lower right, L unrewarded upper right.
  - **This is the same layout as the t−1 inputs in B3h.**
  - **Dropped:** "L rewarded"'s deep attractor (logit −4.98) and its unstable neighbour, each
    reached by 0 of 300 starts. Every start ends on the kept attractor, for every input.
  - **The full-length arrows show the one-step jump into each input's region.** As D1's
    reading note says, they converge where one step lands, not on the star; D1d / D1c show the
    slow drift onto it.

### `fig_D1e_flow_own_context.png` — flow with each state's own context
- **How.**
  - As D1d (bottom, log-length arrows), but each of 20,000 real states is stepped with its own
    session's context.
  - Dots are every mouse's attractor per input: forward iteration for 800 steps from 8 of the
    mouse's own states at its mean context.
  - `d1e_flow_own_context.py` (GPU).
- **Result.**
  - **The flow matches the mean-context version.**
  - **The 614 attractors form a short line per input,** not a point: for example R+ PC1
    −3.4 to −2.9 and logit 3.6–5.3 (10th–90th percentile). The embedding moves each attractor
    along one direction, its depth.
  - **Convergence check:** unrewarded inputs match F2's logits to 0.01. For rewarded inputs
    the median difference is 0.05–0.07, because 800 steps don't fully converge the slowest
    modes.

### `fig_D1f_flow_prototypes.png` — flow fields of prototype mice
- **How.** F9's mice at the two ends of the three named axes. Each is shown with its own states
  and contexts (arrows only where that mouse has states), its own attractor (★) and the
  mean-context attractor (☆).
- **Result.**
  - **The biggest difference is where mice live, not the flow.** The high-determinism mouse
    occupies only the lobes' outer ends; the low-determinism mouse sits in the middle and
    never reaches its rewarded attractors.
  - **Attractors shift modestly with the axes.** For example, the high-determinism and
    high-slow-learning mice have deeper attractors.

### `fig_D2_fixed_points.png` — fixed points and timescales across contexts
- **How.** Fixed points for five contexts: the all-sessions mean, and the mean contexts of F1's
  low- and high-gain and low- and high-bias mice. Their positions are shown in PC1–PC2, and
  the three slowest time constants (τ = −1/ln|λ|) at each stable point. `d_dynamics.py`.
- **Result.**
  - Each constant input has **one stable attractor on the chosen side**. "Left rewarded" is
    bistable in 3 of 5 contexts.
  - Repeating a choice without reward still ends on that side.
  - The slowest modes are very slow (τ 120–650 trials, ≈ 1900 at the second "left rewarded"
    point) but **barely touch the readout**
    (overlap 0.0–0.31), so they do not explain the 2–3-trial behavioural timescale.
  - **Context changes attractor depth, not position.** The "gain" contexts are deeper on
    both sides (high vs low: +4.6 / −5.0 vs +3.8 / −4.0); bias deepens one side and shallows
    the other (high vs low bias: +4.9 / −3.9 vs +3.5 / −5.5).
  - **Revised by F2 (all mice):** depth tracks persistence (ρ +0.50), not reward gain
    (+0.28). D2's "gain" contexts lay along F3's combined gain + persistence direction.
- **Caveat.** A first search stalled and reported no fixed points; it was fixed before use.
  58–84% of candidates converge.

---

## Phase I — is there a shared integration axis? (report r6)

### `fig_I1_cluster_axes.png` — per-cluster encoding axes
- **How.**
  - For each cluster and value variable v, the **encoding axis** is cov(h, v) / var(v): the
    direction the state moves as v changes within the cluster.
  - For v = the model's logit this is Σ_c r, not r itself; testing r would be circular.
  - Two external values are also used: Hattori ΔQ and the true reward-probability difference.
  - Context and trial position are partialled out. Computed from per-cluster joint covariances
    over all trials.
  - `i1_cluster_axes.py`.
- **Result.**
  - **There are two axes, one per previous choice.** Same previous choice gives cosine 0.92
    (logit) or 0.93–0.94 (ΔQ); opposite choices give 0.35–0.52 or 0.49–0.71. The two axes form
    a "V": a shared PC1/PC4 component and opposite-signed PC2.
  - Within each cluster the model's value axis and the RL value axis coincide (0.94–0.96).
  - The true probabilities' axis does not (0.12–0.26).

### `fig_I2_tdr.png` — targeted dimensionality reduction
- **How.**
  - The state is regressed on previous choice, previous reward, choice × reward, value (the
    logit), and value × choice, with context and trial position as nuisances. This uses 1M
    trials.
  - The axes are orthogonalized with value first, and states are projected onto them.
  - `i2_tdr.py`.
- **Result.**
  - The two lobes separate along a choice axis, and **each spans the full value range**.
  - Reward separates rewarded from unrewarded states within each lobe.
  - Five task axes hold 74% of the variance (the top 5 PCs hold 82%).
- **Caveat.** Using the logit as a regressor makes the other axes orthogonal to the readout by
  construction.

### `fig_I2b_value_axis_in_pcs.png` — where the value coordinate lives in PC space
- **How.**
  - 150,000 states in PC1–PC2, PC1–PC4 and PC2–PC4, coloured by v.
  - Arrows from the cloud centre: each unit axis projected onto the plane (length 3 means
    fully in the plane). The axes are the shared value axis, the two lobe value axes, the
    readout direction, and the choice and reward axes.
  - Right: each axis's loadings on PCs 1–8.
  - `i2b_value_axis_in_pcs.py`.
- **Result.**
  - **v's axis lies 83% in PCs 1–4,** mostly along −PC1 (loading −0.84) with a PC4 component
    (−0.35).
  - **The two lobe axes share that −PC1 direction but tilt apart along PC2** (R lobe −0.38,
    L lobe +0.45): the "V" of I1.
  - **The reward axis is almost pure PC2** (−0.90). The choice axis mixes −PC1 and +PC4.
  - **The readout direction is only 24% in PCs 1–4.** v is the main within-cluster carrier of
    the choice signal, not the readout itself.

### `fig_I3_tdr_dynamics.png` — the value coordinate across transitions
- **How.**
  - States at t and t+1 are projected onto the TDR axes (1M transitions).
  - For each input and source lobe (repeat vs switch side), the value coordinate at t+1 is
    regressed on its value at t.
  - Dashed lines mark each input's attractor, found with the wrapper's `find_fixed_points`.
  - `i3_tdr_dynamics.py`.
- **Result.**
  - **The value coordinate is carried through every jump,** including switches to the other
    lobe: slope 0.66–0.80, R² 0.93–0.98.
  - Each map crosses the diagonal near the input's attractor (closely for rewarded stays;
    the maps are curved), v(t+1) ≈ v* + λ(v(t) − v*) with
    λ ≈ 0.75. That is the form of a delta-rule update.
  - The implied update fraction (0.20–0.34) is in the range of the fitted RL learning rates
    (Bari median 0.19, CTT 0.29).

### `fig_I3b_stay_switch.png` — the same transitions, grouped by stay vs switch
- **How.**
  - I3's 1M transitions; one panel for stays and one for switches (choice t vs choice t−1).
  - Trial t's input is shown by colour, with a fitted line per input.
  - The thick diagonal is v unchanged; the dashed lines are each input's attractor (stay panel).
  - `i3b_stay_switch.py`.
- **Result.**
  - **On stays, the outcome barely changes the update.** Rewarded and unrewarded lines on
    the same side nearly coincide (slopes 0.78–0.80); the side chosen does most of the work.
  - **Switches** all cross near v ≈ 0, with shallower slopes (0.66–0.74). That is the pull
    toward neutral after a side change.

### `fig_I11_value_across_lobes.png` — v carried across trials and lobes, in the (choice axis, v) plane
- **How.**
  - I2's TDR choice axis c (which lobe) against the value axis v; states coloured by the t−1
    input (B3h's colours).
  - Top row:
    - an example Uncoupled Baiting session (trials 60–140) in the plane, switches bold;
    - the same trials over time (v, c, the scaled logit, choices);
    - switch-triggered averages (50,000 switches after runs of ≥ 5 same-side choices; signed
      toward the old side, in SDs).
  - Bottom row: per input, the mean one-step displacement in the plane (1M transitions,
    full-length arrows) and I3's attractor.
  - `i11_value_across_lobes.py`.
- **Result.**
  - **Switches are horizontal jumps:** c changes lobe while v keeps its height.
  - **Inside the input's own lobe** the flow runs along v to the attractor.
  - **At a switch, c and the logit flip at once while v crosses over about three trials**
    (0.92 → 0.59 → 0.19 → 0.04).

### `fig_I11d_carry_summary.png` — v's carry-over in three panels
- **How.** I11's example session in the (c, v) plane (switches bold), then I3b's stay and
  switch maps of v after trial t against v before (same 1M transitions, per-input fits, the
  stay panel's dashed attractors), in B3h's input colours. `i11d_carry_summary.py`.
- **Result.** As I11 and I3b: switches are horizontal jumps that keep v, and v is carried
  with slope 0.78–0.80 on stays and 0.66–0.74 on switches.

### `fig_I11b_cv_plane.png` — the (c, v) plane, by t−1 input and by P(right)
- **How.**
  - B3h's 150,000-trial subsample in I11's (choice axis c, value axis v) plane.
  - Coloured by the t−1 input (left; session-first states, which have no input, are left
    out) and by the GRU's P(right) (right).
  - Not coloured by v: that would only repeat the y-axis. Made by `i11_value_across_lobes.py`.
- **Result.**
  - **Each lobe has a main band** (v favouring the lobe's side) **and a smaller just-switched
    band** (v still favouring the old side).
  - **P(right) follows the lobe first and v second:** just-switched states take their lobe's
    colour.
  - **The small cluster near (0.2, 0.15)** in the right panel is the session-first states.

### `fig_I11c_carry_in_pcs.png` — carry-over in PC space: post-switch states coloured by where they came from
- **How.**
  - The first state after each choice switch (548,793 switches; 60,000 drawn), in PC1–PC2
    (top) and PC1–PC4 (bottom), over the grey cloud.
  - Coloured by (1) its own v, (2) the v of the state just before the switch, (3) that
    pre-switch v shuffled among switches of the same direction (a control that keeps the
    side-level colouring). One colour scale throughout.
  - Made by `i11_value_across_lobes.py`.
- **Result.**
  - **The pre-switch v reproduces the gradient of the post-switch states' own v** (Spearman
    0.975). It is more saturated, because v shrinks about 30% across a switch (I3b).
  - **The shuffle scrambles it** (0.20).
  - **So a post-switch state's position is set by the v it carried in.** The post-switch
    states fill the middle of the arch because v lies mostly along −PC1 and is kept.

### `fig_I4_distilled_model.png` — a one-variable model distilled from the GRU
- **How.**
  - v = I2's value coordinate of every trial's state.
  - Per transition group (input × stay/switch/first), fit v(t+1) = v* + λ(v(t) − v*). Fit
    the readout logit ≈ a·v + b[last input] to the GRU's logit.
  - Simulate v from a fixed start along each session, driven only by the animal's choices
    and rewards, with no GRU. Score P(right) on the animals' choices.
  - Two versions: pooled parameters, and per-mouse parameters (pooled values for groups
    with < 50 transitions).
  - Panels: likelihoods; the eight fitted update lines (solid = same side, dashed =
    switch); an example session. `i4_distilled_model.py`.
- **Result.**
  - **The simulated v follows the GRU's v** (corr 0.985 per mouse, 0.976 pooled).
  - **It scores 0.718** (pooled: 0.707), against 0.734 for the GRU and 0.716–0.721 for the
    RL fits. It beats CTT in only 21% of mice.
  - **Most of the gap is the readout** (corrected in I7). A per-mouse readout on the GRU's
    own v scores 0.724, so ~0.010 of the gap is the readout and ~0.006 the dynamics.

### `fig_I5_mouse_dynamics.png` — which dynamics parameter carries each behavioural axis
- **How.**
  - Per mouse, summarise I4's fits:
    - λ_stay and λ_switch;
    - reward push (rewarded − unrewarded stay target);
    - choice push (unrewarded-stay target toward the chosen side);
    - target asymmetry (mean stay target);
    - switch target (signed toward the new side);
    - readout slope and offset.
  - Correlate (Spearman) with F3's GRU kernel features, and decode each from the embedding
    with F3's method (rank R², 5-fold CV, shuffle null). `i5_mouse_dynamics.py`.
- **Result.**
  - **Bias = target asymmetry** (ρ +0.91) and readout offset (+0.78).
  - **Reward gain and persistence ≈ the switch target** (ρ +0.93 / +0.92) and faster
    forgetting (λ_switch −0.79 / −0.74).
  - **The choice push is a separate parameter.**
  - **All of them are decodable from the embedding** (rank R² 0.41–0.74; null ≈ 0).

### `fig_I6_rl_transitions.png` — the same transitions in Hattori and Bari
- **How.**
  - I4's group fits, applied to RL ΔQ = Q_R − Q_L (top row) and to each model's policy
    logit (bottom row), pooled over the 614 mice.
  - Curves are binned means of x(t+1) against x(t) per group.
  - Right column: the per-mouse median λ, and targets normalised to the rewarded-stay
    target (policy logit). `i6_rl_transitions.py`.
- **Result.**
  - **In policy space the three models look alike:** stays retain, switches nearly reset,
    and the normalised targets are similar.
  - **In the carried variable they differ.** The GRU's v is affine and Markov in every
    group (R² ≥ 0.93, switches included), with nearly input-independent λ.
  - **RL ΔQ is not Markov on switches** (R² 0.12–0.86). Hattori uses two retentions (0.49
    rewarded vs 0.90 unrewarded).
  - **The unrewarded-stay push sits in v for the GRU,** partly in ΔQ for Hattori, and only
    in the choice kernel for Bari.

### `fig_I7_second_variable.png` — a second carried coordinate
- **How.**
  - I4 in 2-d. For a coordinate u orthogonal to v, fit [v, u](t+1) = A_g [v, u](t) + c_g
    per group, and the readout logit ≈ a_v·v + a_u·u + b[last input]. Simulate from the
    animals' inputs.
  - Candidates:
    - the readout residual (the part of the readout orthogonal to v; an exact bound);
    - the TDR choice, reward, value×choice and choice×reward axes;
    - PC5;
    - choice + reward (3-d).
  - Panels:
    - likelihoods: teacher-forced per mouse (grey), simulated per mouse and pooled;
    - u(t+1) vs u(t) per group for the residual and for the best interpretable axis;
    - an example session. `i7_second_variable.py`.
- **Result.**
  - **The choice×reward axis is the second variable.** Simulated, it scores 0.7234 and
    closes 34% of the gap, as much as the exact residual. The other axes close ≤ 10%.
  - **It encodes unrewarded choices 2–10 trials back with negative weight** (lose → avoid).
    v mostly encodes choice persistence.
  - **Its update is curved, not affine** (update R² 0.92), which is where most of the
    remaining gap is.

### `fig_I8_v_tests.png` — causal tests of v
- **How.** `i8_v_tests.py`, JAX on CPU with the wrapper's `GruStep`.
  1. **Closure:** held-out R² of v(t+1) per group, from v alone, from v + 20 PCs, and from
     the full state.
  2. **Perturbation:** ±0.5 along v's axis at trial t, then the GRU run forward on the real
     inputs. Δv/δ is compared with I4's Π λ, and the share of the perturbation leaving v's
     axis is tracked.
  3. **Flip kernels:** flip trial t's reward, or its choice with the outcome kept, for
     20,000 probes. Effects on the GRU's v and logit, and on Hattori's and Bari's β·ΔQ and
     choice-kernel term (exact numpy replay), all in log-odds units.
  - Right column: the unrewarded/rewarded choice-effect ratio, and the reward's share of a
    rewarded choice's effect.
- **Result.**
  - **v is approximately closed** (R² 0.93–0.98 from v alone; the rest of the state
    explains most of the remainder).
  - **It is not a single mode.** A perturbation decays faster than Π λ at first, then
    leaves a slow tail, and leaks off v's axis (49% after one step, 87% after ten).
  - **In v, past choices matter far more than their rewards,** with a fixed
    unrewarded/rewarded ratio of about 0.65 from k = 2.
  - **At the policy level the GRU's choice trace is separable** (ratio 0.39–0.48 at all
    lags). Hattori's isn't (0.39 → 1.31).
  - **The GRU's reward memory fades faster than its choice memory.** In the RL models
    both live in Q and persist equally.

### `fig_I9_v_kernel.png` — what v integrates, and whether a 3-parameter description suffices
- **How.**
  - v (and, for contrast, Hattori and Bari ΔQ) regressed on the last 10 choices (±1) and
    rewarded choices.
  - Three descriptions compared by held-out R² (80/20 split by session):
    - free (21 parameters);
    - one choice trace scaled by outcome (4);
    - choice trace plus a separate reward trace (5).
  - Panels: v's kernels with the fitted descriptions; normalised kernels for all three
    targets; the unrewarded/rewarded ratio per lag; R².
  - `i9_v_kernel.py`.
- **Result.**
  - **The separable description explains 0.945** of v's variance, against 0.953 free.
  - **The reward part decays faster** (μ 0.62 vs λ 0.82), so the ratio rises with lag.
  - **ΔQ fits equally well;** the kernel shapes separate the models: Hattori's ratio falls,
    Bari's ΔQ is reward-only, v's rises.

### `fig_I9b_v_kernel_per_mouse.png` — v's history kernels per mouse
- **How.**
  - I9's free, separable and two-rate fits per mouse (610 mice) for v and each mouse's own
    Hattori and Bari ΔQ, with odd sessions held out and split-half reliability.
  - Panels:
    - median and IQR of the normalised kernels against the pooled fit;
    - early vs late outcome ratio per mouse;
    - λ vs μ;
    - correlations of v's parameters with the named axes and the animals' features;
    - reliabilities.
  - `i9b_v_kernel_per_mouse.py`.
- **Result.**
  - **The rising ratio and μ < λ hold in 99–100% of mice for v,** and fail in 98–99% for
    Hattori.
  - **v's choice timescale λ** (0.79, reliability 0.95) is the slow-learning axis (+0.70).
  - **The reward boost κ** tracks win-stay (+0.56).

### `fig_I9c_v_kernel_variants.png` — kernel variants: lag 1 absorbed; two outcome traces
- **How.**
  - I9's fits with and without one offset per last-trial cluster (with offsets, the kernels
    cover lags 2–10 and R² is within cluster).
  - A two-outcome family (separate amplitude and decay for rewarded and unrewarded
    choices) alongside "choice + reward". Pooled and per mouse.
  - `i9c_v_kernel_variants.py`.
- **Result.**
  - **v's older kernel is unchanged by the offsets.** History beyond lag 1 explains 86% of
    v's within-cluster variance, against 67–74% for ΔQ.
  - **Unrewarded choices decay more slowly than rewarded ones in 100% of mice**
    (0.83 vs 0.68); Hattori shows the opposite. I9d refines this: the 0.83 averages a fast trace and a slow residue.
  - **A_rew is the determinism axis** (+0.83); **λ_rew is the slow-learning axis** (+0.65).

### `fig_I9d_v_kernel_long.png` — K = 20 lags and a three-trace model
- **How.**
  - I9c's offsets fits with 20 lags (K = 10 refitted on the same rows) and a six-parameter
    "three traces" model (rewarded and unrewarded traces plus a shared choice trace).
  - Top: pooled free kernels with each fit for v, Hattori ΔQ and Bari ΔQ; fitted − free for v.
    Bottom: gap to the free fit, per-mouse BIC winner, per-mouse two-outcome decays at K = 10
    and 20, per-mouse R².
  - `i9d_v_kernel_long.py`.
- **Result.**
  - **Three traces ≈ free** for v (gap 0.0006; lowest BIC in 99.5% of mice).
  - **v = fast choice trace (λ ≈ 0.67) + reward boost (λ ≈ 0.71) + small unrewarded residue
    (≈ 0.045) that does not decay within 20 trials.** I9c's λ_unr 0.83 averaged the fast and
    slow parts.
  - The last lag still rises, so history older than 20 trials still contributes.

### `fig_I9e_trace_models.png` — trace update rules fitted to the mice
- **How.**
  - Choice / reward / unrewarded traces (T ← λT + input) → logistic choice, with and without
    last-trial weights; Q-learning (Hattori form) and Q + choice kernel refitted the same way.
  - Per mouse, fitted to the animals' choices and scored on held-out sessions (2-fold). The
    GRU and the wrapper's RL fits are shown as references.
  - Panels: held-out likelihood; per-mouse Δ vs Q; paired three-trace comparisons (clipped
    to the 0.5–99.5th percentile); implied kernels; per-mouse timescales and weights;
    parameters × named axes; C4e signature errors.
  - `i9e_trace_models.py` (GPU, about 2.5 h).
- **Result.**
  - **Three traces is the best refitted model** (0.7231 against Q + kernel + last 0.7219;
    GRU 0.7339). It closes 45% of the Q → GRU gap and beats Q + kernel in 68% of mice.
  - **Behaviour has v's slow unrewarded residue** (λ_u 0.98 > λ_w 0.64 in 99% of mice; β_u > 0
    in 85%), but a faster choice trace (λ_c 0.39).
  - **The C4e switch signatures are not fixed;** the GRU is about 10× closer.

### `fig_I10_gates.png` — how the GRU holds v
- **How.**
  - v's exact one-step retention eᵀJe at 20,000 real transitions per input group, split into
    leak (1 − z), gate change and candidate (recurrent re-write) terms.
  - Panels: the split per group; retention against v; v's unit loadings; the units' update
    gates.
  - `i10_gates.py`.
- **Result.**
  - **The gate is mostly open along v** (z̄ about 0.67), so the leak keeps only about 0.33.
  - **The recurrent candidate term adds about 0.35,** giving a total of about 0.7, which
    matches I8. v is re-written each trial, not held by a closed gate.
  - **Stays and switches share the same local retention.**
  - **v is distributed** (participation ratio 32 of 128 units).

---

## Phase E — subspaces and what they do (report r4)

### `fig_E1_subspaces.png` — geometry of the 128-d state
- **How.**
  - Compares unit vectors: the readout direction; the mean state change after each input
    (choice, reward and choice × reward directions); decoder activation patterns; and PCs 1–6.
  - Left panel: the |cosine| matrix.
  - Right panel: each axis's cos² with the readout (the choice-driving part) and its share of
    state variance.
  - `e1_subspaces.py`.
- **Result.**
  - The readout direction carries only **9.9%** of state variance.
  - Each input lands on the axis where its variable is stored: choice input → previous choice /
    ΔQ / PC1 (cos 0.87–0.92); reward input → previous reward / ΣQ / PC2 (0.85–0.99).
  - Reward history, ΣQ and trial position have no component along the readout.
- **Corrected after review.** This alignment is partly by construction, so it shows where
  inputs land, not that the dynamics are linear.

### `fig_E2_ablations.png` — global-mean ablation (superseded)
- **How.** At every step, the state's component in a subspace is replaced by the **global**
  mean state, either read out and carried (full) or carried only (memory-only). Scored by
  likelihood and reward kernel, with 10 random 1-d controls. `e2_ablations.py`.
- **Result.**
  - Not interpretable direction by direction. Random typical-variance directions cost as much
    as meaningful ones (down to −0.23), and removing 127 dimensions costs less than removing one.
    Clamping one coordinate puts the state off-manifold.
  - **One robust result:** keeping only the readout coordinate (a one-dimensional carried
    state) leaves the model at **0.713**. The full GRU reaches 0.734 and CTT 0.721.

### `fig_E2b_conditional_ablations.png` — conditional-mean ablation
- **How.**
  - As E2, but the component is replaced by the mean of the state's **own last-input cluster**,
    so the state stays on its cluster. Memory-only mode.
  - Controls: 10 random 1-d and 10 random 2-d subspaces.
  - Scored by likelihood and by the reward kernel (shaded bands: random ranges).
  - `e2b_conditional_ablations.py`.
- **Result.**
  - Random directions now cost only −0.006 to −0.021.
  - The **reward axis** carries the one-trial reward effect: removing it costs −0.081 and halves
    the effect one back (0.93 → 0.40).
  - The **readout coordinate** (equivalently, the two lobe value axes) carries reward memory
    **beyond two trials**: lags 1–2 stay intact, while lag 3 falls to 0.09 / 0.04. Every random
    control keeps at least 0.17.
  - The choice axis and PC5 are dispensable within a cluster.
- **Caveat.** "Specific" means outside the range of the random controls, not a formal test.

---

## Phase F — what the subject embedding does (report r5)

### `fig_F3_embedding_features.png` — what the embedding predicts
- **How.**
  - Each mouse's features are predicted from its 4-d embedding: GRU and animal kernel features
    (bias, reward gain, persistence), and all RL parameters.
  - 5-fold cross-validated linear regression across mice, scored as **rank R²**, against a
    200-permutation shuffle null.
  - Task-mix control: the mouse's per-task trial fractions alone, and the embedding's
    prediction after regressing them out.
  - Middle panel: cosines between the embedding directions in the embedding-covariance metric.
  - Right panel: GRU vs animal per-mouse features.
  - `f3_embedding_features.py`.
- **Result.**
  - Study 01 r10 **replicates** on this model: RL bias 0.73, learning rate 0.65–0.67.
  - The GRU's own bias, gain and persistence are predicted at 0.78–0.80. They track the
    animals' per mouse (Spearman 0.95, 0.80, 0.96).
  - Not a task-mix effect: task mix alone gives ≤ 0.26, while the embedding still gives
    0.60–0.78 beyond it.
  - **Two separate axes:** bias is orthogonal to gain (−0.05; feature Spearman −0.03). Gain,
    persistence and the RL learning rates form one axis (0.92–0.94).

### `fig_F2_mouse_dynamics.png` — attractors and timescales of every mouse's context
- **How.**
  - For each mouse (mean context of its eval sessions) and each constant input: stable fixed
    points reached from 96 of the mouse's own states, by forward iteration and damped Newton
    in float64 on GPU.
  - Measured at each attractor: its logit, the slowest |eigenvalue| and its readout overlap,
    and v's one-step retention e_vᵀ J e_v.
  - The features are correlated with F3's kernel features and I5's fitted dynamics, and
    decoded from the embedding.
  - `f2_mouse_dynamics.py`, which caches the search in `_cache/f2_fixed_points_C-s0.parquet`.
- **Result.**
  - **One attractor per input on the chosen side in 612 of 614 mice** (2 have no converged
    "left rewarded" search). "L rewarded" is bistable in at least 78 mice; the search misses
    D's second mean-context point, so this is a lower bound.
  - **Bias moves the attractors** (ρ +0.57 with their mean logit).
  - **Reward gain is a speed, not a depth.** It doesn't track attractor depth (−0.05 for
    the rewarded-minus-unrewarded depth), but does track v's retention at the attractor
    (−0.46), which itself agrees with I5's λ (+0.41).
  - **Persistence tracks depth** (+0.50).
  - **The slow modes (τ 270–890 trials, readout overlap ≈ 0.11) are generic.** The
    embedding barely sets them (rank R² 0.14).

### `fig_F4b_context_drift.png` — how the context moves with training
- **How.**
  - Top row:
    - the share of context variance within vs between mice;
    - each mouse's late − early mean context on the embedding PCs (in between-mouse SDs);
    - early → late arrows for 150 mice in PC3–PC4;
    - median changes: animal, GRU total, and the part caused by the context.
  - Bottom row: per mouse, the context-caused change (late sessions rerun with the mean early
    context, soft-label Su2022 refit) against the GRU's total change, and its Spearman with
    the animals' change.
  - `f4b_context_drift.py` (GPU), using F4's median split.
- **Result.**
  - **Contexts move in a shared direction:** deeper attractors (PC2) and a lower learning
    rate (PC3). The within-mouse variance share is only 3%, but F4c shows the distance is
    up to about 1 between-mouse SD, and the path is front-loaded.
  - **This shared drift explains about a third of the GRU's average gain decrease** (opposite
    to the animals' increase) and nearly all of its average bias change.
  - **Per mouse** it is aligned with the GRU's changes but holds ≤ 6% of their variance; the
    input history drives the rest.

### `fig_F4c_context_trajectories.png` — every mouse's context across all its sessions
- **How.**
  - The context for every training session of every mouse, recomputed by the network
    (it matches the cached eval contexts to 3e-5).
  - Top row:
    - 12 example mice (spread over the embedding, ≥ 40 sessions) on PC1–PC2 and PC3–PC4
      over the cloud of all mice;
    - zoomed, from each mouse's own start, coloured by training phase;
    - the median trajectory per PC along training phase.
  - Bottom row: the example trajectories mapped to predicted GRU bias, reward gain and
    persistence (cross-mouse linear maps, CV R² 0.84–0.89), against the between-mouse range;
    and trajectory size against the between-mouse SD.
  - `f4c_context_trajectories.py`.
- **Result.**
  - **Median first → last change:** +0.80 between-mouse SDs along PC2 (99.5% of mice), and
    about 0.3–0.4 along PC1, PC3 and PC4.
  - **The movement is front-loaded:** most of it happens in the first 30–50% of training.
  - **PC4 swings up and partly back.** This is why F4b's half-split sign differs from the
    first → last sign.
  - **The behavioural translation** puts most of the change early, with a small median.

### `fig_F7a_feature_library.png` — behavioural features, reliability and embedding decodability
- **How.**
  - 13 features per mouse (animal and GRU), with odd/even-session reliability (Spearman–Brown).
  - Rank R² from the embedding, alone and after removing determinism.
  - Right: correlations between the animal features. `f7_feature_library.py`.
- **Result.**
  - **Most features are close to their reliability ceiling.**
  - **The animals' stickiness ratio and matching sensitivity are reliable but weakly carried.**
  - **Gain, stickiness and persistence collapse onto determinism.**

### `fig_F7b_named_axes.png` — varimax-rotated embedding axes
- **How.**
  - Loadings of the GRU features on the whitened embedding, varimax-rotated.
  - Every mouse is placed on the axes.
- **Result.** The axes are determinism / stickiness, side bias, slow learning / long choice
  memory, and a dynamics-only axis.
- **Caveat (added 2026-10-27).** The axes are orthogonal by construction. The behaviours they
  are named after are not: determinism and the RL learning rate are 52° apart (r5 item 13).

### `fig_F8_embedding_traversal.png` — synthetic mice along each axis
- **How.**
  - The mean mouse ± 0–2 SD along each axis; the GRU is run on 600 real sessions.
  - Features are measured from its predictions. Causal slopes are compared with the loadings.
  - `f8_embedding_traversal.py` (GPU).
- **Result.**
  - **Axes 1–3 do what they were named for** (agreement 0.94 / 0.92 / 0.78).
  - **Axis 4 is a subtle reward-memory / stickiness direction** (0.71).

### `fig_F9_strategy_prototypes.png` — mice at the ends of each axis
- **How.**
  - For each axis, the mice nearest its 5th and 95th percentile, near the median on the others.
  - Animal vs GRU kernels and feature percentiles.
  - `f9_strategy_cards.py`; per-mouse cards in `_cache/explore/cards/` and the table
    `f9_mouse_strategy_profiles.csv`.

### `fig_F10a_embedding_stickiness_maps.png`, `fig_F10b_embedding_memory_matching_maps.png` — each behavioural aspect mapped over the embedding
- **How.**
  - F7's 13 features per mouse, in the embedding's standardised PC1–PC2 and PC3–PC4.
  - Background: a Gaussian-kernel (0.35 SD) local mean of the feature's percentile rank,
    masked where fewer than 4 mice contribute; dots are mice. Animal and GRU side by side.
  - F10a: determinism, win-stay, lose-stay, stay through losses, stickiness ratio, switch
    rate. F10b: choice and reward memory span, poorer-side rate, return after a failed
    switch, matching sensitivity and bias, side bias.
  - A 2-d map averages over the two PCs not shown. `f10_embedding_behaviour_maps.py`.
- **Result.**
  - **Stickiness is a diagonal gradient in PC1–PC2** (sticky at −PC1/+PC2) **plus PC3**
    (sticky at −PC3).
  - **Choice memory span and the poorer-side rate are vertical bands in PC3** and flat in
    PC1–PC2.
  - **Bias runs along the other diagonal of PC1–PC2 and down PC4**, orthogonal to
    stickiness in PC1–PC2.
  - The animals' maps have the GRU's layout. They are weaker for stickiness ratio, reward
    memory span and matching sensitivity.

### `fig_F10c_embedding_feature_directions.png` — where each feature points
- **How.** The correlation of each feature's rank with each standardised embedding PC, drawn
  as arrows in PC1–PC2 and PC3–PC4 (animal, GRU). The PCs are uncorrelated, so the squared
  lengths sum to the in-sample R².
- **Result.**
  - **The stickiness family** (1–6; switch rate with the opposite sign) shares one
    direction: −PC1, +PC2, −PC3.
  - **Bias** (12–13) points +PC1/+PC2/−PC4, within 10° of orthogonal to everything else.
  - **The slow-learning group** (choice memory span, poorer-side rate, return after a failed
    switch) points +PC3.
  - **Not orthogonal to stickiness.** The slow-learning group is anti-aligned with the RL
    learning rate (−0.96), and that learning-rate axis sits only 52° from determinism
    (embedding-covariance cosine 0.61; r5 item 13).
  - *(Corrected 2026-10-27: this entry first read "three directions", implying three
    orthogonal aspects. Only bias is orthogonal.)*
  - **The stickiness ratio leaves the family on PC3.** It points +PC3 (animal +0.21, GRU
    +0.16), while determinism and win-stay point −PC3. This is the shape component F7 found
    beyond determinism. In the animals its PC1–PC2 part is weak.

### `fig_F10d_embedding_feature_encoding.png` — how much, and on which PCs
- **How.** The F10c correlations as heatmaps; F7's cross-validated rank R² against
  reliability²; the share of each animal feature's in-sample R² on each PC.
- **Result.**
  - **Near the reliability ceiling:** determinism, win-stay, lose-stay, switch rate, bias
    and matching bias.
  - **Far below it in the animals:** stickiness ratio (0.10 of 0.53), stay through losses
    (0.26 of 0.43), matching sensitivity (0.13 of 0.44) and reward memory span (0.03 of
    0.15).
  - **PC3 alone carries** choice memory span, the poorer-side rate and return after a
    failed switch (≥ 75%).
  - **PC4 carries only** bias and matching bias (about 30%).

### `fig_F10e_embedding_feature_tuning.png` — each feature along its held-out embedding direction
- **How.**
  - Each animal feature's rank is predicted from the embedding (5-fold CV linear); mice
    are binned by deciles of the prediction.
  - The median and IQR of the animal and the GRU feature per decile, in raw units.
  - The top − bottom decile contrast as a share of a perfect sort.
- **Result.**
  - **Range in raw units:** win-stay 0.81 → 0.97, lose-stay 0.76 → 0.91, switch rate
    0.07 → 0.21, choice memory span 0.6 → 1.2, bias −0.28 → +0.30.
  - **Contrast against a perfect sort:** 86–95% for the stickiness core and the bias; 45%
    for the animals' stickiness ratio and matching sensitivity; 18% for reward memory span.
  - **The GRU's matching sensitivity runs about 0.1 below the animals' at every decile**,
    with the same slope: the GRU undermatches more.

### `fig_F13_prototype_state_space.png` — the F11 example mice in the GRU's state space
- **How.**
  - B3f's view for the six F11 mice: rows are switch rate, learning rate and bias; columns are
    the low and high mouse.
  - All of each mouse's eval trials in hidden-state PC1–PC2, coloured by the GRU's P(right), over
    B3f's grey background (150,000 trials of all mice).
  - `f13_prototype_state_space.py`.
- **Result.**
  - **Switch rate:** the low-switch mouse fills the confident outer ends of both lobes; the
    high-switch mouse stays nearer the middle.
  - **Bias:** each bias mouse fills its preferred side's lobes.
  - **Learning rate:** the two mice look alike in PC1–PC2.
  - Density also scales with trial count.

### `fig_F12_context_learning_examples.png` — the context path and the change in switch rate
- **How.**
  - Each eval session's context is projected into F10's prototype basis and mapped to predicted
    switch rate by a cross-mouse linear map (mean context → animal switch rate; CV R² 0.80).
  - Left: one mouse's context path over the mice, coloured by their switch-rate percentile.
  - Middle: its per-session switch rate (animal, GRU) and the context-predicted curve against
    training phase.
  - Right: run-length survival in its first vs last 25% of eval sessions (F11's method,
    session-bootstrap SEM).
  - The population test is not drawn, so the caption must carry it: within-mouse ρ for 191 mice
    with ≥ 15 eval sessions against a phase-matched null (another mouse's predicted curve).
  - `f12_context_learning_examples.py`.
- **Result.**
  - Median ρ 0.18 against a null 95th percentile of 0.09; 73% of mice positive.
  - The example (642221, the top of 191) falls 0.23 → 0.03. The context predicts the
    direction and the early timing, about a third of the size.
  - Its P(run ≥ 40) rises from about 0.04 (first four sessions) to about 0.30 (last four).
  - Side preference does not beat the null; adaptation is at it (0.042 vs 0.040). Neither is
    drawn (r5 item 15).

### `fig_F11_prototype_examples.png` — two example mice along each prototype gradient
- **How.**
  - Per prototype (rows), two Uncoupled Baiting mice at opposite ends of the
    embedding-predicted gradient, with the other two prototypes near the median.
  - Columns:
    - their positions in the F10f_best plane;
    - one representative session each (first 400 trials: reward probabilities, the animal's
      10-trial P(right), the GRU's P(right), choice ticks);
    - a targeted summary over all their task trials: run-length survival (switch rate; ± SEM
      from 500 bootstraps over sessions, since runs within a session are not independent),
      P(favoured side) around reward-probability changes (learning rate), P(right) against
      pR − pL (bias).
  - `f11_prototype_examples.py`.
- **Result.**
  - The contrasts are clear in all three.
  - **Caption must carry the population check**, because the panel no longer shows it: across
    330 mice, ρ = 0.45 between the late rise after a reward-probability change and the
    embedding-predicted learning rate.
  - See r5 item 14.

### `fig_F10f_prototype_basis.png`, `_points.png`, `_best.png` — the embedding in a basis built from three prototype features
- **How.**
  - Prototypes: switch rate (stickiness level), CTT learning rate, bias (animal features).
  - Their rank directions in the whitened embedding (where the embedding-covariance metric is
    Euclidean) are Gram–Schmidt-orthogonalised in that order: axis 1 = switch rate,
    axis 2 = learning rate ⊥ switch rate, axis 3 = bias ⊥ both, axis 4 = remainder.
  - Each plane (1–2, 1–3, 2–3) is coloured by each prototype (smoothed rank as in F10a), with
    the three gradient arrows.
  - `_points.png` is the same figure with each mouse as a dot and no smoothed background.
  - `_best.png` shows one panel per prototype, in the plane that shows its gradient best: switch
    rate on 1–3, learning rate on 1–2, bias on 2–3. Points only, equal panels on one
    shared scale. "Decoding R²" in each title is the 5-fold cross-validated rank R² from the
    4-d embedding (F7 / F5: 0.71, 0.67, 0.65).
- **Result.**
  - **Switch rate is axis 1** (r 0.85 with the coordinate).
  - **The learning rate** is −0.62 on axis 1 and +0.78 on axis 2, a diagonal gradient 129°
    from switch rate.
  - **Bias is 0.95 on axis 3** and 0.31 on axis 2.
  - **Axis 4 carries none of the three.**
- **Caveat.** The order sets which shared part goes where. Switch rate goes first because it
  is the most reliable and model-free.

### `fig_F5a_embedding_atlas.png`, `_pc3_pc4.png` — the embedding coloured by everything
- **How.**
  - Each mouse is shown in the embedding's own PC1–PC2 (and PC3–PC4), coloured by one
    variable per panel. Continuous variables are shown by rank; categorical ones by their top
    classes.
  - The variables are RL parameters, GRU features, dynamics (F2), fit quality and session
    metadata from the pinned snapshot.
  - `f5_embedding_atlas.py`.
- **Result.**
  - **Gradients appear in the strategy variables.** The learning rate and gain run along
    PC3, and the bias along PC4.
  - **Session metadata shows none.**

### `fig_F5b_embedding_directions.png` — what is decodable, and a dictionary of the embedding PCs
- **How.**
  - Left: the rank R² of every variable from the embedding (5-fold CV, shuffle null).
  - Right: the Spearman correlation of each decodable variable with the four embedding PCs.
- **Result.**
  - **Strategy and dynamics are decodable** (0.5–0.84); non-choice metadata isn't.
  - **PC3 is the learning-rate / gain axis.**
  - **PC4 is the bias axis.**
  - **PC2 is attractor depth.**
  - **PC1 mixes them** with predictability.

### `fig_F5c_embedding_batch.png` — setup and cohort labels
- **How.**
  - Balanced accuracy (standardised inputs, class-balanced logistic, 5-fold CV) for
    institute, data source, rig type, room, trainer, PI, curriculum and first year.
  - Decoded from: the embedding; the task mix; the animals' own kernels; and animal
    behaviour + task mix (the control).
  - Right panel: the gain from adding the embedding to the control, with an
    embedding-shuffle null.
- **Result.**
  - **The embedding alone decodes several labels** (institute 0.79).
  - **The task mix decodes most of them better;** the animals' kernels do better only for
    institute and data source.
  - **Adding the embedding to behaviour + task mix gains nothing** (−0.05 to +0.028).
  - So no setup or cohort signature is detectable beyond task exposure and behaviour.
- **Revised 2026-10-24.** The first version used the GRU's own kernels (circular), folded
  task mix into behaviour, and didn't standardise inputs.

### `fig_F6_embedding_types.png` — types or a continuum?
- **How.**
  - Gaussian mixtures with k = 1–8 on the embedding and on the animals' standardised
    kernels, compared by BIC.
  - Two nulls: a single Gaussian, and a heavy-tailed multivariate t with its df matched to
    the data's kurtosis.
  - Bottom: the k = 2 components in the embedding's PCs; Mantel tests of embedding distance
    against behavioural distance (permutation null).
  - `f6_embedding_types.py`.
- **Result.**
  - **The embedding's k = 2 preference is a core plus a broad halo** (means about 1 SD
    apart, 1.7× scale), at the edge of what heavy tails alone produce.
  - **The behaviour's k = 2 is a 3% outlier group** in the decay-ratio feature.
  - **Distances correspond** (Spearman 0.49–0.57).
  - So the mice form a continuum, not types.

### `fig_F1_embedding_extremes.png` — mice at the ends of the gain and bias axes
- **How.**
  - The 10% of mice (61) at each end of F3's GRU-gain and GRU-bias directions.
  - Gain row: update maps folded by chosen side, and per-mouse reward kernels for animal and
    GRU.
  - Bias row: P(right) distribution and occupancy in PC1–PC2.
  - `f1_embedding_extremes.py`.
- **Result.**
  - **For high-gain mice the GRU** jumps toward the rewarded side (log-odds ≈ +2) when it
    leaned away or was neutral, and moves along the diagonal beyond that; **for low-gain mice**
    it updates incrementally (the maps are the GRU's logits). The GRU matches the
    animals at both ends (reward effect one back 1.30 vs 1.33 and 0.59 vs 0.53).
  - **The bias axis** shifts time between the two choice lobes: mean P(right) 0.37 vs 0.64,
    model and animal alike.
- **Caveat.** The groups were selected on a GRU-feature direction, so only the animal-side
  contrast is evidence.

### `fig_F4_early_late.png` — change between early and late sessions
- **How.**
  - Each mouse's eval sessions are split at the median training-session index.
  - Per-half logistic fits for the animal and the GRU; the GRU's change is compared with the
    animal's across mice.
  - Repeated within each mouse's dominant task.
  - The noise ceiling refits the GRU on choices sampled from its own P(right).
  - `f4_early_late.py`.
- **Result.**
  - The GRU follows each mouse's change **to the noise ceiling**: bias 0.92 (ceiling 0.85), gain
    0.59 (0.59), persistence 0.93 (0.90).
  - The one discrepancy is the *average* gain shift: animals +0.05, GRU −0.035.
  - It is suggestive given the number of tests (p 0.003 / 0.021).
- **Corrected after review.** "The GRU does not track gain changes" was at the noise ceiling and
  is withdrawn.

---

---

## Phase H — the GRU's strategy vs Q-learning (report r8)

### `fig_H1_model_ladder.png` — the model ladder
- **How.**
  - Cumulative rungs:
    - R0: Q-learning;
    - R1: + a one-trial choice kernel;
    - R2: + a decaying choice trace;
    - R3: + outcome scaling;
    - R4: + reset on a switch;
    - R5: + an outcome-split last-choice kernel.
  - Each rung is fitted per mouse to the GRU's P(right) with Adam on the GPU.
  - Panels: animal likelihood, fit to the GRU, share of mice beating CTT, and C4e signature
    errors.
  - `h1_model_ladder.py`.
- **Result.**
  - **R0 reproduces Hattori** (0.7169), and **R1 reaches CTT** (0.7212).
  - **R2, the choice trace, is the step that matters:** 0.7240, 88% of mice beating CTT,
    and the "stay through losses after a switch" signature fixed.
  - **R3–R5 add little,** to 0.7251.
  - **The plateau closes 32% of the CTT → GRU gap.**

### `fig_H2_learning_rate.png` — effective learning rate by state
- **How.**
  - For 200,000 trials, flip one reward and measure the next trial's log-odds, signed by
    side. GRU via `GruStep`; Hattori and Bari by exact numpy replay.
  - Binned by: confidence, P(chosen), run length, trials since a reward, recent reward rate,
    position, stay vs switch, and task. Models side by side.
  - `h2_learning_rate.py`.
- **Result.**
  - **Hattori's effect saturates** as Q nears 1: it falls with run length, confidence and
    reward rate, and is 2× larger after a switch.
  - **The GRU's is flat over run length, confidence and stay/switch,** and falls only
    moderately with the reward rate (1.12 → 0.83, against Hattori's 1.14 → 0.36).
  - **The GRU does raise its effect after a dry spell** (0.85 → 1.2) and for unexpected
    choices (1.28).
  - **So the GRU's update saturates far less than a delta rule,** and its rate is not
    meta-adaptive in the usual sense.

### `fig_H3_q_phase_portraits.png` — dynamics in Q-learning's coordinates
- **How.**
  - Hattori's Q_L and Q_R are decoded linearly from the GRU state (held-out R² 0.95 / 0.94).
  - Per input on trial t, the binned mean one-trial change in the (Q_L, Q_R) plane is drawn
    for the GRU (decoded), Hattori (true Q) and Bari (true Q), side by side. Grey shows
    occupancy.
  - Right column: slopes of ΔQ = a + b·Q_chosen + c·Q_unchosen.
  - `h3_q_phase_portraits.py`.
- **Result.**
  - **After a reward, the GRU looks like Hattori:** the chosen value moves at rate about
    0.42 and the unchosen side is forgotten at about 0.7.
  - **After a loss it differs:**
    - The GRU's chosen value relaxes slowly (0.10–0.13) toward about 0.25–0.4.
    - Hattori leaves it almost untouched (0.05).
    - Bari drives it toward 0 (0.25).

### `fig_H4_scenarios.png` — identical scripted histories
- **How.**
  - Each mouse starts a fresh session with its context (GRU) or fitted agents (Hattori,
    Bari, CTT) and runs through the same scripted histories.
  - Scenarios A–E: a win streak then losses, a failed switch, a rewarded switch,
    alternating outcomes, and a reversal.
  - Sweeps: F, return after a failed switch against run length; G, staying through losses
    on a new side.
  - Shown as the median and IQR across mice, models side by side. `h4_scenarios.py`.
- **Result.**
  - **Sweep F:** the GRU's return after a failed switch grows with run length
    (0.46 → 0.73). Hattori is high and flat, Bari slow, CTT flat (0.41).
  - **Sweep G:** after switching, the GRU's P(stay) rises with each further loss
    (0.28 → 0.80). Hattori stays flat (0.34 → 0.49).
  - **One outcome on a new side:** the GRU discounts it more than any RL model.
  - **Alternating outcomes:** the GRU's policy is the smoothest.

## Phase G — robustness across seeds (report r7)

Seeds 1–2 of model C (same configuration as seed 0, different initialisation and batch order)
were extracted and run through 35 producers unchanged. Their figures and JSONs sit in
`analysis/seeds/C-s1/` and `analysis/seeds/C-s2/` under the same names as seed 0's, so any
figure above can be compared seed by seed.

### `fig_G1_seed_robustness.png` — every headline number, three seeds side by side
- **How.**
  - `g1_seed_robustness.py` reads each seed's producer JSONs and lists the number behind each
    SUMMARY claim: one column per seed, plus each seed's % difference from seed 0 (symlog
    axis, full range).
  - 18 claims are scored with one test each, with thresholds fixed from seed 0.
  - Only basis-free quantities are compared.
  - RL baselines' own numbers are left out of the figure, because they are identical by
    construction (the same fits on the same trials); they are kept in the JSON.
- **Result.**
  - 14 of 18 claims hold in all three seeds.
  - Seed 1 fails four: v through switches, v closure, the v ablation against the worst random
    direction, and attractor reward depth.
  - The largest spreads are seed 0's L+ bistability (79 / 0 / 9 mice), the per-mouse slowest τ
    (355 / 1670 / 1635 trials) and the embedding PC variance shares.

### `fig_G3a_state_space_by_seed.png` — the state space of each seed in its own PCA
- **How.**
  - PC1–PC2 and PC1–PC4 coloured by the t−1 input, for 40k of `g3_cross_seed_alignment.py`'s
    200k-trial subsample.
  - Each seed is shown in its own PCA, with PC signs flipped to correlate positively with
    seed 0's.
- **Result.**
  - The same four clusters in two choice lobes in every seed, with PC2 separating rewarded from
    unrewarded.
  - η² of PC1–4 by the last input is 0.66–0.67.
  - Seed 2's PC1 is larger (49%).

### `fig_G3b_cross_seed_alignment.png` — how alike the seeds are
- **How.** Five panels:
  1. Canonical correlations of top-10 PC scores per seed pair.
  2. Per-trial logit(right), seeds 1 and 2 against seed 0.
  3. The same for the value coordinate v (each seed's own I2 axis, so only r is comparable).
  4. Canonical correlations of the 4-d subject embeddings.
  5. Spearman ρ between the seeds' Mahalanobis between-mouse distance matrices.
- **Result.**
  - **State:** CCA 0.998 → 0.88 over the first 9 components.
  - **Logit:** r 0.998 (seed 0 vs CTT 0.970).
  - **v:** r 0.989–0.994.
  - **Embedding:** CCA 0.84–0.985; distances ρ 0.89–0.93.

## Exploration-only outputs (not committed)

`_cache/explore/` holds the full B3 exploration: every colouring and every 1–2-back condition
in PCs 1–4, one figure per task, full-session trajectories, per-mouse panels in three PC
planes, and PCs 5–6 under four colourings. It also has `B3g_viewer.html`, an interactive 3-D
viewer of any three of PCs 1–6 under six colourings.

## Still to do

- Trajectory linearisation: Jacobians along real trajectories, compared with C1's kernel.
- Test whether a linearly carried value plus the readout reproduces B4's nonlinear maps.
- Close the rest of I7's gap with a nonlinear update for the choice×reward coordinate.
- H1 follow-ups (R6 non-saturating update, train-session refit); embedding traversal.
- G2: model B (H256) and its comparison with C (G1/G3 for C seeds done, r7).
