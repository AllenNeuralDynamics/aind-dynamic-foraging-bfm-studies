# Study 50 plan

Model **C seed 0** first (`e4-e8-d614-source-20260906-195409-c160507b`, H128/E4/D614), on
the 614 training mice's eval sessions, computed on the local GPU box. Every figure is split
by curriculum (Coupled Baiting / Uncoupled Baiting / Uncoupled Without Baiting) where that
matters. Phases are ordered simple → complex. **Stop after F** and review r1–r8 before G.

Reuse wrapper code rather than copying it; new reusable code goes into the wrapper's
`code/post_training_analysis/` with tests.

## A. Extraction
- **A1** One per-trial table for the whole split: subject, session, trial, curriculum, choice,
  reward, true reward probabilities, logits / P(right), hidden state `h_0..h_127`. Written in
  the column schema the existing `state-space-*` functions read, so they run on it directly.
  Done when the held-out likelihood recomputed from the table matches W&B.
- **A2** (before D) `step(h, choice, reward, c)` with the context `c` as a free input; parity
  test against the full model.

## B. Descriptive (forward pass + plots) → r1
- **B1** Example sessions per curriculum: choices, rewards, reward probabilities, model P(right).
- **B2** Unit heatmap (128 units × trials, sorted by correlation with the logit) for one session.
- **B3** PCA exploration
  - B3a variance explained (20 PCs), cumulative, participation ratio.
  - B3b pairwise grid of PCs 1–4, one figure per colouring: existing overview colours plus
    curriculum, true reward-probability difference, trials since each side was last chosen.
  - B3c same grid by condition (prev action × outcome; history pattern 1–3) — `state-space-condition`.
  - B3d same grid per curriculum, shared PCA.
  - B3e example trajectories, one session per curriculum — `plot_latents_in_space(example_run=…)`.
  - B3f per-mouse panels for ~9 mice spread over the embedding — `state-space-subject`.
  - B3g interactive 3-D viewer (any PCs, any colouring) in `analysis/explore/`.
  - If PCs 5–6 each carry >~5% variance, add one PCs 5–6 vs 1–2 figure for the colourings
    that showed structure. Exploration outputs stay in `_cache/`; chosen views become producers.
- **B4** Update maps: logit(t+1) vs logit(t), 4 panels (choice × reward).
- **B5** Outcome-triggered average: Δlogit over the 10 trials after each of the 4 inputs.
- **B6** Tracking: model P(right) and animal choice vs true reward-probability difference (binned).
- **B7** Ridge decoders from all 128 units, cross-validated, one per variable (last choice/reward,
  joint 1–5-back reward and choice history, true p-difference, trials since each side chosen):
  R², decoder weights in PC coordinates, cosine similarity between decoder axes.

## C. Versus animals and RL → r2
- **C1** Impulse responses (flip one trial's reward) vs logistic-regression kernels of the animals.
- **C2** GRU vs fitted RL P(right) trial by trial; GRU-vs-RL advantage mapped in PC space.
- **C3** Add RL Q-value difference/sum to the B7 decoders.

## D. Dynamics → r3
- **D1** Flow fields in PC1–2, one per input.
- **D2** Fixed points per constant input (batched search on GPU).
- **D3** Jacobian spectra and time constants; slowest τ vs B5 decay and RL learning rates.
- **D4** Overlap of slow modes with B7 decoder axes.

## I. Integration axis — targeted dimensionality reduction → r6
Motivation: the state falls into four clusters by the last input (previous choice × outcome;
B3c), each spanning a range of P(right). Is the within-cluster decision variable carried along
one shared integration axis, with inputs moving the state between clusters, or along
cluster-specific axes? (Added 2026-10-19.)
- **I1** Per-cluster encoding axes u_c = cov_c(h, v) / var_c(v) for v = the model's logit
  (u_c ∝ Σ_c r), Hattori ΔQ and the true P(rew|R) − P(rew|L) (non-circular checks); 4×4
  cosines across clusters, cosines with the readout r and PCs; share of within-cluster
  variance along u_c; local axes per value tercile (curvature); with context and trial
  position partialled out.
- **I2** Targeted dimensionality reduction: regress h on [previous choice, previous reward,
  choice × reward, value] + nuisances (4-d context, trial position); orthogonalise with value
  first; project states and plot value vs choice / reward axes by cluster.
- **I3** Dynamics in TDR coordinates: t → t+1 transitions per input — is the value coordinate
  carried across the jump between clusters? Same-side attractors (from the step function)
  projected onto the axes.
- **I4** Distilled one-variable model: v(t+1) = v*_g + λ_g (v(t) − v*_g), g = input × stay/switch
  (+ first trial), readout logit = a·v + b(last input), fitted to the GRU's value coordinate
  (pooled, and per mouse); simulated on the animals' input sequences without the GRU and
  scored on their choices against the GRU, RL models and E2's 1-d state. (Added 2026-10-20.)
- **I5** Per-mouse λ and targets vs F3's gain / bias / persistence and the embedding: which
  dynamical parameter carries each behavioural axis.
- **I6** The same transition analysis on the RL models' ΔQ (Hattori, Bari) — where the GRU's
  update structure departs from Q-learning. (I4–I6 done 2026-10-20 → r6.)
- **I7** A second carried coordinate for the distilled model: readout residual (bound),
  TDR choice / reward / value×choice / choice×reward axes, PC5. (Done 2026-10-21 → r6.)

- **C4** (added 2026-10-23) Behaviour against the animals: logit-change maps, update and
  history calibration, GRU–RL disagreement. → r2.

## E. Subspace function → r4
- **E1** Subspaces: readout vs readout-null, input-driven, slow, decoder axes.
- **E2** Mean-replacement ablation of each; score with likelihood and `generative` metrics.

## F. Context / embedding → r5
- **F1** B4/B5 small multiples for mice at the embedding extremes.
- **F2** Sweep `c` over all 614 mice: update-map slope/offset, fixed points, τ; colour embedding space.
- **F3** Mechanism behind Study 01 r10: bias part of `c` shifts fixed points? learning-rate part changes gain?
- **F4** Early vs late-session context.

## H. Extracting the GRU's computation vs Q-learning (proposed 2026-10-23; H1–H4 done 2026-10-24)

- **H1 Model ladder.** Relax Q-learning one assumption at a time and fit each rung to the
  GRU's own predicted P(right) (soft labels), then score it on the animals.
  - Rungs:
    - Q-learning (α, forgetting, β, bias);
    - + choice kernel;
    - + a slower, separate choice trace (C4e's run-length memory);
    - + stay/switch-dependent update (I6);
    - + an offset from the last input (I4/I7's b[last input]);
    - unconstrained 2-d (I7);
    - the GRU.
  - Plots: animal likelihood per rung and the share of the GRU − RL gap each step closes.
    Also the C4e curves per rung (which step fixes which failure mode).
  - JAX on CPU. Pooled fits first, then per mouse.
- **H2 Effective learning rate by state.**
  - Measure it two ways: the reward-flip counterfactual (C1 machinery) at many states, as
    the change in next-trial log-odds; and the local slope of v's update (I4).
  - Plot both against GRU confidence, run length, trials since a switch, recent reward
    rate and task, with each RL model's fixed rate as a reference.
  - Tests whether the GRU's learning rate changes with state, versus Q-learning's fixed
    rate.
  - Also the causal check of v's make-up: flip past choices and rewards and measure Δv,
    complementing I7's correlational history weights (see the I7 caveats in r6).
    **Done as I8 (2026-10-23, r6)**, together with the closure and perturbation tests.
    The state-dependent learning-rate part is still open.

- **H1** (done 2026-10-24 → r8) model ladder. Follow-up: train-session refit, an R6 with a
  non-saturating update (H2 found no state-dependent learning rate to add).
- **F5** (done 2026-10-24 → r5) embedding atlas, PC dictionary, batch-effect check.
- **H2** (done 2026-10-24 → r8) reward-flip learning rate by state.
- **F6** (done 2026-10-24 → r5) mixtures and Mantel: types vs continuum.
- **H3** (done 2026-10-24 → r8) The GRU's dynamics in Q-learning's (Q_L, Q_R) coordinates.
- **H4** (done 2026-10-24 → r8) Canonical scripted scenarios: GRU vs the Hattori, Bari and
  CTT fits.

## — stop and review —

## G. Robustness → r7
- **G1** C seeds 1–2. **G2** Model B (H256). **G3** CCA / Procrustes across seeds and models.
- **Done 2026-10-27:** G1 and G3 for C seeds 0–2 (r7). Next: G2 (model B) and G3 across models.
