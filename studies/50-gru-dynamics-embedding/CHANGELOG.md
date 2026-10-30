# Changelog

## 2026-10-29

- `make all` runs offline (CI): only the report renderers. Every producer that reads the
  cache, the GPU, W&B or the wrapper is behind `freeze-rN` (`make freeze` for all);
  `environment.lock` pins wrapper `56ab3ee`.
- Every analysis JSON carries `_meta` v2, re-frozen on the cached A1 tables and re-extracted
  from the cached checkpoints (A1 parquets byte-identical). c1_impulse, d_dynamics, i3, i8, e2 and
  e2b are pinned to CPU, as first produced. Against the previous JSON every number agrees
  within 1e-9 except the GPU fits h1/i9e (3rd decimal; r8 `w_T` 0.295 -> 0.297, `kappa_unrew`
  5.06 -> 5.07, `w_ck_unrew` 0.455 -> 0.457) and two C-s2 seed JSONs (<= 2e-4). No claim
  changes. r6's RL medians come from the new frozen `rl_fit_medians.json`.
- Reports cite the source studies' W&B groups as `upstream {study, wandb_group}`;
  `wandb_groups` names this study's own launches, which there are none of.

## 2026-10-28

- Migrated from the dispatcher's `studies/pk-01-gru-dynamics-embedding/` (never committed there).
  Study pk-01 is now 50; older notes and W&B text may say pk-01 or 12.
- Layout for this repository: `source_runs.json` moved to `analysis/`, variant `base` added,
  non-`rN` summaries moved to the study root, report frontmatter on schema v1.
- PNG figures are not committed (gitignored) until they are redrawn as SVGs.

## 2026-10-15

- Created the study. Pinned the D=614/E4 GRU sources: C (H128, `e4-e8-d614-source`,
  primary) and B (H256, `nxd-grid`, next); both frozen from W&B.

## 2026-10-16

- A1: per-trial hidden-state table for C seed 0 (wrapper `gru_trial_states`), with the
  session table's `task`; likelihood and readout-alignment checks recorded.
- Phase B producers (B1–B7) and r1 draft. B5 switched to a logit-matched reward effect
  (raw curves were regression to the mean); B7 locates variables by activation pattern,
  not decoder weight, and adds unique R² for history lags.
- Phase C and r2 draft. Pinned Study 01's Bari/CTT/Hattori per-mouse fits (actual W&B
  groups recorded; Study 01's script hardcodes other launch suffixes). Replays reproduce
  every mouse's logged eval likelihood. C1 counterfactual flips ran on CPU after the
  container lost GPU access.

## 2026-10-19

- Table-only parts of E and F (r4, r5 drafts): E1 geometry without slow modes; F3 embedding
  decoding with a task-mix control (replicates Study 01 r10 on model C); F1 embedding
  extremes; F4 early vs late within mice. E1's static readout-removal decoding test was
  dropped as uninformative (redundancy) — noted in r4.
- Review fixes (independent reviews of the wrapper code and of the analyses): C1 switched
  from the rank-deficient Hattori2019 design to Su2022, added a 20-lag check and the
  distribution of counterfactual flip effects; C2 advantage split by pre-choice predicted
  P(switch) instead of the realised switch; F3 cosines in the embedding-covariance metric;
  F4 noise ceiling; r1/r2/r4/r5 claims corrected (see each report's "Corrections").
- A2: wrapper `gru_step` (free-context step, parity-tested). D rerun in float64 with forward
  iteration + Newton refinement after the first fixed-point search stalled at q ~ 1e-5.

## 2026-10-20

- New phase I (integration axis, r6): per-cluster encoding axes (I1), targeted
  dimensionality reduction (I2), value-coordinate transitions and attractors (I3). G's
  report renumbered r6 → r7 in PLAN.md. Wrapper `gru_step.find_fixed_points` (forward
  iteration + damped Newton, tested) used by I3; D keeps its own earlier copy.
- B3c: added a PC1–PC4 version of the previous-outcome figure.
- E2 ablations (r4). Global-mean replacement (E2) turned out off-manifold (random directions
  as costly as meaningful ones), so E2b replaces the ablated component by the state's own
  last-input-cluster mean, with 10 random 1-d and 2-d controls. Wrapper
  `GruStep.run_ablated` (global or per-input means; tested).
- I4–I6 (r6): distilled one-variable model (I4), per-mouse dynamics parameters vs F3
  features and the embedding (I5), the same transition fits on Hattori/Bari ΔQ and on every
  model's policy logit (I6). Shared helpers in `analysis/_transitions.py`. Groups for first
  trials of a session are fitted as constants (a slope is unidentified there).

## 2026-10-21

- I7 (r6): second carried coordinate for the distilled model, with n-d helpers in
  `_transitions.py`. Corrects r6 item 6: with per-mouse readouts on both sides, ~0.010 of
  I4's gap is the readout and ~0.006 the dynamics. The earlier claim compared a pooled
  readout with per-mouse simulations.

## 2026-10-23

- B3c deeper history: `b3c_history.py` writes `fig_B3c_prev_outcome_t2/_t3` (+ PC1–PC4),
  splitting the previous-outcome figure by the last two (16 panels) and three (64 panels)
  inputs.
- C2c in every PC pair: `c2c_pc_pairs.py` writes `fig_C2c_advantage_pc_pairs.png` (mean and
  share-of-total advantage over CTT for the six pairs of PCs 1–4).
- C4 (r2): behaviour against the animals: logit-change maps (C4a), update calibration (C4b),
  three-trial-history calibration (C4c), GRU–RL disagreement (C4d). `c4_behaviour.py`.
- C4e (r2): two behavioural signatures, return after a switch vs the old run's length and
  P(stay) through loss streaks, animals vs each model. `c4e_signatures.py`.
- I8 (r6): causal tests of v — closure, perturbation along v's axis, and input-flip
  kernels for the GRU vs Hattori/Bari (numpy RL replay validated against the cached
  rollouts). Corrects the "v is closer to a choice kernel than to ΔQ" wording: ΔQ with
  fitted forgetting is choice-dominated too, and the GRU-specific properties are a
  separable choice trace and a short-lived reward memory.

## 2026-10-24

- GPU restored after a container restart. F2 (r5): attractors, timescales and v's retention
  for all 614 mouse contexts. Bias shifts the attractors; reward gain shows up as v's
  retention, not as attractor depth; slow modes are generic. Reconciles D2's "gain deepens
  attractors" (D2's contexts were chosen along the combined gain + persistence direction).
- H3, H4 (new report r8): the GRU's dynamics in Q-learning's coordinates, and identical
  scripted histories through the GRU and the RL fits.
- H1 (r8): model ladder fitted to the GRU. A choice trace is the component that matters;
  the ladder plateaus at 0.7251. `c4e_signatures.py` refactored to expose
  `signature_curves` (output unchanged).
- F5 (r5): embedding atlas, a dictionary of the embedding PCs, and a batch-effect check;
  session metadata cached from the snapshot's session table.
- F5c batch test redesigned (animal kernels as behaviour, task mix as a separate control,
  standardised inputs, incremental test with an embedding-shuffle null). The conclusion
  stands.
- H2 (r8): effective learning rate by state. The GRU's is steady (no Q-saturation), with
  a boost after dry spells and for unexpected choices.
- F6 (r5): types vs continuum — mixtures with Gaussian and heavy-tailed nulls, Mantel
  tests. A continuum with heavy tails.

## 2026-10-26

- F4b (r5 item 11): how the context moves with training, with a counterfactual rerun of
  late sessions using the early context. F4 now caches per-mouse deltas and session halves
  (its results are unchanged, checked).
- Four independent reviews (phases A–C; D, E, I; F, H; wrapper + bookkeeping). Fixed:
  - **Bugs:**
    - B6 and H2 binned discrete values on bin edges (floating-point splits); bins are now
      centred on them.
    - F2 missed attractors; it reran with 96 starts per mouse and input, and reports the
      mice without a converged attractor.
    - H1 reported λ_T = 0.73, which was its initial value; it now starts neutral, saves
      per-mouse parameters and reports the spread.
    - F5 treated missing / "None" as a class and had duplicate trainer names.
    - B1's running mean zero-padded the session ends.
  - **Claims corrected:**
    - **r1:** PC5 carries outcome too; B5's decay; B3f is not causal.
    - **r2:** flip vs logistic; 20-lag wording; history order stated.
    - **r3:** D2 depth vs gain (F2); timescale and convergence ranges.
    - **r4:** v's E2b role; cosine range.
    - **r5:** F1 wording; trainer decodability; F6 null.
    - **r6:** Hattori α_unrew; attractor vs v*; ratios; switch reset.
    - **r8:** stale H2 expectations; sweep and R² numbers.
    - **FIGURES.md, INDEX, HANDOFF, README, PLAN:** stale status lines.
  - **Figure layouts:** B3c t−3, C4c, H1, H2, F2, F5c, F6, F4b.
  - Remaining minor items, not fixed: D1 arrow density, D2 colour key, E2 legend (superseded
    figure), C2a truncated axis, C1 SEM band.
- F4c (r5 item 11): context trajectories over every training session. They correct "the
  context barely moves": the variance share is 3%, but the distance is up to about 1
  between-mouse SD, front-loaded and non-monotonic (PC4 swings up and back, which explains
  F4b's opposite-sign half-split drift).
- F7–F9 (r5 item 12): behavioural feature library with reliabilities; varimax-named
  embedding axes; causal traversal along them (GPU); per-mouse strategy profiles, prototypes
  and cards.

## 2026-10-27

- D1b (r3, SUMMARY): repeated constant inputs from real states confirm D's fixed points and
  explain why D1's one-step arrows don't meet at the attractors. D1 caption clarified.
- I3b: I3's transitions regrouped by stay vs switch, inputs by colour.
- D1c: the flow field at two scales (direction + step-size map; zoomed slow drift), the readable replacement for D1.
- D1d: D1 with arrow length scaled to the step size, linear and log10.
- I9 (r6 item 11): v's history kernels and a test of the 'choice trace × outcome' description. Refined to: a choice trace (λ 0.82) plus a faster-fading reward boost (μ 0.62).
- I10 (r6 item 12): gate decomposition of v's retention. v is held by recurrent re-writing (candidate term), not a closed update gate.
- F10 (r5 item 13): F7's behavioural features mapped over the embedding PCs (smoothed maps, direction arrows, PC shares, held-out tuning in raw units). Stickiness is one direction across PC1–PC3, slow learning is PC3, bias is the other PC1–PC2 diagonal plus PC4; the animals' stickiness ratio, reward memory span and matching sensitivity are weakly mapped.
- D1e / D1f: flow fields with each state's own context and every mouse's attractor; prototype mice.
- F10 direction cosines (r5 item 13, F7b and F10c entries): only bias is orthogonal to the other aspects. Stickiness (determinism) and the RL learning rate are 52° apart (cosine 0.61), not orthogonal. Corrects F10's 'three directions' wording and notes that F7's named axes are orthogonal only by construction (varimax).
- I2b: the value axis (with lobe, readout, choice and reward axes) drawn in PC space.
- F10f (r5 item 13): the embedding in a basis built from the prototypes switch rate → CTT learning rate → bias (Gram–Schmidt in the whitened embedding). The learning rate is a diagonal in plane 1–2 (129° from switch rate); bias is its own axis.
- F11 (r5 item 14): two Uncoupled Baiting example mice per prototype (switch rate, learning rate, bias), matched on the other two; session traces and targeted summaries; block-change adaptation vs predicted learning rate ρ = 0.45 across 330 mice. F10 now caches its prototype basis for F11.
- I9b (r6 item 13): per-mouse kernel fits. v's rising outcome ratio and μ < λ hold in 99–100% of mice (opposite in Hattori); λ is the slow-learning axis.
- F11 fix: the bias panel binned pR − pL with edges at ±0.3, splitting that condition by float rounding (spurious peak at +0.2). Now snapped to the task's grid (0, ±0.3, ±0.6); r5 item 14 corrected.
- I9c (r6 item 14): lag-1 cluster offsets leave v's older kernel unchanged; a two-outcome-trace description (λ_unrew 0.83 > λ_rew 0.68 in 100% of mice) maps onto the determinism and slow-learning axes.
- I9d (r6 item 15): K = 20 lags and a three-trace model. Three traces ≈ free fit (lowest BIC in 99.5% of mice): a fast choice trace (λ ≈ 0.67), a reward boost (λ ≈ 0.71) and a small non-decaying unrewarded residue. Corrects I9c: λ_unr 0.83 was a compromise of the fast and slow parts; SUMMARY and r6 item 14 reworded.
- I9e (r6 item 16): the three-trace description as a per-mouse update-rule model fitted to the animals (held-out, 2-fold). Best refitted model (0.7231 vs Q + kernel + last 0.7219; GRU 0.7339; 45% of the Q → GRU gap). Behaviour shows the slow unrewarded residue (λ_u 0.98 > λ_w in 99% of mice) but not the GRU's switch signatures.
- F12 (r5 item 15): context paths over training vs per-session behaviour. Switch rate beats a phase-matched null (median within-mouse ρ 0.18 vs 0.09); side preference and adaptation do not. Example mouse 642221. F10 now also caches its basis transform.
- F13 (r5 item 14): the six F11 mice in hidden-state PC1–PC2 (B3f's view). Switch and bias contrasts are visible as lobe depth and side occupancy; the learning-rate pair looks alike. F12 subtitle drops ρ.
- B3h: summary view of the state space, PC1–PC2 and PC1–PC4 coloured by P(right) and by the t−1 input.
- D1f: all four inputs' fixed points in one panel plus each input's log-length flow field, in B3h's input colours.
- D1f revised: only fixed points reached by real states are drawn (the deep L-rewarded basin and the unstable point are reached by 0 of 300 starts); flow arrows at full length.
- I11 (r6 item 16): v carried across lobes in the (choice axis, v) plane: session ladder, per-input flow, switch-triggered averages (c and the logit flip at once, v crosses over ~3 trials).
- I11b: the (c, v) plane coloured by t−1 input and by P(right); each lobe has a main and a just-switched band, and the prediction follows the lobe first.
- B3h companion: PC1–PC2 and PC1–PC4 coloured by v; v is graded along −PC1, while P(right) is two-toned by lobe.
- I11c (r6 item 16): carry-over shown in PC space; post-switch states coloured by the pre-switch v reproduce their own-v gradient (ρ 0.975; shuffled within direction 0.20).
- G1 / G3 (r7): seeds 1–2 of model C fetched at their pinned digests and extracted (likelihoods reproduce the logged values to < 1e-7); `_pk01.MODEL_KEY` now comes from `S12_MODEL_KEY`, and any non-default key writes to `analysis/seeds/<key>/` (caches stay shared; `keyed_cache` for the two unkeyed cache files; e2 and f9 path fixes). `g_run_seed.sh` reran 35 producers per seed. The seeds are one model up to a basis change; 14 of 18 headline claims pass fixed tests in all three seeds; seed 1 fails four (v through switches, closure, ablation vs worst random direction, reward depth). Corrections: the embedding PC dictionary (PC3 = learning rate, PC4 = bias), the L-rewarded extra fixed points and bistable mice, and the slow-mode τ are seed 0's, not the solution's. SUMMARY items 7 and §8 updated.
- B3a truncated to 10 PCs (`fig_B3a_pca_variance_10.png`, plot-only from the cached PCA). F12 middle-panel y-range reverted to autoscale.
- D1f 2×2: the four flow panels alone (`fig_D1f_flow_2x2.png`), rows = side, columns = outcome.
- I11d: three-panel summary of v's carry-over (I11 session in the (c, v) plane + I3b stay/switch maps).
