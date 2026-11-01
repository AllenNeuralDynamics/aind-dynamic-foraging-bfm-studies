---
schema_version: 1
id: r7
slug: robustness
status: draft  # G1 + G3 for model C; G2 model B not yet run
authors: [pck, claude]
wandb_groups: []
inputs:
  scripts:
    - analysis/g_run_seed.sh
    - analysis/g1_seed_robustness.py
    - analysis/g3_cross_seed_alignment.py
    - analysis/report_r7.py
  data:
    - analysis/trial_states_C-s1.json
    - analysis/trial_states_C-s2.json
    - analysis/seeds/C-s1
    - analysis/seeds/C-s2
    - analysis/g1_seed_robustness.json
    - analysis/g3_cross_seed_alignment.json
upstream:
  - study: 01-gru-scaling-law
    wandb_group: e4-e8-d614-source@20260906-195409
  - study: 01-gru-scaling-law
    wandb_group: rl-baseline-bari@20260713-010244
  - study: 01-gru-scaling-law
    wandb_group: rl-baseline-ctt@20260713-010305
  - study: 01-gru-scaling-law
    wandb_group: rl-baseline-hattori@20260713-010255
reproduce: make -C studies/50-gru-dynamics-embedding r7
---

# Result 7 — do the findings hold for other seeds of model C? (PLAN.md phase G, G1 + G3)

**Question.** Everything in r1–r6 and r8 comes from one trained network, C seed 0. Study 01
trained the same configuration with seeds 1 and 2: same mice, same data and split, same code
and hyper-parameters, different initialisation and batch order. Which findings belong to the
trained *solution*, and which belong to seed 0 alone?

**How.**
- **Inputs.** Seeds 1 and 2 were fetched at their pinned digests (`source_runs.json`) and
  extracted like seed 0. Each reproduces its logged eval likelihood to < 1e-7 (0.7337670 and
  0.7337656; seed 0 0.7338507).
- **Reruns.** 35 core producers (B3, B7, C1–C4, D, E1, E2, E2b, F1–F3, F5–F7, F10, H2–H4, I1–I5,
  I7–I10) were rerun unchanged on each seed (`g_run_seed.sh`, `S12_MODEL_KEY=<key>`), writing to
  `analysis/seeds/<key>/`. All 70 runs completed.
- **G1** pulls the number behind each headline claim in SUMMARY.md and applies one fixed test
  per claim to every seed. The thresholds were set from seed 0's numbers before seeds 1–2 were
  seen. Only basis-free quantities are compared, because hidden units and embedding axes have
  no correspondence across seeds.
- **G3** compares the seeds directly, using the fact that the three trial tables are
  row-aligned (checked):
  - per-trial predictions;
  - canonical correlations of the hidden state;
  - cross-seed prediction of PCs;
  - v, trial by trial;
  - subject-embedding CCA and between-mouse distances.

## Summary

1. **The three seeds are essentially the same model, up to a change of basis (G3).**
   - **Prediction.** Per-trial logit(right) correlates 0.998 between every pair of seeds,
     against 0.970 between seed 0 and CTT. Mean |ΔP(right)| is 0.011 between seeds and 0.049
     against CTT; mean |Δ log-lik| is 0.023 vs 0.096 nats. Pooled likelihoods agree to 1e-4.
   - **Hidden state.** Canonical correlations of the top-10 PCs are 0.998 → 0.88 for the first
     nine components. Each seed's PCs 1–4 are predicted from another seed's 128-d state at
     held-out R² 0.96–0.999.
   - **Clusters.** The last input explains 66–67% of PC1–4 variance in every seed, and the last
     choice explains 77–85% of PC1. The four-cluster, two-lobe picture (B3c) is the same in all
     three (fig G3a).
   - **v** correlates 0.989–0.994 across seeds, trial by trial.
   - **Embedding.** The 4-d embeddings are linearly equivalent: canonical correlations
     0.985 / 0.96–0.98 / 0.96–0.97 / 0.84–0.91. The between-mouse distance structure agrees at
     Spearman 0.89–0.93.
2. **14 of 18 headline claims hold in all three seeds (G1).** These are:
   - the GRU beats all three RL models in 100% of mice;
   - its choice kernels match the animals' (gap 0.011–0.016, RL 0.07–0.20);
   - it is calibrated over 3-trial histories (0.18–0.24 pp; RL 1.1–2.2 pp);
   - a low-dimensional state (PC1 42–49%, participation ratio 3.6–4.3, 9 PCs to 90%);
   - attractors on the chosen side;
   - memory off the readout (readout variance share 9–10%);
   - a one-variable model of v (corr 0.979–0.989; likelihood 0.7175–0.7185);
   - the three-trace kernel for v (lowest BIC in 99.2–99.8% of mice);
   - a non-saturating learning rate (reward effect 1.005–1.028 logit, switch ≈ stay);
   - the delta rule after rewards in Q coordinates (a 0.39–0.43 ≈ −b);
   - the embedding decodes bias, gain and learning rate (rank R² 0.66–0.78, within 3% across
     seeds);
   - the embedding is a continuum (see 4);
   - bias is orthogonal to stickiness and learning rate, which sit 53° apart (cosine
     0.596–0.607; bias |cos| ≤ 0.23);
   - every mouse's context has all four attractors.
3. **Four claims fail in seed 1 only.** Three share one cause: seed 1 carries v less
   faithfully across lobe switches. The structure is the same; it is just less
   one-dimensional. The fourth is attractor reward depth.
   - **Carry-through.** Stay jumps keep R² 0.95–0.96, but switch jumps fall to 0.85–0.93
     (slopes 0.60–0.66, against 0.66–0.74 in seed 0). In every seed, switch jumps have lower
     slope and R² than stay jumps.
   - **Closure.** v-only R² falls to 0.84 after a rewarded switch (seed 0 0.93, seed 2 0.96).
   - **Ablation.** Removing v costs 0.026, more than the average random direction (0.017) but
     less than the worst of seed 1's 10 random directions (0.040). The 2-d lobe value axes
     still beat every random 2-d direction (0.039 vs at worst 0.024).
   - **Reward depth (D).** Rewarded attractors are deeper than unrewarded in seeds 0 and 2
     (+1.0 to +1.9 logit) but barely on the right side in seed 1 (+0.02). In two of its five
     contexts the unrewarded attractor is deeper. The per-mouse median falls from 1.07 to 0.36
     (seed 2 0.98).
   - **What this means.** "v is a single carried coordinate" holds in all seeds to within
     about 0.1 R². The stronger reading, "v alone closes the dynamics through switches", is
     seed 0's best case, not a property of the solution.
4. **The continuum verdict (F6) replicates.**
   - In every seed BIC prefers k = 2. The gain is within a heavy-tailed null, at +0.2, +2.1
     and −13.8 nats relative to its 95th percentile.
   - The two components sit 0.96–1.20 Mahalanobis apart, not separate types.
   - Seed 0's +0.2 was borderline, so this is the first independent check.
   - The Mantel test of embedding against behavioural distances gives ρ 0.47–0.49 in every
     seed.

## Seed-specific: corrections to earlier claims

- **The embedding PC dictionary (F5, F10) is seed-specific; the directions are not.**
  - **Seed 0.** PC3 = learning rate / gain and PC4 = bias.
  - **Seed 1.** Bias is still PC4, and learning rate is PC3 with the opposite sign.
  - **Seed 2.** Bias loads on PC1 and PC4, and learning rate on PC2. Its PC variance shares
    are 42 / 35 / 19 / 4%, against seed 0's 58 / 22 / 14 / 6%.
  - **What replicates.** The embeddings are linearly equivalent (CCA above). The angles
    between named behaviour directions in the whitened embedding replicate to ±0.05.
  - **Correction.** Statements such as "PC3 is the learning-rate axis" should read "a
    learning-rate direction (seed 0's PC3)".
- **Extra fixed points and bistability (D, F2) are seed 0 only.**
  - **Pooled context.** Seed 0 has three fixed points for the L-rewarded input; seeds 1 and 2
    have one.
  - **Per mouse.** Seed 0 has 79 bistable mice (78 for L+); seed 1 has 0 and seed 2 has 9.
  - **Correction.** The L+ multiplicity is a quirk of seed 0, not part of the computation.
- **Slow-mode timescales (D, F2) are not stable.**
  - The per-mouse median slowest τ is 355 trials in seed 0 and about 1,650 in seeds 1 and 2.
  - These modes have |λ| within about 0.003 of 1, so τ is poorly determined.
  - That these modes are slow and orthogonal to the readout holds; their exact timescales
    should not be quoted.
- **Second-order details vary by 10–35%.**
  - Seed 2's state is more concentrated (PC1 49% vs 42–43%).
  - The fast choice-trace λ_c is 0.51–0.65.
  - The v axis spreads over 32–45 effective units.
  - Seed 1's rewarded Q update is 0.39 against 0.43.
  - The qualitative readings hold; the second digit does not.

## By construction vs finding

- **By construction:**
  - The RL baselines' own numbers are identical across seeds: the same per-mouse fits are
    replayed on the same trials. They appear in the JSON, not in fig G1.
  - The trial rows are the same across seeds (same split).
  - v's unit is each seed's own TDR axis, so only its correlation across seeds is meaningful,
    not its scale.
- **Findings:**
  - Every cross-seed agreement above is a finding. The seeds were trained independently, and
    nothing in the analysis aligns them except G3's explicit linear maps.
  - The thresholds were fixed on seed 0, so the 14/18 is a replication, not a fit.

## Not covered yet

- **G2:** model B (H256) and its seeds.
- Seed reruns of B1, B2, B4–B6, C3, D1b–f, E2 (only E2b was compared), F4, F8, F9, F11–F13,
  H1, I6, I9b/c and I11. Their JSONs for seeds 1–2 were not produced, so they are untested.

<!-- BEGIN result-7 -->
[regenerated by `analysis/report_r7.py` — do not edit by hand]

### G1 claims (criteria fixed from seed 0's numbers before seeds 1–2 were seen)

| claim | C-s0 | C-s1 | C-s2 |
|---|---|---|---|
| GRU beats every RL model, in ≥ 99% of mice | ✓ | ✓ | ✓ |
| GRU's choice kernels match the animals' (< 0.03) and ≥ 3× better than any RL model | ✓ | ✓ | ✓ |
| GRU calibrated over 3-trial histories (< 0.5 pp; every RL model ≥ 3× worse) | ✓ | ✓ | ✓ |
| Low-dimensional state (PC1 > 35%, participation ratio < 6) | ✓ | ✓ | ✓ |
| Attractors on the chosen side; one per input for ≥ 3 of 4 inputs | ✓ | ✓ | ✓ |
| Rewarded attractors deeper than unrewarded (every context × side) | ✓ | ✗ | ✓ |
| Readout carries < 15% of the state variance (memory sits off the readout) | ✓ | ✓ | ✓ |
| v carried through every jump (slopes 0.6–0.85, R² > 0.9) | ✓ | ✗ | ✓ |
| A one-variable model of v tracks the GRU (corr > 0.97) | ✓ | ✓ | ✓ |
| v closes on itself (v-only R² > 0.9 in every transition group) | ✓ | ✗ | ✓ |
| Three-trace kernel wins by BIC in > 95% of mice | ✓ | ✓ | ✓ |
| Ablating v costs more than any random direction; the reward axis costs most | ✓ | ✗ | ✓ |
| Non-saturating learning rate (reward effect 0.8–1.2 logit; switch/stay 0.9–1.1) | ✓ | ✓ | ✓ |
| Delta rule after rewards in Q coordinates (\|a + b\| < 0.1), small update after losses (a < 0.15) | ✓ | ✓ | ✓ |
| Embedding decodes bias, reward gain and learning rate (rank R² > 0.5) | ✓ | ✓ | ✓ |
| Embedding is a continuum (k = 2 components < 1.5 SD apart) | ✓ | ✓ | ✓ |
| Bias is orthogonal to stickiness and learning rate (\|cos\| < 0.3); those two are not (cos > 0.4) | ✓ | ✓ | ✓ |
| Every mouse's context has all four attractors (> 99% of mice) | ✓ | ✓ | ✓ |

### G1 numbers

| metric | C-s0 | C-s1 | C-s2 | range / mean |
|---|---:|---:|---:|---:|
| eval likelihood (GRU) | 0.7339 | 0.7338 | 0.7338 | 0.0% |
| likelihood ctt | 0.721 | 0.721 | 0.721 | 0.0% |
| frac mice GRU > ctt | 1 | 1 | 1 | 0.0% |
| likelihood bari | 0.7181 | 0.7181 | 0.7181 | 0.0% |
| frac mice GRU > bari | 1 | 1 | 1 | 0.0% |
| likelihood hattori | 0.7159 | 0.7159 | 0.7159 | 0.0% |
| frac mice GRU > hattori | 1 | 1 | 1 | 0.0% |
| median per-mouse gain vs CTT | 0.01172 | 0.01172 | 0.01165 | 0.6% |
| kernel max \|gru − animal\| | 0.01091 | 0.01576 | 0.01484 | 35.1% |
| kernel max \|ctt − animal\| | 0.07492 | 0.07492 | 0.07492 | 0.0% |
| kernel max \|bari − animal\| | 0.2018 | 0.2018 | 0.2018 | 0.0% |
| kernel max \|hattori − animal\| | 0.1968 | 0.1968 | 0.1968 | 0.0% |
| 3-trial history calibration error pp (gru) | 0.1833 | 0.2372 | 0.1969 | 26.2% |
| 3-trial history calibration error pp (hattori) | 2.241 | 2.241 | 2.241 | 0.0% |
| 3-trial history calibration error pp (bari) | 2.127 | 2.127 | 2.127 | 0.0% |
| 3-trial history calibration error pp (ctt) | 1.098 | 1.098 | 1.098 | 0.0% |
| return after failed switch error (gru) | 0.004508 | 0.0059 | 0.005257 | 26.6% |
| return after failed switch error (hattori) | 0.09284 | 0.09284 | 0.09284 | 0.0% |
| return after failed switch error (bari) | 0.08261 | 0.08261 | 0.08261 | 0.0% |
| return after failed switch error (ctt) | 0.04862 | 0.04862 | 0.04862 | 0.0% |
| PC1 variance share | 0.4274 | 0.4242 | 0.4898 | 14.7% |
| PC2 variance share | 0.1802 | 0.1838 | 0.1565 | 15.7% |
| PC3 variance share | 0.07643 | 0.0912 | 0.06657 | 31.5% |
| PC4 variance share | 0.06797 | 0.0604 | 0.06209 | 11.9% |
| PC5 variance share | 0.05617 | 0.05307 | 0.05013 | 11.4% |
| participation ratio | 4.311 | 4.319 | 3.601 | 17.6% |
| PCs for 90% variance | 9 | 9 | 9 | 0.0% |
| readout variance share | 0.09868 | 0.1047 | 0.08908 | 16.0% |
| fixed points (R rewarded) | 1 | 1 | 1 | 0.0% |
| attractor on chosen side (R rewarded) | 1 | 1 | 1 | 0.0% |
| main attractor logit (R rewarded) | 4.344 | 3.647 | 4.496 | 20.4% |
| fixed points (R unrewarded) | 1 | 1 | 1 | 0.0% |
| attractor on chosen side (R unrewarded) | 1 | 1 | 1 | 0.0% |
| main attractor logit (R unrewarded) | 3.049 | 3.632 | 3.478 | 17.2% |
| fixed points (L rewarded) | 3 | 1 | 1 | 120.0% |
| attractor on chosen side (L rewarded) | 1 | 1 | 1 | 0.0% |
| main attractor logit (L rewarded) | -4.978 | -3.789 | -3.93 | 28.1% |
| fixed points (L unrewarded) | 1 | 1 | 1 | 0.0% |
| attractor on chosen side (L unrewarded) | 1 | 1 | 1 | 0.0% |
| main attractor logit (L unrewarded) | -3.045 | -3.181 | -3.086 | 4.4% |
| reward depth R (\|rew\| − \|unrew\| logit) | 1.295 | 0.0151 | 1.019 | 164.9% |
| reward depth L (\|rew\| − \|unrew\| logit) | 1.933 | 0.6082 | 0.8441 | 117.4% |
| reward depth > 0, share of context × side | 1 | 0.8 | 1 | 21.4% |
| v carry slope min | 0.6623 | 0.605 | 0.6918 | 13.3% |
| v carry slope max | 0.7975 | 0.7926 | 0.7851 | 1.6% |
| v carry R² min | 0.9346 | 0.849 | 0.9556 | 11.7% |
| v carry R² max | 0.983 | 0.9641 | 0.982 | 1.9% |
| 1-variable model corr(v) | 0.9847 | 0.9792 | 0.9886 | 1.0% |
| 1-variable model likelihood (per mouse) | 0.718 | 0.7185 | 0.7175 | 0.1% |
| v closure R² (v only), min over groups | 0.9332 | 0.8435 | 0.9553 | 12.3% |
| v perturbation retained at k=1 | 0.691 | 0.6726 | 0.7062 | 4.9% |
| v kernel held-out R² (free) | 0.9529 | 0.9386 | 0.9653 | 2.8% |
| three traces best BIC share (mice) | 0.9951 | 0.9984 | 0.9918 | 0.7% |
| share mice λ_unr > λ_rew (two outcomes) | 1 | 0.9984 | 1 | 0.2% |
| three traces median A_rew | 0.2916 | 0.3127 | 0.22 | 33.7% |
| three traces median lambda_rew | 0.7189 | 0.7417 | 0.7846 | 8.8% |
| three traces median A_unr | 0.0449 | 0.04589 | 0.05067 | 12.2% |
| three traces median lambda_unr | 0.999 | 0.999 | 0.9956 | 0.3% |
| three traces median A_c | 0.3482 | 0.3436 | 0.4759 | 34.0% |
| three traces median lambda_c | 0.651 | 0.5062 | 0.6131 | 24.5% |
| v axis participation ratio (units) | 31.85 | 44.9 | 39.49 | 33.7% |
| v retention eᵀJe (median over groups) | 0.6772 | 0.6609 | 0.7206 | 8.7% |
| ablation Δ value axis | -0.0275 | -0.02621 | -0.03217 | 20.8% |
| ablation Δ reward axis | -0.08123 | -0.08453 | -0.08944 | 9.6% |
| ablation Δ choice axis | -0.001997 | -0.003575 | -0.003161 | 54.2% |
| ablation Δ readout direction | -0.01929 | -0.01882 | -0.01771 | 8.5% |
| ablation Δ random 1-d (mean) | -0.01336 | -0.01686 | -0.01369 | 23.9% |
| ablation Δ random 1-d (worst) | -0.02061 | -0.0403 | -0.02325 | 70.2% |
| reward effect on logit (GRU) | 1.005 | 1.028 | 1.012 | 2.2% |
| reward effect switch / stay (GRU) | 1.018 | 1.008 | 1.002 | 1.6% |
| Q decoding R² (min of Q_L, Q_R) | 0.9393 | 0.9422 | 0.9417 | 0.3% |
| Q_chosen update a (R rewarded) | 0.434 | 0.3869 | 0.429 | 11.3% |
| Q_chosen update b (R rewarded) | -0.4335 | -0.3719 | -0.4209 | 15.1% |
| Q_chosen update a (R unrewarded) | 0.05075 | 0.04481 | 0.03817 | 28.2% |
| Q_chosen update b (R unrewarded) | -0.1293 | -0.1263 | -0.119 | 8.2% |
| mice with all four attractors | 1 | 1 | 1 | 0.0% |
| mice bistable for any input | 79 | 0 | 9 | 269.3% |
| per-mouse reward depth (median) | 1.073 | 0.3581 | 0.9757 | 89.1% |
| per-mouse slowest τ (median, trials) | 355.4 | 1670 | 1635 | 107.7% |
| embedding decodes attractor depth (rank R²) | 0.8476 | 0.889 | 0.88 | 4.7% |
| embedding decodes v retention (e_vᵀJe_v) (rank R²) | 0.7971 | 0.8252 | 0.7084 | 15.0% |
| embedding PC1 variance share | 0.576 | 0.524 | 0.4177 | 31.3% |
| embedding PC2 variance share | 0.2203 | 0.2314 | 0.3528 | 49.4% |
| embedding PC3 variance share | 0.1403 | 0.2223 | 0.1939 | 44.2% |
| embedding PC4 variance share | 0.06351 | 0.0223 | 0.03557 | 101.8% |
| embedding decodes gru bias (rank R²) | 0.7816 | 0.774 | 0.7682 | 1.7% |
| embedding decodes gru reward gain (1 back) (rank R²) | 0.7795 | 0.7562 | 0.7802 | 3.1% |
| embedding decodes ctt learn_rate (rank R²) | 0.6733 | 0.6755 | 0.6599 | 2.3% |
| embedding decodes hattori softmax_inverse_temperature (rank R²) | 0.4695 | 0.4791 | 0.4782 | 2.0% |
| mixture best k (BIC) | 2 | 2 | 2 | 0.0% |
| mixture BIC gain − heavy-tail null 95% | 0.1891 | 2.074 | -13.82 | 412.6% |
| mixture k=2 separation (Mahalanobis) | 0.9716 | 0.9595 | 1.202 | 23.2% |
| Mantel ρ embedding vs animal behaviour | 0.4899 | 0.4706 | 0.4869 | 4.0% |
| embedding explains animal determinism (R²) | 0.7437 | 0.7665 | 0.7475 | 3.0% |
| embedding explains animal switch rate (R²) | 0.7217 | 0.7441 | 0.7343 | 3.0% |
| embedding explains animal bias (R²) | 0.6546 | 0.6444 | 0.6217 | 5.1% |
| cos(determinism, ctt learn_rate) in embedding | 0.6065 | 0.5964 | 0.6052 | 1.7% |
| cos(determinism, bias) in embedding | -0.1123 | -0.08537 | -0.1364 | 45.8% |
| cos(ctt learn_rate, bias) in embedding | 0.1845 | 0.2312 | 0.2142 | 22.2% |
| cos(determinism, reward gain (1 back)) in embedding | 0.836 | 0.8338 | 0.8023 | 4.1% |

### G3 cross-seed alignment

| pair | logit r | mean \|ΔP(right)\| | mean \|Δ log-lik\| (nats) | v r | state CCA 1–4 | state CCA 5–8 | PCs 1–4 R² from other state | embedding CCA | embedding distance ρ |
|---|---:|---:|---:|---:|---|---|---|---|---:|
| C-s0 vs C-s1 | 0.9978 | 0.0114 | 0.0228 | 0.989 | 0.998, 0.996, 0.987, 0.981 | 0.961, 0.956, 0.953, 0.929 | 0.999, 0.995, 0.975, 0.984 | 0.985, 0.963, 0.960, 0.910 | 0.921 |
| C-s0 vs C-s2 | 0.9978 | 0.0118 | 0.0235 | 0.994 | 0.999, 0.997, 0.985, 0.980 | 0.972, 0.963, 0.949, 0.916 | 0.999, 0.993, 0.975, 0.987 | 0.984, 0.969, 0.959, 0.839 | 0.892 |
| C-s1 vs C-s2 | 0.9980 | 0.0113 | 0.0226 | 0.991 | 0.998, 0.996, 0.989, 0.982 | 0.970, 0.966, 0.945, 0.927 | 0.998, 0.993, 0.977, 0.987 | 0.985, 0.978, 0.971, 0.900 | 0.930 |
| C-s0 vs CTT (scale reference) | 0.9699 | 0.0489 | 0.0963 | | | | | | |

| seed | η² of PC1–4 by last input | η² of PC1–4 by last choice | η² of PC1 by last choice |
|---|---:|---:|---:|
| C-s0 | 0.663 | 0.469 | 0.807 |
| C-s1 | 0.673 | 0.483 | 0.848 |
| C-s2 | 0.673 | 0.503 | 0.774 |

![G1](../fig_G1_seed_robustness.png)
![G3a](../fig_G3a_state_space_by_seed.png)
![G3b](../fig_G3b_cross_seed_alignment.png)
<!-- END result-7 -->
