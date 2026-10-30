# Study 51 findings: a GRU that can predict ignored trials

We analysed a GRU trained to predict left, right **and** ignore (**C3**) the way Study 50
analysed its two-action twin (**C**), asking how the two models differ and what an ignored
trial does to the network's state, dynamics and predicted behaviour.

- **Model:** C3 = Study 02 `nxd-3way`, H128, D = 614 mice, seeds 0–2 (seed 0 unless stated)
- **Twin:** C = Study 50 model C (2-way)
- **Data:** 4.76M eval timesteps, 8.9% ignored
- **Status:** drafts r1–r7 in this folder, as of 26 Oct 2026; uncommitted

Figures are in `../` (the `analysis/` folder). Every number comes from the JSON file next to
each figure; the detailed reports are [r1](analysis/reports/r1-first-look.md)–[r7](analysis/reports/r7-robustness.md)
(index: [INDEX.md](analysis/reports/INDEX.md)).

## Contents

1. [Summary](#summary)
2. [Models and terms](#models-and-terms)
3. [Setup and data](#setup-and-data)
4. [What the state looks like](#what-the-state-looks-like)
5. [How inputs move the state](#how-inputs-move-the-state)
6. [Predicting engagement](#predicting-engagement)
7. [Side choice: C3 vs C, the animals and RL](#side-choice-c3-vs-c-the-animals-and-rl)
8. [Dynamics and attractors](#dynamics-and-attractors)
9. [The carried value](#the-carried-value)
10. [Separable subspaces](#separable-subspaces)
11. [Engagement in the subject embedding](#engagement-in-the-subject-embedding)
12. [Robustness across seeds](#robustness-across-seeds)
13. [Corrections and caveats](#corrections-and-caveats)
14. [Open questions](#open-questions)

---

## Summary

1. **Adding the ignore action leaves the side-choice machinery intact.** C3's left/right
   predictions, update rules, reward kernel, state clusters and attractors match C's on the
   same trials (reward-flip effect +1.04 vs +1.00; update maps within ~0.17 log-odds).
2. **Engagement is a mostly separate variable.** Its readout direction is near-orthogonal to
   the side direction (cosine 0.10), it is carried by different units (49 side units vs 34
   engagement units, 5 shared), and removing it leaves side prediction intact (cost 0.010,
   the same as a random direction).
3. **An ignore gives the network a fifth state cluster and its own attractor.** Repeated
   ignores drive every state to one stable point that is disengaged (P(ignore) 0.97–0.98)
   and has near-zero side preference.
4. **Animals forget their side preference over an ignore bout, and C3 has learned the rate;
   C cannot.** P(repeat last choice) falls from 0.88 with no gap to 0.59 after 11+ ignores;
   C3 tracks this within 0.01, C stays near 0.82. The animals' history kernel shrinks to 0.64
   of normal after one ignore and 0.23 after 6+; C3 gives 0.63 and 0.24.
5. **So C3's whole advantage over C on side choice comes from trials after ignores.**
   Overall 0.7351 vs 0.7338; on trials right after an engaged trial C3 is marginally worse,
   and the advantage grows from +0.03 nats per trial after one ignore to +0.25 after 11+, in
   85–98% of mice.
6. **The value is carried through ignores with one leak rate.** Causally nudging the state
   along the TDR value axis, about 0.67 of the nudge survives one step for every input,
   ignores included. An ignore is an ordinary leaky step that adds no new evidence and drops
   the readout's last-choice push.
7. **C3's subject embedding carries each mouse's engagement style; C's carries none.** A
   mouse's ignore rate decodes from C3's embedding with R² 0.47–0.68 across seeds, from C's
   with 0.01. Mice differ in how fast one ignore disengages them, not in where they end up.
8. **Everything above holds across C3 seeds 0–2**, mostly to two or three decimals.

---

## Models and terms

| Term | Meaning |
|---|---|
| **C3** | The 3-way GRU (left / right / ignore). Same configuration as C except `ignore_policy=include` and the wrapper commit; the code changes in between do not affect GRU training or the data splits. |
| **C** | Study 50's 2-way GRU. It never sees ignored trials: its sequence skips them. |
| **side readout** | `logit_diff` = logit_R − logit_L = log P(R \| engaged) / P(L \| engaged). Comparable to C's logit(right). |
| **engage readout** | log P(engaged) / P(ignore), with P(engaged) = P(L) + P(R). |
| **gap** | The number of ignored trials immediately before an engaged trial. |
| **shared sessions** | The 8,846 eval sessions both models evaluate. They hold identical engaged trials (4,314,340; same trial numbers, choices and rewards). All C3-vs-C comparisons use them. |
| **step** | A GRU timestep. For C3 an ignored trial is a step; for C it is not. |
| **teacher-forced** | The model sees the animal's real history (the normal evaluation mode), as opposed to a free-running rollout. |

Each analysis separates **what holds by construction** from **what is a finding**. For
example, C's flat curves after a gap are expected, because it cannot see the gap; the
finding is that the animals depend on the gap, and by how much.

---

## Setup and data

### Source models (`source_runs.json`)

- **How:** Searched every W&B project for 3-way runs (`ignore_policy=include`, or
  `output_size=3` without it). Pinned the D = 614 / H128 cell, three seeds, with immutable
  artifact digests, and diffed its configuration field by field against C.
- **Finding:** The only 3-way GRUs are Study 02's `nxd-3way` grid. The H128/D614 cell
  differs from C only in `ignore_policy` and the wrapper commit (5 Jul vs 5 Sep). The wrapper
  changes in between touch logging and add unused features. Held-out L/R likelihood on
  engaged trials is 0.7301–0.7304.

### A1 trial table (`extract_trial_states.py`)

- **How:** Extended the wrapper's per-trial extractor so ignored trials stay in the table at
  their GRU timestep (`action = 2`, with P(ignore)). One forward pass over the eval split.
  Checked that the table reproduces all three logged likelihoods, and that the change leaves
  Study 50's 2-way table unchanged.
- **Finding:** 4,761,751 timesteps (423,024 ignored), 8,906 sessions, 614 mice. Recomputed
  vs logged likelihood: 3-way 0.642176 vs 0.642175; L/R on engaged 0.735084 vs 0.735084;
  engagement 0.850046 vs 0.850045. Study 50's C table reproduced exactly (same rows and
  columns; floats within GPU-vs-CPU error).

### Aligning the two models' eval sets (`split_alignment.py`)

- **How:** Compared the sessions and trials in C's and C3's eval sets, then traced the
  difference to the loader code.
- **Finding:** The 2-way loader drops sessions where every trial was ignored before it
  alternates sessions into train and eval. For five mice that flips which later sessions
  land in eval: 57 sessions are only in C's set and 56 only in C3's. The other 8,846 sessions
  match exactly (4,314,340 engaged trials, zero mismatches), and every comparison uses them.

---

## What the state looks like

### Example sessions (B1)

- **How:** Study 50's four example sessions, one per task, with C3 drawn above C on the same
  session. The x-axis is the original trial number, so ignored trials line up across panels.
- **Finding:** C3's P(R | engaged) and C's P(R) are nearly the same curves; per-session L/R
  likelihood differs by at most 0.004. C3's P(ignore) follows the animal's ignore bouts, most
  visibly in the Coupled Without Baiting session (77 ignores).

| Task | Trials | Ignored | L/R lik. C3 | L/R lik. C | Engage lik. C3 |
|---|---:|---:|---:|---:|---:|
| Uncoupled Baiting | 480 | 1 | 0.720 | 0.720 | 0.986 |
| Coupled Baiting | 568 | 13 | 0.552 | 0.555 | 0.927 |
| Uncoupled Without Baiting | 486 | 3 | 0.871 | 0.870 | 0.970 |
| Coupled Without Baiting | 646 | 77 | 0.821 | 0.819 | 0.729 |

![B1](analysis/fig_B1_example_sessions.png)

### Units (B2)

- **How:** Correlated every unit with each readout, over the example session and over 200k
  pooled steps. Each unit is assigned to the readout it follows more closely.
- **Finding:** The units largely specialise: 49 correlate strongly (|r| > 0.5) with the side
  readout, 34 with engagement, only 5 with both. The two readouts are uncorrelated across
  steps (r = −0.01). The side units switch with the mouse's blocks; several engagement units
  drift slowly over the session.

![B2](analysis/fig_B2_unit_heatmap.png)

### Dimensionality and readout geometry (B3a, B3b)

- **How:** PCA of the 128-d state over all timesteps. Then found where each readout direction
  lies: the share of its variance on each PC, and the cosine between the side direction
  (w_R − w_L) and the linear part of the engagement readout.
- **Finding:** C3 is slightly higher-dimensional than C. Side and engagement occupy different
  PCs and are near-orthogonal.

| | C3 | C |
|---|---:|---:|
| Participation ratio | 4.70 | 4.31 |
| PCs for 90 / 95 / 99% variance | 11 / 18 / 53 | 9 / 14 / 39 |
| Variance of PC1, PC2, PC3 (%) | 41.3, 14.0, 10.8 | 42.7, 18.0, 7.6 |
| Side readout variance on PC1 / PC2 | 61% / 25% | — |
| Engage readout variance on PC1 / PC2 | 21% / 63% | — |
| Cosine, side vs engage direction | 0.10 | — |

![B3a](analysis/fig_B3a_pca_variance.png)
![B3b](analysis/fig_B3b_state_space_readouts.png)

### A fifth state cluster (B3c)

- **How:** Coloured the states in PC1–PC2 by the input just received (L+, L−, R+, R−, ignore).
- **Finding:** C has four clusters. C3 adds a separate fifth cluster after an ignore, at high
  PC1 and low PC2, where P(ignore) is highest (mean 0.57, vs 0.03–0.06 after engaged trials).
  Inside it, P(R | engaged) still spans 0–1, so the side belief survives an ignore.

![B3c](analysis/fig_B3c_prev_input.png)

### What the state encodes (B7)

- **How:** Ridge decoders from all 128 units, 5-fold cross-validated with folds grouped by
  session, on 400k steps. For history lags, a unique R² beyond all shorter lags. Choice is
  coded +1 / −1 / 0 for R / L / ignore.
- **Finding:** Ignore history is encoded as well as choice and reward history: ignored 1
  back R² 0.98, ignored 2 back unique R² 0.95. The recent ignore rate (R² 0.94), trials since
  the last ignore (0.85) and trial position (0.83, almost entirely outside PCs 1–4) are also
  decodable. Deeper choice and reward lags decode slightly worse than in C (reward 4 back 0.50
  vs 0.60), but C3's lags count ignore steps, so the comparison is not exact.

![B7](analysis/fig_B7_decoders.png)

---

## How inputs move the state

### Update maps (B4)

- **How:** Each readout at step t+1 plotted against step t, separately for each input
  received at t, for C3's side readout, C's side readout and C3's engagement readout, on the
  shared sessions.
- **Finding:**
  - After engaged inputs, C3's side updates match C's to within about 0.17 log-odds.
  - An ignore shrinks the side readout about 30% toward neutral, symmetrically: −3 → −2.15,
    +3 → +2.11, and 0 stays at −0.08.
  - On engagement, a reward pushes toward engaging (from 0 to about +1.0, vs +0.2 after an
    unrewarded choice) and does so equally for either side. An ignore pushes hard toward
    disengaging (from 0 to −0.52; from +5 to +2.45).

![B4](analysis/fig_B4_update_maps.png)

### How long one input's effect lasts (B5)

- **How:** Matched effects, as in Study 50: the difference in Δreadout(t → t+k) between two
  input groups, within bins of the readout at t, which removes regression to the mean. Also
  P(ignore) and P(R | engaged) on the next 10 steps after each input, model vs animal.
- **Finding:**
  - The reward effect on side is the same in both models: C3 +0.89, C +0.86 one step later,
    both about 0.8 for six steps.
  - A reward's effect on engagement is brief: +0.7 one step later, +0.25 after six.
  - An ignore's effect on engagement lasts: −1.06 vs an unrewarded choice one step later,
    still −0.58 after ten.
  - The model's P(ignore) after each input matches the animals' observed rate to within 0.003
    at every step (after an ignore: 0.570 vs 0.571, 0.531 vs 0.532, …).

![B5](analysis/fig_B5_outcome_triggered.png)

---

## Predicting engagement

### Calibration and bout dynamics (B8)

- **How:** Teacher-forced. Compared predicted P(ignore) with the observed ignore rate in six
  ways: by prediction bin, after k consecutive ignores (the hazard of staying disengaged),
  around bout onset and end, by trial position, and by trials since the last reward.
- **Finding:**
  - Calibration is close: the observed rate is within 0.005 of the prediction in every bin.
    The overall rate is 0.0888, the mean prediction 0.0889.
  - The bout-exit hazard is matched at every run length, within 0.034.
  - The ignore rate rises through the session (0.045 → 0.20 by trial 700) and with trials
    since reward; the model tracks both.
  - Ignore bouts average 2.16 trials; 63% are single trials, the 99th percentile is 14.

| Ignores just before (k) | 0 | 1 | 2 | 3 | 5 | 10 | 15+ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Observed P(ignore) | 0.042 | 0.376 | 0.575 | 0.674 | 0.756 | 0.812 | 0.922 |
| C3 predicted | 0.042 | 0.376 | 0.582 | 0.647 | 0.764 | 0.846 | 0.915 |

*By construction:* in the onset and end panels, the trial before an onset and the trial at an
end are engaged by definition. The reward-history panel counts ignores as unrewarded, so part
of its rise is bout continuation.

![B8](analysis/fig_B8_engagement.png)

### Tracking and reward richness (B6)

- **How:** Choice against the true reward-probability difference, per task, for C3 (engaged
  trials) and C. Then P(ignore) against reward richness, P(rew | R) + P(rew | L).
- **Finding:** Both models track the reward probabilities about equally well: the largest
  steady-state gap to the animals is 0.01–0.06 per task for each. The ignore rate has no
  consistent relation to richness. Each probability set belongs to particular training eras,
  so richness is confounded, and no richness effect is claimed. The model matches the
  observed rate in each bin.

![B6](analysis/fig_B6_tracking.png)

---

## Side choice: C3 vs C, the animals and RL

### Likelihood by gap (C2)

- **How:** Scored the animal's left/right choice under C3 (P(R | engaged)), C and three RL
  models from Study 01 (CTT, Bari, Hattori), on the shared engaged trials, overall, per task,
  per mouse and by gap.
- **Finding:** C3 beats C overall (0.7351 vs 0.7338) and in every task, in 71% of mice; both
  GRUs beat every RL model in 100% of mice. **All of C3's gain is after ignores.** With no gap
  C3 is marginally worse (−0.0005 nats per trial). After long bouts, C and CTT fall below
  chance.

| Gap | Trials | C3 | C | CTT | C3 − C per trial (nats) | Mice with C3 > C |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 4,135,637 | 0.742 | 0.742 | 0.729 | −0.0005 | 34% |
| 1 | 113,192 | 0.614 | 0.597 | 0.586 | **+0.033** | 85% |
| 2 | 28,830 | 0.574 | 0.535 | 0.525 | **+0.079** | 85% |
| 3–5 | 23,981 | 0.552 | 0.496 | 0.487 | **+0.104** | 87% |
| 6–10 | 8,999 | 0.528 | 0.440 | 0.434 | **+0.182** | 91% |
| 11+ | 3,701 | 0.516 | 0.406 | 0.408 | **+0.251** | 98% |

![C2](analysis/fig_C2_likelihood.png)

### Behaviour after a gap (C4)

- **How:** For each engaged trial: did the animal repeat its previous engaged choice? Split by
  the gap in between and by whether that previous choice was rewarded. A model's P(stay) is
  its probability of the previous side.
- **Finding:** Animals forget their side preference over an ignore bout, and C3 has learned
  the rate, after rewarded and unrewarded choices alike. C and CTT cannot see the gap and keep
  predicting stays near 0.82. C's confidence |P − 0.5| stays near 0.32; C3's falls from 0.38
  to 0.11.

| Gap | Animal | C3 | C | CTT |
|---|---:|---:|---:|---:|
| 0 | 0.879 | 0.878 | 0.874 | 0.871 |
| 1 | 0.769 | 0.766 | 0.838 | 0.840 |
| 2 | 0.709 | 0.716 | 0.828 | 0.831 |
| 3–5 | 0.669 | 0.669 | 0.817 | 0.821 |
| 6–10 | 0.613 | 0.620 | 0.812 | 0.816 |
| 11+ | 0.593 | 0.583 | 0.813 | 0.818 |

![C4](analysis/fig_C4_after_ignore.png)

### History kernels by gap (C1b)

- **How:** Su2022 logistic design (rewarded and unrewarded choice, 10 engaged trials back),
  fitted to the animals' choices and, as soft labels, to C3, C and CTT, separately for each
  gap group.
- **Finding:** With no gap the kernels agree (rewarded lag 1: animal 1.85, C3 1.85, C 1.80).
  After a gap the animals' kernel shrinks, and C3's shrinks with it; C's and CTT's do not.

| Lag-1 weight (rewarded / unrewarded), × no gap | gap 1 | 2 | 3–5 | 6+ |
|---|---:|---:|---:|---:|
| Animal | 0.64 / 0.58 | 0.45 / 0.36 | 0.38 / 0.33 | 0.23 / 0.23 |
| C3 | 0.63 / 0.56 | 0.49 / 0.41 | 0.40 / 0.35 | 0.24 / 0.21 |
| C | 0.98 / 1.00 | 0.96 / 0.99 | 0.95 / 0.99 | 0.96 / 1.02 |
| CTT | 1.01 / 1.00 | 1.00 / 0.99 | 1.00 / 0.99 | 0.98 / 1.03 |

![C1 kernels](analysis/fig_C1_kernels.png)

### Counterfactual edits of one input (C1a)

- **How:** Re-ran C3 with one trial's input edited and every later input as observed, 20,000
  edits of each kind: a reward flip; an engaged trial turned into an ignore; an ignored trial
  turned into an unrewarded choice. The wrapper's counterfactual code was extended to 3-way
  models and runs at full float32 precision, since on GPU the default TF32 made edited and
  baseline runs differ by about 5e-4 even with no edit.
- **Finding:**
  - The causal reward kernel on side is unchanged: C3 +1.04 → +0.62 → +0.35,
    C +1.00 → +0.60 → +0.35.
  - Turning an engaged trial into an ignore lowers engagement by 2.27 log-odds, with 0.89
    (39%) still there ten steps later; a reward's effect on engagement fades to 6%.
  - On side, that edit removes the trial's evidence: −1.24, or −1.84 if the trial had been
    rewarded.
  - The reverse edit raises both readouts by about +1, with a shorter-lived effect on
    engagement. The two edits start from different states, so this is not a matched
    comparison.

![C1 impulse](analysis/fig_C1_impulse.png)

---

## Dynamics and attractors

### Flow fields and fixed points (D1, D2)

- **How:** The free-context step function reproduces C3's stored readouts to 4e-7. For each
  constant input, at the mean context: 2,000 real starting states, candidates from Adam
  minimisation of ½‖F(h) − h‖² and from forward iteration, each refined by damped Newton in
  float64 and accepted at q < 1e-12. Stability and time constants come from the Jacobian.
- **Finding:** Each input has exactly one stable fixed point. The engaged inputs' attractors
  sit on their side, as in C. The ignore input's attractor is new: side-neutral and almost
  fully disengaged. The slow modes are very slow and barely touch either readout.

| Constant input | Side logit at attractor (C3) | C | P(ignore) | Slowest τ (steps) |
|---|---:|---:|---:|---:|
| R+ | +3.86 | +4.34 | 0.078 | 3,652 |
| R− | +4.86 | +3.05 | 0.050 | 555 |
| L+ | −4.83 | −4.98 (bistable) | 0.019 | 2,347 |
| L− | −3.65 | −3.05 | 0.018 | 248 |
| ignore | **+0.04** | — | **0.978** | 566 |

Attractor depths vary by up to about 1.5 log-odds across C3 seeds (see
[Robustness](#robustness-across-seeds)), so no depth difference between C3 and C should be
read from this table.

![D1](analysis/fig_D1_flow_fields.png)
![D2](analysis/fig_D2_fixed_points.png)

### Relaxation under repeated ignores (D3)

- **How:** From 5,000 real engaged states, each with its own session context: fed 30 ignore
  inputs, then unrewarded choices of the pre-bout side, tracking both readouts.
- **Finding:**
  - The side readout keeps 0.61 / 0.45 / 0.36 of its size after 1 / 2 / 3 ignores (0.10
    after 30). This matches the animals' kernel shrinkage (0.64 / 0.45 / 0.38) without having
    been fitted to it.
  - Disengagement is fast: P(ignore) goes 0.05 → 0.15 → 0.31 → 0.47 → 0.62 over four ignores.
  - Re-engagement is slower: P(ignore) drops to 0.61, 0.37 and 0.30 over three choices, but
    only to 0.18 after ten. The side preference rebuilds in about three choices.

![D3](analysis/fig_D3_relaxation.png)

---

## The carried value

Study 50 found that C carries one value coordinate from trial to trial through all four
input clusters. These analyses ask whether that holds through ignored trials, and how much of
C3's ignore handling a simple rule captures.

### Encoding axes per cluster (I1)

- **How:** Within each last-input cluster, the direction along which the state moves as a
  readout changes, u_c = cov_c(h, v) / var_c(v), compared by cosine across clusters, for C3
  and C.
- **Finding:** C's two-lobe side structure is kept: the left clusters share one axis (cosine
  0.95), the right clusters another (0.96), nearly orthogonal to each other (0.09–0.15; C:
  0.91 / 0.92 within, about 0.2 across). The post-ignore cluster uses a blend of both
  (0.52–0.64). Engagement shares one axis across the engaged clusters (0.84–0.97).

![I1](analysis/fig_I1_cluster_axes.png)

### TDR with ignored trials (I2)

- **How:** Study 50's targeted dimensionality reduction on 1M steps: regress the state on
  previous choice and reward, their product, value (the standardised side readout) and
  value × choice, plus two new terms, "previous step ignored" and the engagement readout.
  Nuisance terms: session context and trial position. Axes orthogonalised with value first.
  Also refitted Study 50's exact design on steps after engaged trials only.
- **Finding:** The value axis is the same with or without ignored trials in the fit (cosine
  0.997). Its cosine with the side readout is 0.48 (C: 0.50), and it contains no engagement
  component (0.00). The lobe axes have cosine 0.28 (C: 0.40). The raw "ignored" axis aligns
  with the raw "choice" axis (0.81), consistent with the post-ignore cluster sitting next to
  the right-choice states.

![I2](analysis/fig_I2_tdr.png)

### Is the value coordinate carried? Regression (I3)

- **How:** Regressed value(t+1) on value(t) along the TDR value axis, per input received and
  per source (left lobe, right lobe, post-ignore). Study 50's criterion for "carried" is a
  tight linear map. High R² means carried rather than reset; a slope below 1 means leaky
  carrying. Also fitted side readout ≈ a·v + b within each cluster.
- **Finding:** The value is carried through ignores as tightly as through engaged inputs, and
  carried on when the mouse re-engages. What an ignore changes is the readout's offset: the
  post-ignore cluster's offset puts readout neutrality (v = +0.85) at the cluster's mean value
  (+0.92), so the last-choice push disappears while v itself carries over.

| Transition | Slope | R² |
|---|---:|---:|
| Ignore, from right lobe | 0.79 | 0.97 |
| Ignore, from left lobe | 0.62 | 0.95 |
| Ignore, already post-ignore | 0.84 | 0.99 |
| Engaged inputs, C3 | 0.58–0.79 | 0.90–0.96 |
| Engaged inputs, C (Study 50) | 0.66–0.80 | 0.93–0.98 |
| Engaged inputs arriving after an ignore | 0.81–0.86 | 0.96–0.99 |

![I3](analysis/fig_I3_tdr_dynamics.png)

### Is the value coordinate carried? Causal test (I3c)

- **How:** For 20,000 real steps, 4,000 per input, nudged the state ±0.5 along the value axis
  and re-ran the GRU for 10 steps on the observed inputs with the session's own context. The
  unedited rollout reproduces the stored states to 4e-8. Controls: the same nudge along 5
  random directions orthogonal to the value axis, and along the engagement axis.
- **Finding:**
  - **One leak rate for every input.** 0.61–0.69 of the nudge is still on the value axis one
    step later in every input × source cell, ignores included (0.65–0.69), with an SD of
    0.02–0.04 within each cell. The regression slopes in I3 spread wider because they also
    pick up components that move with v.
  - **Fast, then slow decay:** 0.68, 0.51, 0.42, 0.36 over the first four steps, 0.19 after
    ten. That is slower than the product of I3 slopes predicts after about four steps (0.07
    after ten).
  - **The axis is specific:** random orthogonal nudges barely feed v (−0.09 to +0.17), and an
    engagement nudge does not at all (|Δv| ≤ 0.03), though it moves engagement strongly
    (+1.03, still +0.80 four steps later).
  - **The value causally drives side choice:** the side readout shifts by 1.48 per unit, then
    0.78, 0.53, 0.42. Engagement barely changes (−0.04 to −0.06).

![I3c](analysis/fig_I3c_value_causal.png)

### A two-variable distilled model (I4)

- **How:** The teachers are C3's own readouts: side v and engagement e. Fitted an affine
  update x(t+1) = A x(t) + c per input group (choice × reward × stay / switch / first, plus
  ignore), then ran the model from the animals' inputs alone, without the GRU, and scored it.
  Variants: side only with the ignore rule; side only with ignores skipped, which is C's
  implicit rule; and the same model distilled from C.
- **Finding:** **One rule, v ← 0.71·v per ignore, recovers 83% of C3's post-ignore advantage
  over C** (after 6+ ignores: 0.505 with the rule vs 0.426 without; GRUs 0.525 vs 0.430). The
  engagement variable captures 87% of C3's engagement gain over a constant base rate (0.836 vs
  GRU 0.850 and base 0.741). The two variables do not interact at ignore steps (cross terms
  ≤ 0.01). Distillation costs about 0.02 everywhere, as it did for C in Study 50.

| Model | L/R overall | Gap 0 | Gap 3–5 | Gap 6+ |
|---|---:|---:|---:|---:|
| C3 (GRU) | 0.7351 | 0.742 | 0.552 | 0.525 |
| Distilled 2-var (v, e) | 0.7176 | 0.725 | 0.532 | 0.506 |
| Distilled v, with ignore rule | 0.7137 | 0.721 | 0.528 | 0.505 |
| Distilled v, ignores skipped | 0.7117 | 0.720 | 0.482 | 0.426 |
| C (GRU) | 0.7338 | 0.742 | 0.496 | 0.430 |
| C distilled v | 0.7117 | 0.720 | 0.483 | 0.427 |

![I4](analysis/fig_I4_distilled_model.png)

---

## Separable subspaces

### Mean-replacement ablations (E2)

- **How:** On 1,500 random eval sessions, clamped the state's component along a direction to
  its mean at every step, either in what is read out and carried, or only in what is carried
  forward. Directions: side, engagement, both, and 5 random directions as controls. Scored L/R
  and engagement likelihood against the intact model (L/R 0.7391, engagement 0.8484).
- **Finding:** The dependence is one-way. Clamping engagement costs side prediction only
  0.010, within the random-direction range, while engagement loses 0.139 (0.100 when only
  memory is clamped). Clamping the side direction costs L/R 0.239 (chance, by construction),
  or 0.118 memory-only, and also costs engagement 0.04–0.05, about 8 times a random direction.
  One possibility, untested: a neutral side value resembles the disengaged state.

| Clamped | L/R lost (full / memory) | Engagement lost (full / memory) |
|---|---:|---:|
| Side direction | 0.239 / 0.118 | 0.049 / 0.042 |
| Engagement direction | 0.010 / 0.010 | 0.139 / 0.100 |
| Random 1-d (range of 5) | 0.001–0.019 | 0.001–0.006 |

![E2](analysis/fig_E2_ablations.png)

---

## Engagement in the subject embedding

### Decoding and per-mouse dynamics (F)

- **How:** Per-mouse engagement features from the data: ignore rate, bout length,
  P(staying disengaged) and the rise over a session. Decoded each from C3's and C's 4-d
  subject embeddings with 5-fold cross-validated linear regression. Then ran C3's ignore
  dynamics at each mouse's own mean context.
- **Finding:**
  - C3's embedding carries engagement style; C's carries none. Ignore rate decodes with R²
    0.68 from C3 seed 0 (0.47–0.68 across seeds) vs 0.01 from C; the rise over a session with
    0.45 vs −0.01.
  - Mice differ in how quickly one ignore disengages them, not in where they end up. Every
    mouse's ignore attractor sits at P(ignore) 0.97–0.98. P(ignore) after one ignore ranges
    0.03–0.58 across mice and correlates r = 0.90 with the mouse's observed ignore rate.

*By construction:* C3 was trained to fit each mouse's ignores, so some engagement information
in its embedding is expected. The findings are that C's has none, and that C3 carries it
mainly through how fast a mouse enters disengagement.

![F](analysis/fig_F_engagement_embedding.png)

---

## Robustness across seeds

### C3 seeds 0–2 (G1)

- **How:** Extracted seeds 1 and 2 (each reproduces its logged likelihoods within 6e-8) and
  recomputed the headline numbers and dynamics for all three seeds, against C seed 0. Study 50
  found C's seeds within 0.0001 held-out likelihood of each other.
- **Finding:** Every headline result holds, mostly to two or three decimals. Seed-specific:
  how well the embedding decodes ignore rate, and the depths of the engaged attractors.

| Metric | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| L/R likelihood, C3 (C = 0.7338) | 0.7351 | 0.7350 | 0.7345 |
| C3 − C per trial, gap 0 | −0.0007 | −0.0008 | −0.0014 |
| C3 − C per trial, gap 6+ | +0.199 | +0.197 | +0.195 |
| Max \|C3 − animal\| P(stay) over gaps | 0.006 | 0.006 | 0.005 |
| Kernel lag-1 after 1 / 6+ ignores (× no gap) | 0.63 / 0.24 | 0.64 / 0.25 | 0.63 / 0.23 |
| Ignore rule λ | 0.710 | 0.708 | 0.703 |
| \|side\| after 1 / 2 / 3 ignores (× start) | 0.61 / 0.45 / 0.36 | 0.60 / 0.44 / 0.35 | 0.59 / 0.42 / 0.35 |
| Ignore attractor P(ignore), side logit | 0.978, +0.04 | 0.973, +0.71 | 0.983, −0.25 |
| Ignore rate from embedding, R² (C: 0.01) | 0.68 | 0.51 | 0.47 |
| R− attractor side logit | +4.86 | +4.40 | +3.34 |

![G1](analysis/fig_G1_seeds.png)

---

## Corrections and caveats

**Corrected during the study**

- **"Each ignore shrinks the side preference toward neutral"** is true of the side readout.
  On the TDR value coordinate an ignore carries v with the same leak as any input (I3c); what
  goes is the readout's last-choice offset. Repeated ignores pull v toward the ignore
  attractor's value coordinate (+1.8), not toward zero.
- **Embedding R² for ignore rate** is 0.47–0.68 across seeds, not seed 0's 0.68.
- **Attractor depths** are seed-specific, so no depth difference between C3 and C should be
  read from seed 0.

**Caveats that apply throughout**

- C3 counts an ignored trial as a step and C does not, so "k back" and "k steps on" differ
  between the models wherever ignores intervene.
- "Gap" counts only the ignores immediately before a trial; older bouts are not modelled in
  C4 or C1b.
- Fixed points and flow fields are at the mean context; the relaxation and causal tests use
  each state's own context.
- C comparisons use C seed 0 only; C's seed spread is taken from Study 50.
- Everything is teacher-forced. Free-running behaviour has not been tested.

---

## Open questions

- **Free-running rollout on C3.** Does the nearly absorbing ignore attractor explain Study
  02's finding that generated ignore bouts run too long (ignore → ignore +0.145)?
  Teacher-forced, C3's exit hazard matches the animals, so the excess may come from
  compounding when the model feeds on its own sampled ignores.
- **The same causal value test on C,** to confirm Study 50's 2-way model has the same uniform
  leak.
- **Per-mouse distilled models:** do mice differ in the ignore leak λ?
- **Conditional-mean ablations** (clamping to the mean for the last input), as in Study 50's
  E2b.
- **Why the value-axis relaxation is asymmetric after an ignore** (toward 0 from the left
  lobe, toward +1.9 from the right) when the side readout shrinks symmetrically. Probably a
  property of the coordinates; untested.
