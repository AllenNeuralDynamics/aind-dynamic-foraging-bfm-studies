---
schema_version: 1
id: r1
slug: first-look
status: draft
authors: [pck, claude]
wandb_groups: []
inputs:
  scripts:
    - analysis/extract_trial_states.py
    - analysis/split_alignment.py
    - analysis/b1_example_sessions.py
    - analysis/b2_unit_heatmap.py
    - analysis/b3_pca.py
    - analysis/b3_state_space.py
    - analysis/b4_update_maps.py
    - analysis/b5_outcome_triggered.py
    - analysis/b6_tracking.py
    - analysis/b7_decoders.py
    - analysis/b8_engagement.py
    - analysis/report_r1.py
  data:
    - analysis/trial_states_C3-s0.json
    - analysis/split_alignment.json
    - analysis/b1_example_sessions.json
    - analysis/b2_unit_heatmap.json
    - analysis/b3_pca.json
    - analysis/b3_state_space.json
    - analysis/b4_update_maps.json
    - analysis/b5_outcome_triggered.json
    - analysis/b6_tracking.json
    - analysis/b7_decoders.json
    - analysis/b8_engagement.json
upstream:
  - study: 50-gru-dynamics-embedding
    report: r1
  - study: 02-gru-scaling-law-ignore
    wandb_group: nxd-3way@20260707-075401
  - study: 01-gru-scaling-law
    wandb_group: e4-e8-d614-source@20260906-195409
reproduce: make -C studies/51-gru-dynamics-embedding-ignore r1
---

# r1 — First look at the 3-way GRU (C3 seed 0)

**Question.** What does the 3-way (left / right / ignore) GRU's state look like next to its
2-way twin C (Study 50), and does it predict when the animals disengage and re-engage?

Covers phase B: B1–B8. Comparisons with C use the sessions both eval splits share
(`split_alignment.json`), with C3 and C in separate panels.

## Summary

1. **The side prediction is unchanged.** On Study 50's four example sessions, C3's
   L/R likelihood on engaged trials is within 0.004 of C's. P(right | engaged) and C's
   P(right) are nearly the same curves (B1).
2. **Engagement gets its own direction.** The engage readout direction is nearly
   orthogonal to the side direction, and the two readouts load on different PCs: side on
   PC1 (61% of its variance), engagement on PC2 (63%). The state is slightly higher
   dimensional than C's (participation ratio 4.70 vs 4.31) (B3).
3. **A fifth state cluster.** C's state falls into four clusters by the last input
   (choice × outcome). C3 adds a fifth, separate cluster reached after an ignored trial,
   where P(ignore) is highest. Inside it, P(right | engaged) still spans 0–1, so the side
   belief is carried through an ignore (B3c).
4. **Teacher-forced, engagement is well calibrated** — overall, per probability bin, as a
   function of how many ignores just happened (the bout-exit hazard), of trial position in
   the session, and of trials since reward (B8), and in the ignore rate over the 10 steps
   after each input (B5d).
5. **Units specialise.** Pooled over 200k steps, 49 units correlate strongly (|r| > 0.5)
   with the side readout and 34 with the engage readout, only 5 with both; the two readouts
   are uncorrelated (r = −0.01) (B2).
6. **Side updates are C's.** After each engaged input (L±, R±), C3's side readout moves as
   C's does (update maps within ~0.17; matched reward effect +0.89 vs +0.86 one step on,
   ≈0.8 for six steps in both), and tracking of the reward probabilities is as close to
   the animals as C's (B4–B6).
7. **What an ignore does.** To the side readout, an ignore is a symmetric ~30% decay toward
   indifference (−3 → −2.15, +3 → +2.11) with no push to either side; on average it says
   nothing about the next side (P(right | engaged) ≈ 0.51 for ten steps). To the engage
   readout it is a large, lasting push toward disengagement: −1.06 log-odds versus an
   unrewarded choice one step on, still −0.58 after ten. A reward pushes toward engagement
   too, but briefly (+0.7 one step on, +0.25 by six), and equally for either side (B4, B5).
8. **Ignore history is in the state as well as choice and reward history** (ignored 1 back
   R² 0.98; 2 back unique R² 0.95), as are the recent ignore rate (0.94) and trial
   position (0.83, almost entirely outside PCs 1–4) (B7).

## What is by construction vs what is a finding

- **By construction.** `logit_diff` is exactly linear in the state (side·h + b), so its R²
  from the full state is 1; the engage direction is defined as the linear part of
  `engage_logit`. Calibration of the *overall* ignore rate is expected from a model trained
  by likelihood on the same population. In B8c–d, the trial before a bout onset and the
  trial at a bout end are engaged by definition, and the first two trials of a ≥2-trial
  bout are ignored by definition; those points are not findings. The B7 "check" rows
  (decoding the model's own readouts) are near 1 by construction. B5's matched effects
  remove regression to the mean only within the matching variable's bins.
- **Findings.** How much of each readout lives in the top PCs and on which; that the two
  directions are near-orthogonal; the fifth cluster and the side information inside it;
  calibration *conditional* on history (by bin, by k, by position, by reward history),
  which a model could fail while matching the overall rate.

## Relation to Study 02 r3

Study 02 r3 found that free-running rollouts of a 3-way GRU produce ignore bouts that are
too long (ignore→ignore +0.145). Here, with the animals' actual history as input, the
bout-exit hazard matches the data at every run length (B8b). So the per-step prediction of
re-engagement is right; the too-long bouts in rollouts likely come from the model feeding
on its own sampled ignores (an error that compounds), not from a miscalibrated exit
hazard. *Inference, not tested here*: it needs a rollout analysis on C3. (Study 02 r3 used
D=100/H256, not this model.)

## Caveats

- B8f counts ignored trials as unrewarded, so the rise of P(ignore) with trials since
  reward partly reflects bout continuation, not reward history alone.
- C3 counts an ignored trial as a step; C does not. So "k back" / "k steps on" differ between
  the models wherever ignores intervene. B7's slightly lower R² for deeper choice and reward
  lags in C3 (e.g. reward 4 back 0.50 vs C's 0.60) may reflect that, not lost memory.
- B6's ignore rate against reward-probability sum is not monotonic within tasks: each
  probability set belongs to particular eras / curricula, so richness is confounded with
  who ran it. The model still matches the observed rate per bin; no richness effect is
  claimed.
- One seed (C3-s0). Seeds 1–2 are pinned for phase G.

## Numbers

<!-- BEGIN result-1 -->
[regenerated by `analysis/report_r1.py` — do not edit by hand]

Model C3 seed 0, checkpoint step 90000: 4,761,751 timesteps (423,024 ignored, 8.9%), 8,906 eval sessions, 614 training mice. Logged vs recomputed eval likelihood: all 0.642175 vs 0.642176; LR_engaged 0.735084 vs 0.735084; engage 0.850045 vs 0.850046.

Shared with C (Study 50): 8,846 sessions, 4,314,340 engaged trials, identical choices and rewards (5 mice have shifted eval sessions; see `split_alignment.json`).

### B1 Example sessions (same sessions as Study 50)

| task | trials | ignored | L/R likelihood, engaged: C3 | C | C3 engage likelihood |
|---|---:|---:|---:|---:|---:|
| Uncoupled Baiting | 480 | 1 | 0.720 | 0.720 | 0.986 |
| Coupled Baiting | 568 | 13 | 0.552 | 0.555 | 0.927 |
| Uncoupled Without Baiting | 486 | 3 | 0.871 | 0.870 | 0.970 |
| Coupled Without Baiting | 646 | 77 | 0.821 | 0.819 | 0.729 |

![B1](../fig_B1_example_sessions.png)

### B2 Units

| | units with \|r\| > 0.5, side | engage | both | follow side more | follow engage more |
|---|---:|---:|---:|---:|---:|
| example session | 58 | 37 | 7 | 73 | 55 |
| pooled (200,000 steps) | 49 | 34 | 5 | 71 | 57 |

Correlation of the two readouts over the pooled steps: -0.011.

![B2](../fig_B2_unit_heatmap.png)

### B3 State space

| | C3 | C (Study 50) |
|---|---:|---:|
| PC1–6 variance (%) | 41.3, 14.0, 10.8, 6.5, 5.2, 4.1 | 42.7, 18.0, 7.6, 6.8, 5.6, 3.1 |
| participation ratio | 4.70 | 4.31 |
| PCs for 90/95/99% | 11/18/53 | 9/14/39 |

| readout | share of its variance on PC1, PC2, PC3, PC4 | R² from PCs 1–4 | direction's share inside PCs 1–4 |
|---|---|---:|---:|
| side | 0.61, 0.25, 0.00, 0.01 | 0.869 | 0.21 |
| engage | 0.21, 0.63, 0.09, 0.04 | 0.944 | 0.41 |

Cosine between the side direction (w_R − w_L) and the engage direction ((w_L + w_R)/2 − w_ignore): 0.10.

| previous input | timesteps | mean P(right \| engaged) | mean P(ignore) |
|---|---:|---:|---:|
| L+ | 975,107 | 0.085 | 0.027 |
| L− | 1,170,949 | 0.155 | 0.053 |
| R+ | 966,024 | 0.916 | 0.028 |
| R− | 1,221,229 | 0.846 | 0.055 |
| ignore | 419,536 | 0.514 | 0.569 |

![B3a](../fig_B3a_pca_variance.png)
![B3b](../fig_B3b_state_space_readouts.png)
![B3c](../fig_B3c_prev_input.png)

### B4 Update maps (shared sessions)

Mean readout on step t+1 when the readout on step t was at the listed value:

| row | input at t | n pairs | at -3 | at +0 | at +3 |
|---|---|---:|---:|---:|---:|
| C3 side: logit(R \| engaged) | L+ | 969,302 | -3.64 | -1.40 | -0.94 |
| C3 side: logit(R \| engaged) | L− | 1,165,013 | -2.73 | -0.56 | +0.25 |
| C3 side: logit(R \| engaged) | R+ | 959,916 | +0.90 | +1.38 | +3.42 |
| C3 side: logit(R \| engaged) | R− | 1,214,727 | -0.49 | +0.49 | +2.56 |
| C3 side: logit(R \| engaged) | ignore | 416,729 | -2.15 | -0.08 | +2.11 |
| C side: logit(R) | L+ | 968,760 | -3.60 | -1.25 | -0.88 |
| C side: logit(R) | L− | 1,163,860 | -2.70 | -0.51 | +0.25 |
| C side: logit(R) | R+ | 959,384 | +0.87 | +1.25 | +3.36 |
| C side: logit(R) | R− | 1,213,490 | -0.32 | +0.42 | +2.54 |
| C3 engage: logit(engaged / ignore) | L+ | 969,302 | +1.02 | +2.61 | +5.28 |
| C3 engage: logit(engaged / ignore) | L− | 1,165,013 | +0.21 | +1.81 | +4.60 |
| C3 engage: logit(engaged / ignore) | R+ | 959,916 | +1.03 | +2.57 | +5.28 |
| C3 engage: logit(engaged / ignore) | R− | 1,214,727 | +0.23 | +1.83 | +4.56 |
| C3 engage: logit(engaged / ignore) | ignore | 416,729 | -0.52 | +0.74 | +2.45 |

(engage rows at engage_logit 0 / +2 / +5.)

![B4](../fig_B4_update_maps.png)

### B5 Outcome-triggered averages (shared sessions)

| matched effect | k = 1 | 2 | 3 | 5 | 10 |
|---|---:|---:|---:|---:|---:|
| C3 reward → side, chose R | +0.89 | +0.86 | +0.82 | +0.79 | +0.68 |
| C reward → side, chose R | +0.86 | +0.84 | +0.80 | +0.80 | +0.66 |
| C3 reward → side, chose L | -0.94 | -0.88 | -0.85 | -0.86 | -0.74 |
| C reward → side, chose L | -0.91 | -0.86 | -0.85 | -0.85 | -0.71 |
| C3 reward → engage, chose R | +0.71 | +0.54 | +0.36 | +0.26 | +0.07 |
| C3 reward → engage, chose L | +0.70 | +0.53 | +0.41 | +0.31 | +0.09 |
| C3 ignored − unrewarded → engage | -1.06 | -0.87 | -0.76 | -0.70 | -0.58 |

P(ignore) on step t+k after each input, model / animal:

| input | k = 1 | 2 | 5 | 10 |
|---|---:|---:|---:|---:|
| R+ | 0.028 / 0.028 | 0.033 / 0.033 | 0.042 / 0.042 | 0.050 / 0.050 |
| R− | 0.055 / 0.054 | 0.058 / 0.058 | 0.062 / 0.061 | 0.064 / 0.064 |
| L+ | 0.027 / 0.027 | 0.032 / 0.033 | 0.041 / 0.042 | 0.050 / 0.050 |
| L− | 0.053 / 0.053 | 0.056 / 0.056 | 0.060 / 0.060 | 0.062 / 0.062 |
| ignore | 0.570 / 0.571 | 0.531 / 0.532 | 0.476 / 0.474 | 0.430 / 0.430 |

![B5](../fig_B5_outcome_triggered.png)

### B6 Tracking (shared sessions; trials ≥ 10 after a probability change)

| task | max \|model − animal\|: C3 side | C side | C3 ignore vs richness |
|---|---:|---:|---:|
| Uncoupled Baiting | 0.033 | 0.041 | 0.006 |
| Coupled Baiting | 0.058 | 0.062 | 0.024 |
| Uncoupled Without Baiting | 0.022 | 0.014 | 0.033 |
| Coupled Without Baiting | 0.022 | 0.010 | 0.023 |

![B6](../fig_B6_tracking.png)

### B7 Decoders (cross-validated R², 400k steps)

| variable | R² | unique R² | share in PCs 1–4 | C (Study 50) R² |
|---|---:|---:|---:|---:|
| choice 1 back | 0.990 |  | 0.80 | 0.999 |
| reward 1 back | 0.996 |  | 0.71 | 0.998 |
| choice×reward 1 back | 0.975 |  | 0.39 | 0.966 |
| ignored 1 back | 0.982 |  | 0.60 |  |
| choice 2 back | 0.975 | 0.935 | 0.72 | 0.987 |
| reward 2 back | 0.967 | 0.964 | 0.26 | 0.984 |
| choice×reward 2 back | 0.928 | 0.892 | 0.39 | 0.953 |
| ignored 2 back | 0.963 | 0.950 | 0.39 |  |
| choice 3 back | 0.927 | 0.780 | 0.70 | 0.935 |
| reward 3 back | 0.812 | 0.781 | 0.20 | 0.868 |
| choice×reward 3 back | 0.740 | 0.579 | 0.46 | 0.823 |
| ignored 3 back | 0.849 | 0.737 | 0.39 |  |
| choice 4 back | 0.808 | 0.422 | 0.73 | 0.861 |
| reward 4 back | 0.497 | 0.395 | 0.22 | 0.595 |
| choice×reward 4 back | 0.596 | 0.302 | 0.53 | 0.677 |
| ignored 4 back | 0.664 | 0.382 | 0.44 |  |
| choice 5 back | 0.748 | 0.235 | 0.72 | 0.817 |
| reward 5 back | 0.362 | 0.222 | 0.27 | 0.414 |
| choice×reward 5 back | 0.522 | 0.170 | 0.57 | 0.583 |
| ignored 5 back | 0.573 | 0.211 | 0.49 |  |
| P(rew|R) − P(rew|L) | 0.397 |  | 0.06 | 0.426 |
| P(rew|R) + P(rew|L) | 0.281 |  | 0.51 | 0.312 |
| trials since left chosen (≤20) | 0.885 |  | 0.68 | 0.906 |
| trials since right chosen (≤20) | 0.876 |  | 0.76 | 0.925 |
| trials since last ignore (≤20) | 0.848 |  | 0.75 |  |
| ignore rate, last 10 steps | 0.943 |  | 0.63 |  |
| trial position | 0.831 |  | 0.05 |  |
| model logit(R | engaged) (check) | 1.000 |  | 0.87 |  |
| model logit(engaged / ignore) (check) | 0.996 |  | 0.95 |  |

C's choice / reward variables are coded over engaged trials only (no ignore steps), so their R² compare like with like only approximately.

![B7](../fig_B7_decoders.png)

### B8 Engagement

Mean predicted P(ignore) 0.0889 vs observed rate 0.0888. Calibration: largest |observed − predicted| over 13 bins = 0.0054. Ignore runs: 183,618 (178,976 not cut by a session edge; mean length 2.16, 63% single trials, 90th / 99th percentile 4 / 14).

P(ignore | k ignores just before), observed / model:

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| observed | 0.042 | 0.376 | 0.575 | 0.674 | 0.739 | 0.756 | 0.791 | 0.794 | 0.817 | 0.838 | 0.812 | 0.858 | 0.866 | 0.875 | 0.876 | 0.922 |
| model | 0.042 | 0.376 | 0.582 | 0.647 | 0.724 | 0.764 | 0.790 | 0.809 | 0.823 | 0.836 | 0.846 | 0.855 | 0.863 | 0.870 | 0.876 | 0.915 |
| n | 4,342,215 | 182,918 | 68,335 | 38,978 | 25,989 | 19,002 | 14,212 | 11,086 | 8,669 | 6,979 | 5,728 | 4,550 | 3,834 | 3,265 | 2,805 | 23,186 |

Largest |observed − model| for k ≤ 14: 0.034.

![B8](../fig_B8_engagement.png)
<!-- END result-1 -->
