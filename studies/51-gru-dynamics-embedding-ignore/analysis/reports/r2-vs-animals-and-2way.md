---
schema_version: 1
id: r2
slug: vs-animals-and-2way
status: draft
authors: [pck, claude]
wandb_groups: []
inputs:
  scripts:
    - analysis/c2_compare.py
    - analysis/c4_after_ignore.py
    - analysis/c1_kernels.py
    - analysis/c1_impulse.py
    - analysis/c1_figure.py
    - analysis/report_r2.py
  data:
    - analysis/c2_compare.json
    - analysis/c4_after_ignore.json
    - analysis/c1_kernels.json
    - analysis/c1_impulse.json
upstream:
  - study: 50-gru-dynamics-embedding
    report: r2
  - study: 02-gru-scaling-law-ignore
    wandb_group: nxd-3way@20260707-075401
  - study: 01-gru-scaling-law
    wandb_group: e4-e8-d614-source@20260906-195409
reproduce: make -C studies/51-gru-dynamics-embedding-ignore r2
---

# r2 — The 3-way GRU against the 2-way GRU, the animals and RL

**Question.** What does seeing ignored trials change in the GRU's side prediction, and what
does an ignore input do inside the network? Compared on the engaged trials both C3 and C
(Study 50) evaluate (8,846 shared sessions, 4.31M trials), with Study 01's per-mouse RL fits.

## Summary — how C3 differs from C, and what the ignore input does

1. **C3 predicts side choice better than C, and all of the gain is after ignores.** Pooled
   L/R likelihood 0.7351 vs 0.7338, better in every task and in 71% of mice. On trials whose
   previous trial was engaged (96%), C3 is marginally *worse* (−0.0005 nats per trial). After
   1 ignore it is +0.033 nats per trial better, after 2 +0.079, after 3–5 +0.10, after 6–10
   +0.18, after 11+ +0.25, in 85–98% of mice. After long bouts C and the RL models drop
   below chance (0.41–0.44); C3 stays above it (C2).
2. **Why: animals forget their side preference over an ignore bout, and C3 has learned the
   rate.** The probability of repeating the last engaged choice falls from 0.88 (no gap) to
   0.77 after one ignore, 0.71 after two and 0.59 after 11+; C3 tracks this within ~0.01 at
   every gap, after rewarded and unrewarded choices alike. C and CTT predict ~0.81–0.84
   whatever the gap: they cannot see it (C4).
3. **The animals' whole history kernel shrinks after a gap, and C3's with it.** The lag-1
   weight falls to 0.64 of its no-gap value after one ignore, 0.45 after two, 0.38 at 3–5
   and 0.23 at 6+; C3: 0.63, 0.49, 0.40, 0.24. C and CTT stay at 0.95–1.03. With no gap the
   three kernels agree (rewarded lag 1: animal 1.85, C3 1.85, C 1.80) (C1b).
4. **The reward kernel itself is unchanged by the third action.** Flipping one reward moves
   C3's side readout +1.04 → +0.62 → +0.35 over the next steps, C's +1.00 → +0.60 → +0.35 (C1a).
5. **An ignore input is a strong, persistent push toward disengaging.** Turning one engaged
   trial into an ignore lowers the engage readout by 2.27 log-odds on the next step and still
   by 0.89 ten steps later (39%); a reward's effect on engagement is smaller (+0.86) and
   almost gone by ten steps (6%). On the side readout the edit removes that trial's
   evidence (−1.24, larger when the trial had been rewarded: −1.84 vs −0.76) and fades like
   any single input. The reverse edit, an ignore turned into an unrewarded choice, raises
   both readouts (+0.99 toward the inserted side, +0.98 engage), with a shorter-lived engage
   effect (C1a).

Together with r1 (B4: each ignore shrinks the side log-odds ~30% toward 0), the picture is
consistent across behaviour, regression and causal edits: an ignore both withholds evidence
and erodes the stored side preference, and it sets the network on a slow path toward
disengagement. C, which never sees ignores, keeps its full preference through a bout and is
over-confident afterwards.

## What is by construction vs what is a finding

- **By construction.** C and the RL models cannot respond to a gap they never receive, so
  their flat curves in C2, C4 and C1b are expected. C3 can see the gap; that it *could* use
  it is by design. The 3-way head being scored only on L/R here is our choice (Study 02's
  metric caveat), not something the model optimised alone.
- **Findings.** That the animals' choice depends on the gap at all, and by how much; that C3
  reproduces the dependence quantitatively (stay probabilities, kernel shrinkage, likelihood
  gain) rather than only qualitatively; that the gain is confined to post-ignore trials with
  a slight cost elsewhere; that the reward kernel is unchanged; the size, persistence and
  asymmetry of an ignore's causal effect on engagement.

## Caveats

- **Gap is only the ignores just before.** Ignores further back are not modelled in C1b or C4;
  C3 sees them, so some of its edge after short gaps may come from older bouts.
- **The two engage edits start from different states.** An engaged → ignore edit starts on an
  engaged trial; an ignore → engaged edit starts inside the post-ignore cluster. So the
  asymmetry in C1a describes each edit in its own context, not a matched pair.
- **C1a removes or adds one input; it does not change the animal's later behaviour.** Later
  inputs are as observed, so it isolates the network's response, not a behavioural effect.
- **The RL fits are Study 01's,** replayed on C's trials (they skip ignores, like C).
- One seed (C3-s0).

## Numbers

<!-- BEGIN result-1 -->
[regenerated by `analysis/report_r2.py` — do not edit by hand]

### C2 Side-choice likelihood on the shared engaged trials

| trials | n | C3 | C | ctt | bari | hattori |
|---|---:|---:|---:|---:|---:|---:|
| all | 4,314,340 | 0.7351 | 0.7338 | 0.7210 | 0.7180 | 0.7159 |
| Uncoupled Baiting | 2,103,617 | 0.7430 | 0.7421 | 0.7303 | 0.7273 | 0.7254 |
| Coupled Baiting | 1,383,117 | 0.6854 | 0.6843 | 0.6688 | 0.6665 | 0.6652 |
| Uncoupled Without Baiting | 675,464 | 0.8143 | 0.8116 | 0.8022 | 0.7976 | 0.7934 |
| Coupled Without Baiting | 148,807 | 0.7578 | 0.7547 | 0.7413 | 0.7403 | 0.7340 |

Per mouse (611): C3 > C in 71% (median Δ +0.0010); C3 > every RL model in 100%.

| ignored just before (gap) | trials | mice | C3 | C | CTT | C3 − C per trial (nats, mean over mice ± 1.96 SEM) | mice with C3 > C |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4,135,637 | 611 | 0.742 | 0.742 | 0.729 | -0.0005 ± 0.0002 | 34% |
| 1 | 113,192 | 603 | 0.614 | 0.597 | 0.586 | +0.0329 ± 0.0032 | 85% |
| 2 | 28,830 | 430 | 0.574 | 0.535 | 0.525 | +0.0792 ± 0.0079 | 85% |
| 3–5 | 23,981 | 363 | 0.552 | 0.496 | 0.487 | +0.1043 ± 0.0106 | 87% |
| 6–10 | 8,999 | 150 | 0.528 | 0.440 | 0.434 | +0.1817 ± 0.0253 | 91% |
| 11+ | 3,701 | 44 | 0.516 | 0.406 | 0.408 | +0.2507 ± 0.0500 | 98% |

![C2](../fig_C2_likelihood.png)

### C4 Repeating the previous engaged choice after a gap

| previous outcome | gap | n | animal | C3 | C | CTT | confidence C3 | confidence C |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rewarded | 0 | 1,876,056 | 0.915 | 0.916 | 0.912 | 0.906 | 0.418 | 0.414 |
| rewarded | 1 | 33,979 | 0.815 | 0.819 | 0.888 | 0.887 | 0.324 | 0.391 |
| rewarded | 2 | 8,076 | 0.750 | 0.767 | 0.879 | 0.879 | 0.276 | 0.382 |
| rewarded | 3–5 | 6,563 | 0.712 | 0.720 | 0.872 | 0.873 | 0.229 | 0.375 |
| rewarded | 6–10 | 2,443 | 0.649 | 0.653 | 0.867 | 0.863 | 0.168 | 0.371 |
| rewarded | 11+ | 1,027 | 0.599 | 0.596 | 0.867 | 0.867 | 0.121 | 0.369 |
| unrewarded | 0 | 2,251,877 | 0.849 | 0.847 | 0.843 | 0.841 | 0.354 | 0.349 |
| unrewarded | 1 | 78,725 | 0.749 | 0.743 | 0.817 | 0.819 | 0.256 | 0.324 |
| unrewarded | 2 | 20,584 | 0.693 | 0.696 | 0.808 | 0.812 | 0.213 | 0.315 |
| unrewarded | 3–5 | 17,217 | 0.653 | 0.650 | 0.796 | 0.801 | 0.169 | 0.304 |
| unrewarded | 6–10 | 6,408 | 0.600 | 0.607 | 0.791 | 0.798 | 0.130 | 0.299 |
| unrewarded | 11+ | 2,539 | 0.590 | 0.577 | 0.791 | 0.798 | 0.104 | 0.302 |

![C4](../fig_C4_after_ignore.png)

### C1b History kernels by gap (Su2022, 10 engaged trials back)

Lag-1 weight as a fraction of its value at gap 0 (rewarded / unrewarded choice):

| source | 0 | ≥1 | 1 | 2 | 3–5 | 6+ |
|---|---:|---:|---:|---:|---:|---:|
| animal | 1.00 / 1.00 | 0.53 / 0.48 | 0.64 / 0.58 | 0.45 / 0.36 | 0.38 / 0.33 | 0.23 / 0.23 |
| C3 | 1.00 / 1.00 | 0.54 / 0.48 | 0.63 / 0.56 | 0.49 / 0.41 | 0.40 / 0.35 | 0.24 / 0.21 |
| C | 1.00 / 1.00 | 0.97 / 1.00 | 0.98 / 1.00 | 0.96 / 0.99 | 0.95 / 0.99 | 0.96 / 1.02 |
| ctt | 1.00 / 1.00 | 1.00 / 1.00 | 1.01 / 1.00 | 1.00 / 0.99 | 1.00 / 0.99 | 0.98 / 1.03 |

Lag-1 weights at gap 0: animal 1.85 / 0.91; C3 1.85 / 0.90; C 1.80 / 0.87; ctt 1.73 / 0.94.

![C1 kernels](../fig_C1_kernels.png)

### C1a Counterfactual edits (one trial's input; later inputs as observed)

Replay check: side median |err| 4.8e-07, engage 5.5e-07. Ignored trials enter as (choice 2, reward 0): checked on 419,536 rows.

| effect (mean over 20,000 edits) | k = 1 | 2 | 3 | 5 | 10 |
|---|---:|---:|---:|---:|---:|
| reward flip → side, C3 | +1.036 | +0.617 | +0.354 | +0.157 | +0.021 |
| reward flip → side, C (Study 50) | +1.004 | +0.596 | +0.346 | +0.160 | +0.017 |
| reward flip → engage, C3 | +0.864 | +0.521 | +0.286 | +0.173 | +0.048 |
| engaged → ignore → side (was rewarded) | -1.843 | -0.813 | -0.453 | -0.174 | -0.032 |
| engaged → ignore → side (was unrewarded) | -0.757 | -0.263 | -0.133 | -0.038 | -0.011 |
| engaged → ignore → engage (was rewarded) | -2.742 | -2.047 | -1.651 | -1.347 | -0.956 |
| engaged → ignore → engage (was unrewarded) | -1.889 | -1.543 | -1.339 | -1.105 | -0.827 |
| ignore → unrewarded choice → side | +0.991 | +0.476 | +0.282 | +0.182 | +0.075 |
| ignore → unrewarded choice → engage | +0.976 | +0.605 | +0.381 | +0.282 | +0.154 |

![C1 impulse](../fig_C1_impulse.png)
<!-- END result-1 -->
