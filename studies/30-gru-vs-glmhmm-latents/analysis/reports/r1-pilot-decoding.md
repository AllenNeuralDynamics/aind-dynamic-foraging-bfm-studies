---
schema_version: 2
id: r1
slug: pilot-decoding
status: draft
draft_reason: "claims check and first-author review pending"
authors:
- han
issue: 136
tags:
  model: [gru, glmhmm]
  question: [interpretation, comparison]
wandb_groups:
- hurdle@20261027-212905
- reference@20261027-213006
inputs:
  script: analysis/r1_pilot_decoding.py
  data:
  - analysis/frozen_r1.json
  - analysis/frozen_r1_sessions.npz
  claims: analysis/r1_pilot_decoding.json
  freeze: analysis/freeze_r1.py
  oof_refit: analysis/oof_r1.py
  pilot_code: analysis/pilot
claims:
- id: F1
  text: "TODO"
  pointer: analysis/r1_pilot_decoding.json#/f1/latents_min_pooled_r2/point
- id: F2
  text: "TODO"
  pointer: analysis/r1_pilot_decoding.json#/f2/latents/engaged_r2_S1_vs_S2/point
- id: F3
  text: "TODO"
  pointer: analysis/r1_pilot_decoding.json#/f3/latents_null_max_within
- id: F4
  text: "TODO"
  pointer: analysis/r1_pilot_decoding.json#/f4/latents_K3_minus_K10_pooled_r2/point
prose_reviewed:
  date: '2026-11-18'
  blocks: {}
upstream:
- study: 02-gru-scaling-law-ignore
  wandb_group: nxd-3way@20260707-075401
- study: 14-glmhmm-vs-gru-heldout
  wandb_group: hurdle@20261027-212905
- study: 14-glmhmm-vs-gru-heldout
  wandb_group: reference@20261027-213006
reproduce: make -C studies/30-gru-vs-glmhmm-latents r1
---
# r1 — The GRU's latents carry the GLM-HMM's state posterior at every K, beyond engagement (training mice)

## Question

Study 02's 3-way GRU was trained only to predict the next choice; study 14's GLM-HMM explains the same
trials with K discrete states. Does the GRU's hidden state contain, in linearly readable form, the
GLM-HMM's belief about which state the mouse is in, beyond what the GRU's own prediction and simple
choice and reward history carry? If it does, the expressive model and the interpretable one describe the
same behavioural variables, and the GLM-HMM's states can be used to read the GRU.

## Answer

- **F1.** TODO
- **F2.** TODO
- **F3.** TODO
- **F4.** TODO

## Methods

<!-- BEGIN methods -->
(written by analysis/r1_pilot_decoding.py)
<!-- END methods -->

As in the study methods ([README](../../README.md), Design), on the training-mice pilot
([`variants/training-mice-pilot`](../../variants/training-mice-pilot/notes.md)). The target is the
GLM-HMM's one-step **predictive state posterior** π_t = p(z_t | trials < t)
([CONTEXT.md](../../CONTEXT.md), predictive posterior): the probability of each state before trial t's
choice, the same information the GRU's hidden state h_t has when it predicts trial t. The decoder is a
**softmax linear readout**: a one-layer network (multinomial logistic regression) that maps h_t to K
state probabilities, fitted to the soft target by cross-entropy. Three scores
([CONTEXT.md](../../CONTEXT.md), probe scores): **pooled R²**, the share of the posterior's variance
the decoder reproduces, summed over states; the same **within session**, after removing each session's
mean from target and prediction, so that only changes inside a session count; and **information
recovered**, 1 − KL(decoder) / KL(best constant guess). Baselines get the same decoder: the GRU's own
3 output logits, linear history features (the GLM-HMM's 40 lag regressors, running averages of ignores
and rewards, trial number), latents plus history, the untrained GRU, and trial number alone.

Uncertainty: a **bootstrap over mice** resamples the 50 mice with replacement 1,000 times and
re-scores the fixed out-of-fold predictions; the interval is the 2.5th to 97.5th percentile. It captures
which mice were sampled, not GRU seed (one seed) or GLM-HMM refit variability. The out-of-fold
predictions were regenerated for every probe (`analysis/oof_r1.py`) because the pilot saved them for
latents and history only; every refit reproduces the pilot's scores (the freeze asserts it).

## Results

### Fig. 1 — At every K the latents decode the state posterior; the GRU's output and history do not keep up (F1)

<!-- BEGIN result-1 -->
(written by analysis/r1_pilot_decoding.py)
<!-- END result-1 -->

### Fig. 2 — The decoded state is more than engagement (F2)

<!-- BEGIN result-2 -->
(written by analysis/r1_pilot_decoding.py)
<!-- END result-2 -->

### Fig. 3 — Shuffle nulls: the decoding follows this session's state at this moment (F3)

<!-- BEGIN result-3 -->
(written by analysis/r1_pilot_decoding.py)
<!-- END result-3 -->

### Readability does not pick K (F4)

<!-- BEGIN result-4 -->
(written by analysis/r1_pilot_decoding.py)
<!-- END result-4 -->

## Supplementary

### SupFig. 1 — Where the state sits in the population

<!-- BEGIN supp-1 -->
(written by analysis/r1_pilot_decoding.py)
<!-- END supp-1 -->

### SupFig. 2 — Example sessions with clear state transitions

<!-- BEGIN supp-2 -->
(written by analysis/r1_pilot_decoding.py)
<!-- END supp-2 -->

### SupFig. 3 — Study 14's saved `filtered_k` includes trial t

<!-- BEGIN supp-3 -->
(written by analysis/r1_pilot_decoding.py)
<!-- END supp-3 -->

Any analysis that predicts trial t from study 14's saved posteriors must use filtered_{t−1} · A, or it
leaks trial t's choice into the predictor. r1 uses the predictive posterior throughout.

## What it means

- **The GRU learned the variables the GLM-HMM uses to define its states (F1, F2).** A linear readout
  of its hidden state recovers the GLM-HMM's state belief on mice the readout never saw, including the
  split between the two engaged states, which differ in how strongly the mouse follows reward history.
  The GRU's own output and linear history features carry engagement but not that split.
- **It is not a generic slow-signal artefact (F3).** The decoder fails when the target comes from
  another mouse's session or is shifted in time within the same session, so it tracks this session's
  state at this moment. The circular shift keeps part of the all-trials score because a session's average
  state survives the rotation.
- **This is evidence for shared variables, not for discrete states (F4).** The readout recovers the
  K = 10 posterior almost as well as the K = 3 one, although study 14's per-mouse score selects K = 3.
  Whether the mouse switches between discrete modes or drifts along these variables is open.
- **The state is a small direction in the population (SupFig. 1).** Choice and outcome dominate the
  latents' variance; the state variables add a small unique share on held-out mice. The axes for
  "choice t−1" and "ignore t−1" are nearly parallel: the 3-way GRU receives the previous choice as one
  number (0 = left, 1 = right, 2 = ignore), so an ignore enters as an "extra-right" choice. A one-hot
  input would separate them.

## Caveats

- **Training mice, internal validation.** The decoder is held out by mouse, but the GRU learned these
  mice's subject embeddings from their even sessions, GRU early stopping used these odd sessions
  (eval likelihood rose at every checkpoint, so the final checkpoint was chosen and the selection had no
  effect), and the GLM-HMM global fit used their even sessions ([PLAN.md](../../PLAN.md), leakage
  audit). The held-out variant removes all three.
- **One GRU** (single seed) and the GLM-HMM's **global** parameters, not per-mouse refits. The
  intervals cover mouse sampling only.
- **Linear readout and linear history baseline.** The GLM-HMM posterior is a nonlinear recursion over
  the whole session, so the claim is linear readability; a nonlinear history decoder could close part of
  the gap in F1 and F2.
- **The untrained-GRU decoder** reached the iteration cap in most fits (Fig. 1 caption) and has no
  subject or session context, so its score is a loose baseline.
- **Few shuffles** per null cell (Fig. 3 table); the nulls are far below the real scores.
- **Selected examples.** SupFig. 2's sessions were chosen for clear transitions; every score pools all
  trials.
- **SupFig. 1** comes from one TDR fit; its intervals cover test-mouse sampling (and training-mouse
  sampling for the cosine), not the choice of regressors or the 12-PC denoising.

## Next

Repeat on the 149 held-out mice (variant `heldout-mice`, report r2): GRU embeddings fine-tuned offline with study 02's recipe, and per-mouse GLM-HMM refits with state
labels matched across mice ([PLAN.md](../../PLAN.md)). Then more GRU seeds, a one-hot 3-way GRU and a
nonlinear history decoder.

## Provenance

GRU: study 02 run
[`nxd-3way-20260707-075401-80a6d102`](https://wandb.ai/AIND-disRNN/mice_ignore_scaling/runs/nxd-3way-20260707-075401-80a6d102)
(group `nxd-3way@20260707-075401`). GLM-HMM: study 14 groups
[`hurdle@20261027-212905`](https://wandb.ai/AIND-disRNN/mice_glmhmm/groups/hurdle@20261027-212905) and
[`reference@20261027-213006`](https://wandb.ai/AIND-disRNN/mice_glmhmm/groups/reference@20261027-213006).
Frozen inputs: `analysis/frozen_r1.json` and `analysis/frozen_r1_sessions.npz` (`make freeze-r1`, which
reads the pilot's scratch outputs and the out-of-fold refit, each by sha256). Reproduce with
`make -C studies/30-gru-vs-glmhmm-latents r1`.
