# Plan — study 30

Issue #136. Vocabulary: [`CONTEXT.md`](CONTEXT.md). Result so far: [r1](analysis/reports/r1-pilot-decoding.md)
(training-mice pilot).

## Where the study stands

r1 is an **internal validation**: a readout of study 02's GRU latents recovers study 14's GLM-HMM predictive
state posterior on mice the readout never saw, but the GRU and the GLM-HMM global fit had seen those mice.
The next variant, `heldout-mice`, is the **external validation**: mice that neither model was trained on,
scored with the study 02 / study 14 held-out protocol.

## r1 leakage audit

| possible leak | r1 | verdict |
|---|---|---|
| the probe saw the test mice | probe, scaler, Rastermap and TDR fitted on the other 40 mice (5-fold CV by mouse) | no leak |
| the target contains the answer | predictive posterior p(z_t \| trials < t); study 14's `filtered_k` (includes trial t) is not used | no leak |
| the GRU trained on the scored trials | GRU trained on even sessions; r1 scores odd sessions | no leak |
| examples picked by decoding quality | picked by GLM-HMM transitions; scores pool every trial | no leak |
| generic slow-signal artefact | session-swap and circular-shift nulls ≤ 0.03 within session | ruled out |

**Contact that is not a leak, but limits the claim:**
- The GRU learned these mice's subject embeddings from their even sessions.
- GRU early stopping used these odd sessions. Eval likelihood rose at every checkpoint, so the final checkpoint
  was chosen and the selection had no effect.
- The GLM-HMM global fit used these mice's even sessions.

`heldout-mice` removes all three.

## Variant `heldout-mice` (next)

| | r1 (`training-mice-pilot`) | `heldout-mice` |
|---|---|---|
| mice | 50 GRU-training mice | the 149 reserved held-out mice |
| GRU | embeddings learned in training | network frozen; new embedding rows fine-tuned on each mouse's even sessions |
| GLM-HMM target | global parameters, one shared set of states | each mouse's MAP refit (study 14 protocol); labels matched across mice |
| probe | trained and scored within the 50 mice, CV by mouse | (1) CV by mouse within the 149 mice; (2) trained on training mice, scored on held-out mice |
| scored trials | odd (eval) sessions | odd (eval) sessions |

### Inputs and what exists (checked 2026-11-11)

- **GLM-HMM: ready.** Run `u4j7ukti` (hurdle K = 3) has per-mouse refits for all 149 held-out mice in its
  `fit.json`, and their posteriors in `heldout_trials.parquet`. Check the other K-grid runs before using them.
- **GRU: missing.** The H256 / D614 / seed 0 run (`nxd-3way-20260707-075401-80a6d102`) ran on Beaker. Its
  held-out fine-tune output was never uploaded, and its scratch folder (2026-07-13) is empty. Only the global
  checkpoint is available, from the W&B artifact r1 used.

### Steps

1. **Redo the fine-tune offline.** Same recipe as study 02: start from the global checkpoint, add one embedding
   row per held-out mouse, train only those rows for 500 steps (lr 1e-3) on the mouse's even sessions. The job
   runs with W&B off. **Never** use `resume_heldout_beaker.py`: it resumes the study 02 run and overwrites its
   results. Files: [`variants/heldout-mice/`](variants/heldout-mice/notes.md).
2. **Reproduction check.** The redone fine-tune must reproduce the run's logged held-out score
   (`heldout/final/eval_likelihood`), read from study 02's committed `analysis/frozen_runs.json`, never retyped. A
   mismatch beyond 1e-4 stops the variant (the fine-tune is deterministic). Then `upload_wandb.py` logs the
   result to study 30's W&B project (decision below).
3. **Latents.** GRU latents (and the untrained-GRU control) on the held-out mice's odd sessions.
4. **Targets.** Predictive posterior from each mouse's own refit: π_t = α_{t−1} · A_mouse (π⁰ at session start).
   Match each mouse's states to the global states by emission weights (study 14 r1 method; 141 / 149 matched at
   K = 3). Report unmatched mice; score them separately or drop them, decided before scoring.
5. **Probes.** The r1 decoder (softmax linear readout, soft labels, L2 10⁻⁴), baselines and shuffle nulls,
   under both probe protocols above.
6. **Report r2**: held-out results beside r1's, with the same figures where they apply.

**What would change the r1 conclusion:** held-out within-session R² far below r1's 0.72–0.88 means part of r1's
score came from the GRU's familiarity with the training mice.

## Decided

- **Fine-tuned embeddings go to W&B (decision (b), Han, 2026-11-11).** After the reproduction check, one run
  and one artifact (fine-tuned parameters, subject maps, embeddings, per-trial predictions) in
  `NeuralDynamics/dfbfm_30-gru-vs-glmhmm-latents`, group `heldout-mice@<launch_id>`, with a launch record.
  Other studies (for example 51, or an HB comparison) can reuse them. The alternative was local files only.
- **No W&B upload of r1's per-trial intermediates** (Han, 2026-11-11). They rebuild from durable sources (the
  study 02 GRU artifact, the study 14 runs, snapshot 20260603) in about 15–30 min, plus 3–4 CPU-hours of probes;
  r1's freeze records each source's sha256.

## Later follow-ups (order to be set)

1. More GRU seeds and sizes.
2. A one-hot 3-way GRU. The current GRU receives the previous choice as one number (0 / 1 / 2); r1's TDR shows
   the choice and ignore axes at cosine 0.9. Needs a wrapper change first.
3. A nonlinear history decoder (small MLP or gradient boosting): the fair test of "beyond history".
4. Tests of discreteness: bimodality and sharpness along the decoded axes, dwell-time shape.
5. Decoding error against trials since the last ignore or reward (r1's misses and lags).
6. Embeddings: do the GRU's per-mouse embeddings predict per-mouse GLM-HMM parameters?
