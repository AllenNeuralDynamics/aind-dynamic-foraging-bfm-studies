# Plan — 17-gru-crossfit-predictability

Issue #148. Companions: study 41 (#149, the strategy study) and
[MEvolve #87](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-mevolve/issues/87)
(scientist mode). Status: **design settled 2026-11-12; wrapper prerequisite open; nothing
launched.** Terms are defined in [`CONTEXT.md`](CONTEXT.md).

## Why

The reference GRU trained on the whole training pool, so only its 149 held-out mice have
held-out GRU predictions, and every comparison with the GRU spends the test set (studies 01, 12,
13, 15 and 40 all scored there). Cross-validated folds give a held-out prediction for every
mouse. Study 41 uses the result as its ceiling and residual map: where an interpretable model
loses to the GRU, structure is missing. The spread across folds shows how stable that ceiling is.

## Design (decided with Han, 2026-11-12)

**Rotation over all 763 mice, 5 folds.** The test fold is one of the 5 folds; each fold model
trains on every mouse outside its fold, test mice included.

| model | held out (scored) | trains on | source |
|---|---|---|---|
| 0 | test fold: the freeze's 149 test subjects (`pool == heldout`) | the 614 pool mice | **reused**: study 02 `nxd-3way` H256 seed 0, run `nxd-3way-20260707-075401-80a6d102` (`AIND-disRNN/mice_ignore_scaling`, group `nxd-3way@20260707-075401`) |
| 1–4 | pool fold k (≈153 mice) | the other 3 pool folds + the 149 test mice (≈610) | new training, seed 0 |

- **Model: 3-way GRU** (left / right / ignore), the study 02 `nxd-3way` recipe at **H256**
  (`ignore_policy: include`, snapshot `20260603`, λ-forward session conditioning, 150k steps,
  gated early stop). Study 02 found the ignore class costs nothing on left/right prediction; H256
  at 614 mice is its best cell (Han, 2026-11-12, switched from H128).
- **Pool folds:** the 614 pool mice in 4 parts, stratified by curriculum × session-count thirds
  with a fixed seed (as study 40 ADR 0003), then checked so the 296 discovery and 315
  confirmation mice spread across all 4. The map is committed before training as
  `reference/folds.json` (mouse → fold; `analysis/freeze_folds.py`, which also writes the
  `pool-folds` sweep).
- **Mouse lists follow the freeze.** 764791, 808057, 823164 (on study 01's 2-way held-out list)
  are pool mice here; 722683, 795395, 820243 (freeze test mice that study 01 trained on) are in
  the test fold. Study 02's model already uses exactly the freeze's lists.
- **Two passes per held-out mouse**, with the freeze's own session lists: pass A fine-tunes the
  subject embedding on the `fit` sessions and scores the `score` sessions; pass B swaps them.
  Every session is scored once, and pass A's scored half is the half MEvolve scores. Fine-tune
  settings stay study 01's (500 steps, lr 1e-3, one mouse at a time).
- **Checkpoint rule:** a fold model's checkpoint and early stop use only the eval sessions of its
  training mice; no held-out mouse is scored during training. The recipe already does this
  (wrapper `gru_trainer.py`: `eval_likelihood` on `bundle.eval_set`; per-checkpoint held-out eval
  is skipped for the multisubject GRU).
- **Seeds:** one per fold. Training noise comes from study 02's H256 seeds 1–2 of model 0
  (`-3c79d1d1`, `-a3171d8e`) on the test fold, at the cost of rescoring only.
- **Platform: Cedar** (Han, 2026-11-12), A100 cluster, launched with dispatcher `launch_cedar.py`
  (dispatcher #220): one session per fold model, 256Gi RAM (the `cedar-smoke` variant showed the
  recipe peaks near 190 GB), 48h lifetime until the first run gives a step rate. No automatic
  resume: a stopped run is restarted by hand in the same output folder.

- **What "held-out prediction" means (Han, 2026-11-12):** fit and score sessions alternate by
  date, so a mouse's embedding is tuned on sessions both before and after the one it predicts. The
  map measures how well the GRU predicts a session once it knows the mouse from its other
  sessions, not a forecast of later sessions. Study 01 and MEvolve use the same definition.
- **Known limitation (Han, 2026-11-12):** fold models 1–4 train on the 149 test mice, so the
  residuals and embeddings study 41 uses on discovery and confirmation mice carry a weak,
  population-level trace of the test mice. Test-mouse predictions come only from model 0, which
  never saw them. Kept to give every model about 610 training mice; study 41 should cite this next
  to any test-tier claim.
- **Reference numbers for study 41 (Han, 2026-11-12):** on the test mice, the headline GRU number
  is study 01's on the 146 shared test subjects (continuity with studies 01, 12, 13, 40); every
  trial-level comparison and residual uses model 0. Check 2 ties the two.

## Variants

- `test-fold-ref`: rescoring only (passes A and B) of model 0, plus seeds 1–2 for training noise.
- `pool-folds`: one launch, 4 Cedar sessions (one per pool fold), one W&B group.
- `cedar-smoke`: the first Cedar launch, a 300-step test of the platform (not a result).

## Checks before the map is frozen

1. **Model 0 reproduces study 02.** Pass A per-mouse scores equal study 02's held-out scores on
   the 146 test mice whose sessions match (788733, 778103 and 810758 differ: under `include`, an
   all-ignore session shifts study 02's alternating split).
2. **Left/right agrees with study 01.** Left/right likelihood on engaged trials over the 146
   shared test subjects lies within study 01's 3-seed spread (read from its committed JSON).
3. **Join.** Every predicted trial joins the freeze on `(session_id, trial)`; each pass's scored
   sessions equal the freeze's `score` (pass A) or `fit` (pass B) sessions.

## Output (confirmed with study 41)

- Per trial: `subject_id`, `session_id`, `trial`, p(left), p(right), p(ignore), fold, pass,
  whether the session was fine-tuned or scored, checkpoint digest.
- Per session and per mouse: summed log-likelihood and trial count, for left/right on engaged
  trials and for 3-way.
- Per mouse and pass: the fine-tuned subject embedding.
- `reference/folds.json`; fold-to-fold spread per tier (study 41 `reference/tiers.json`).
- Registered in the cross-study registry (#63, PR #67) with a reader and a `*_SOURCE` path.

**Report headline:** left/right likelihood on engaged trials, per mouse and per session; ignore
detection (PR-AUC) second; full 3-way likelihood in tables only.

## Prerequisites

1. **Wrapper change (needed for scoring, not for training):** the held-out fine-tune takes its fit/score split only from the
   `eval_every_n` alternation and cannot swap it, and its per-trial export
   (`test_trial_predictions.csv`, `build_binary_trial_predictions`) is 2-way only. Needed: an
   explicit per-session fit/score list with a swap, and 3-way per-trial export with the subject
   embedding.
2. Training on an explicit mouse list: dispatcher `mice_snapshot_scaling.yaml` `subject_ids`
   (bypasses the ranked split); held-out list: `test_subject_ids`. Both exist.
3. Study 02 seed 0 checkpoint: W&B artifact `gru-output-<run>:v0` (final checkpoint = best eval).
4. Registry #63 (PR #67) merged before registration.

## Done when (from #148)

- [x] Study scaffolded (PR #151).
- [x] Fold plan and recipe fixed before training (this plan).
- [x] `reference/folds.json` committed.
- [ ] Wrapper prerequisite merged (needed for scoring, not for training).
- [ ] Model 0 reproduces study 02 (check 1).
- [ ] All folds trained; per-trial predictions frozen and registered.
- [ ] Report: predictability per mouse and per session, and the spread across folds.
