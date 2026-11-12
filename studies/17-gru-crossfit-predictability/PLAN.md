# Plan — 17-gru-crossfit-predictability

Issue #148. Companions: study 41 (#149, the strategy study) and
[MEvolve #87](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-mevolve/issues/87)
(scientist mode). Status: **planning; no training until the open decisions below are settled.**

## Why

The study-01 reference GRU (v2 H128) trained on all 614 training-pool mice. Only the 149 test mice
have held-out GRU predictions, so every comparison with the GRU spends the test set (studies 01,
12, 13, 15 and 40 all scored there). Cross-validated folds give a held-out prediction for every
mouse. Study 41 uses the result as its ceiling and residual map: where an interpretable model
loses to the GRU, structure is missing. The spread across folds shows how stable that ceiling is.

## Agreed with Han (2026-11-12, in the study 40 session)

1. Study 17 exists and answers #148.
2. About 5 folds.
3. The existing 149-mouse test split stays one fold as it is, so earlier results stay comparable.
4. Output: per-trial held-out predictions for every mouse, plus per-session and per-mouse
   scores, frozen, keyed by digest, and registered as a shared reference (registry #63).

Nothing else is decided: no deadline, no GPU budget, no choice between Beaker and HPC.

## Mouse sets

| set | mice | source |
|---|---|---|
| test (held out by study 01) | 149 | MEvolve freeze `20260603` `sessions.parquet`, `pool == heldout` |
| training pool | 614 | same, `pool == train` |
| ↳ study 40 discovery cohorts | 296 | `studies/40-cognitive-llm-discovery/reference/cohorts_seed0_monitor.json` → `cohorts` |
| ↳ study 41 confirmation tier | 315 | study 41 `reference/tiers.json` (draft PR #150) |
| ↳ excluded (on the GRU's own test list) | 3 | same study-40 file → `excluded`: 764791, 808057, 823164 |

## Open decisions (for Han)

Each item gives the suggestion from the study 40 session or this session. None is decided.

**D1. Fold plan.** Suggested: the test fold reuses the study-01 run (trained on all 614; not
retrained), plus K new trainings over the training pool, each holding out one part of it.
- K: 4 parts (≈153 mice each, the same size as the test fold) or 5 (≈122 each)?
- Each new fold trains on about 460–490 mice, not 614. Study 01 found the held-out gain
  saturates by about 100 mice, so this is probably harmless, but it is a known difference
  between the test fold and the others.
- Alternative: retrain the test fold too, so all folds use the same code and wrapper SHA. Then it
  must match study 01's held-out score within seed noise.

**D2. How to split the pool.** Suggested: stratify as study 40 drew its cohorts (study 40
ADR 0003: by curriculum and by session-count thirds, fixed seed), and spread the discovery (296)
and confirmation (315) mice across all folds, so no model is scored on one tier only.

**D3. The 3 excluded mice.** Suggested: use their existing study-01 held-out predictions (to verify
against that run's inputs: the freeze puts them in the training pool, but they are on the GRU's test list, so that
GRU never trained on them) and keep them out of the new folds. Study 41 excludes them anyway.

**D4. Which sessions get scored.** This decides what "every session" means.
- Study 01's held-out protocol (`heldout_offline.yaml`, inherited `eval_every_n=2`) fine-tunes
  only the subject embedding on half of a held-out mouse's sessions (500 steps, lr 1e-3) and
  scores the other half. So one pass scores only half of each mouse's sessions.
- Options: (a) two passes per mouse with the halves swapped, so every session is scored once
  (suggested); (b) zero-shot, no fine-tuning (`heldout_fewshot_k0`), which scores every session
  but is a weaker ceiling; (c) keep one pass and accept half coverage.
- Study 01's per-subject score must be reproduced by the pass that matches its protocol.

**D5. Platform.** Beaker (hub clusters only) or HPC. Capacity check before more than 4 GPUs.

**D6. Free spread check (suggested).** The 149 test mice are held out from every fold model.
Scoring them with each new fold model costs only evaluation and gives fold-to-fold spread on
the same mice the reference scored.

## Output contract (to confirm with the [41] sessions)

- Per trial: `subject_id`, `session_id`, trial index, p(right) (and p(ignore) if 3-way), fold
  id, checkpoint digest.
- Per session and per mouse: summed log-likelihood and trial count (as MEvolve's 2-level
  `session_log_lik` and `session_n_trials`).
- Join key: MEvolve freeze `20260603` (`sessions.parquet` `session_id`, `trials.parquet` trial
  index). The wrapper surfaces the database `session_id` as `ses_idx`; check that it and the
  trial index line up with the freeze before freezing anything.

## Dependencies checked

- Training on an explicit mouse list: dispatcher data config `mice_snapshot_scaling.yaml` has
  `subject_ids` (bypasses the ranked split). Held-out scoring on an explicit list:
  `config_heldout_subject_finetuning.yaml` `test_subject_ids`. Both exist; no wrapper change is
  known to be needed, but the per-trial prediction export is not yet checked.
- Registration needs registry #63 (PR #67, open).

## Done when (from #148)

- [x] Study scaffolded (this PR).
- [ ] Fold plan and recipe fixed before training; the test fold reproduces study 01's held-out score.
- [ ] All folds trained; per-trial predictions frozen and registered.
- [ ] Report: predictability per mouse and per session, and the spread across folds.
