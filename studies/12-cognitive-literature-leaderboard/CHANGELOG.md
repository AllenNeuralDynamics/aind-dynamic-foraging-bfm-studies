# Changelog — 12-cognitive-literature-leaderboard

## 2026-11-12

- r3 result-3 (`r3_fallback.json`), per model: how many held-out subjects' 2-level fits failed by MEvolve's
  per-mouse fallback rule (a 2-level score more than 0.005 nats per trial below the subject's per-subject
  score), and the score with those subjects scored by their per-subject fit. The headline stays the raw
  2-level score: Han judged raw more transparent. The atlas shows the fallback in its hover card
  (AllenNeuralDynamics/aind-dynamic-foraging-bfm-atlas#74). `r3_2level_em.json` is unchanged.

## 2026-11-03

- New launches log to W&B `NeuralDynamics/dfbfm_12-leaderboard` (studies #99). Runs already in `aind-dynamic-foraging-model` stay there.

## 2026-11-02

- r1 (per subject) and r3 (2-level EM, the headline ranking) merged (#78).
- Variant `ctt-credit` launched (`ctt-credit@20261102-132626`): the credit form of CompareToThreshold and the original, refit together on foraging-models v0.29.0. Frozen by `freeze_ctt_credit.py`; report r2.

## 2026-11-01

- Study created.
