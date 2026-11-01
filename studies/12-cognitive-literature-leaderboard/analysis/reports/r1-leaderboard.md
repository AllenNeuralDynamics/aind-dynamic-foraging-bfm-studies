---
schema_version: 1
id: r1
slug: leaderboard
status: draft
authors:
- han
wandb_groups:
- per-subject-de@20261101-012148
- per-subject-de@20261101-071303
inputs:
  script: analysis/r1_leaderboard.py
  data: analysis/per_subject_fits.csv
  runs: analysis/per_subject_fits.json
  freeze: analysis/freeze_fits.py
  figure: analysis/fig_r1_leaderboard.svg
upstream:
- study: 01-gru-scaling-law
  report: r1
- study: 01-gru-scaling-law
  report: r2
- study: 01-gru-scaling-law
  report: r8
- study: 13-hb-subject-vs-gru-heldout
  report: r5
- study: 10-gru-reservoir-ablation
  report: r1
reproduce: make -C studies/12-cognitive-literature-leaderboard r1
tags:
  model: [cognitive]
  question: [comparison]
---
# r1 — Literature models on the held-out subjects: the leaderboard

**Question.** Fit per subject on our subjects, how well does each literature model predict
held-out sessions, and how far is the best one from the hierarchical cognitive baselines
(HB-CTT, HB-Hattori) and from the GRU?

**Method.** Each of the 20 literature models ([CONTEXT.md](../../CONTEXT.md)) gets one parameter
set per subject, fit by differential evolution on the subject's even sessions and scored on its
odd sessions (study 01 r8's protocol). A score pools over subjects: exp(Σ log p / Σ n). The
leaderboard ranks 19 of them (Win-Stay-Lose-Shift is a strawman: fit, but not ranked) on the 149
held-out subjects, the cohort that HB-CTT and HB-Hattori (study 13) were
scored on. The GRU's held-out list differs by 3 subjects each way (study 13 r2), so the paired
comparison with the GRU uses the 146 subjects on both lists.

## Leaderboard

<!-- BEGIN result-1 -->
![Leaderboard](../fig_r1_leaderboard.svg)

**a**, pooled held-out likelihood of each literature model (free parameters in brackets) on the 149 held-out subjects (1,013,172 scored trials); dashed lines: HB-CTT and HB-Hattori (study 13 r5, same subjects) and the GRU at D≈614 (study 01 r1, on its own held-out list). **b**, per-subject likelihood minus the GRU's (3-seed mean) on the 146 subjects on both lists; boxes are quartiles, whiskers 1.5 IQR. Models below 0.69 are drawn off scale (grey) with their value. One fit per subject and model, so no error bars.

| rank | literature model | params | held-out (149) | median subject | training pool (614) | shared (146) | GRU's list (149) | median − GRU | GRU higher |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Lopez-DoubleTrace | 6 | 0.7187 | 0.7344 | 0.7227 | 0.7181 | 0.7181 | −0.0086 | 99% |
| 2 | Lebedeva-PR | 5 | 0.7172 | 0.7332 | 0.7212 | 0.7166 | 0.7165 | −0.0095 | 100% |
| 3 | CompareToThreshold | 4 | 0.7170 | 0.7331 | 0.7210 | 0.7165 | 0.7164 | −0.0095 | 100% |
| 4 | Zid-HistoryKernel2-ForagingRL | 5 | 0.7163 | 0.7318 | 0.7203 | 0.7157 | 0.7156 | −0.0101 | 100% |
| 5 | Miller-RHG | 7 | 0.7160 | 0.7350 | 0.7209 | 0.7156 | 0.7155 | −0.0090 | 99% |
| 6 | Bari2019 | 5 | 0.7149 | 0.7311 | 0.7181 | 0.7144 | 0.7142 | −0.0113 | 100% |
| 7 | Grossman-MetaLearning | 7 | 0.7127 | 0.7281 | 0.7159 | 0.7122 | 0.7121 | −0.0146 | 100% |
| 8 | Hattori2019 | 5 | 0.7127 | 0.7285 | 0.7159 | 0.7121 | 0.7120 | −0.0136 | 100% |
| 9 | Chen-RLCK | 4 | 0.7124 | 0.7263 | 0.7163 | 0.7118 | 0.7118 | −0.0143 | 99% |
| 10 | Beron-RFLR | 3 | 0.7112 | 0.7298 | 0.7143 | 0.7107 | 0.7105 | −0.0139 | 100% |
| 11 | Eckstein-BI | 4 | 0.7100 | 0.7300 | 0.7132 | 0.7094 | 0.7093 | −0.0139 | 99% |
| 12 | Eckstein-RL | 4 | 0.7088 | 0.7265 | 0.7120 | 0.7082 | 0.7081 | −0.0173 | 100% |
| 13 | FeedbackDependent-RL+Bias+CK1 | 5 | 0.6969 | 0.7171 | 0.6999 | 0.6964 | 0.6962 | −0.0271 | 100% |
| 14 | Alsio-DualRateSticky-RL | 4 | 0.6961 | 0.7147 | 0.6993 | 0.6955 | 0.6954 | −0.0283 | 100% |
| 15 | LossCounting | 3 | 0.6698 | 0.6922 | 0.6671 | 0.6691 | 0.6692 | −0.0441 | 100% |
| 16 | ActorCritic | 4 | 0.6273 | 0.6588 | 0.6258 | 0.6279 | 0.6287 | −0.0703 | 100% |
| 17 | FeedbackDependent-RL+Bias | 4 | 0.5558 | 0.5525 | 0.5535 | 0.5559 | 0.5553 | −0.1811 | 100% |
| 18 | FeedbackDependent-RL | 3 | 0.5489 | 0.5414 | 0.5478 | 0.5490 | 0.5486 | −0.1861 | 100% |
| 19 | Rescorla-Wagner | 3 | 0.5422 | 0.5407 | 0.5395 | 0.5422 | 0.5419 | −0.1944 | 100% |
| | *HB-CTT (study 13)* | | 0.7225 | | | | | | |
| | *HB-Hattori (study 13)* | | 0.7213 | | | | | | |
| | *GRU D≈614 (study 01)* | | 0.7268† | | | 0.7268 | 0.7268 | | |

† On the GRU's own held-out list, which shares 146 of the 149 subjects. Shared subjects whose scored-trial count differs from the GRU's: 0. "GRU's list" pools each literature model over the GRU's own 149 test subjects; the 3 of them in our training pool are fit per subject like the rest, so the pool does not change their score.

Best literature model: **Lopez-DoubleTrace**, 0.7187, which is −0.0038 from HB-CTT and −0.0087 from the GRU on the shared subjects.

Fitted but not ranked: Win-Stay-Lose-Shift (strawman: deterministic policy, no lapse term); pooled held-out likelihood 0.0062. Its fits are in the freeze.
<!-- END result-1 -->

## Check: the re-fit reproduces study 01 r8

<!-- BEGIN result-2 -->
Per-subject held-out likelihood of the re-fit against study 01 r8's fit of the same model (seed 0, same data and protocol). Tolerance: |Δ| ≤ 0.0001 per subject.

| literature model | subjects | max abs Δ | over tolerance | pooled, re-fit | pooled, r8 |
|---|---:|---:|---:|---:|---:|
| Hattori2019 | 149 | 7.27e-07 | 0 | 0.7127 | 0.7127 |
| Bari2019 | 149 | 1.39e-06 | 0 | 0.7149 | 0.7149 |
| CompareToThreshold | 149 | 2.34e-04 | 1 | 0.7170 | 0.7170 |
<!-- END result-2 -->

## What it says

- **No literature model reaches the hierarchical baselines or the GRU.** The best,
  Lopez-DoubleTrace (0.7187), sits 0.0038 below HB-CTT and 0.0087 below the GRU on the shared
  subjects. The GRU beats every ranked model on 99–100% of subjects (panel b).
- **The top is crowded.** Five models fall within 0.003 of each other (Lopez-DoubleTrace,
  Lebedeva-PR, CompareToThreshold, Zid-HistoryKernel2, Miller-RHG). What they share: a learned
  value and a choice trace that decays over several trials (stickiness), whether explicit
  (Lopez's two traces, Lebedeva's perseveration) or implicit (CompareToThreshold's run term).
  Single fits, no error bars: within this cluster, the order is not a firm ranking.
- **Simple value learners without stickiness or a side bias fall to near chance.**
  Rescorla-Wagner, FeedbackDependent-RL and its +Bias form score 0.54–0.56. Adding the one-step
  choice kernel to FeedbackDependent-RL lifts it to 0.6969, the largest single gain on the board.
- **The re-fits reproduce study 01 r8.** Hattori2019 and Bari2019 match per subject to about
  1e-6. CompareToThreshold matches on 148 of 149 subjects. On subject 722989 the re-fit found a
  slightly better training optimum (log-likelihood −1886.09 vs −1886.51) with the threshold at its
  −1 bound, a difference of 2.3e-4 in held-out likelihood. The pooled values are identical at 4
  decimals.
- **The three-subject pool swap between the GRU's split and ours does not matter.** Scores on
  the GRU's own 149-subject list differ from those on the 146 shared subjects by at most 0.0008.
