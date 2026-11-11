# 40-cognitive-llm-discovery — reports

| report | question | verdict |
|---|---|---|
| [r1 — Gate 4: program recovery on simulated mice](r1-gate4-program-recovery.md) | Does the fitter recover simulated parameters, and does a short search recover a planted program, with and without seeds? | Pass: Hattori2019 recovered; both searches reach the target (seeded at proposal 3, no seeds at 71) |
| [r2 — The first production search: curve, mechanisms and gains on unseen mice](r2-search-curve.md) | How far did the first production search climb, what did it converge on, and does the climb hold on unseen subjects? | +0.167 over Miller-RHG on validation, nearly all in the first 500 proposals; a shared core; the early gain holds on 100 monitor subjects, the late gain does not |
| [r3 — The search's best on the selection and test subjects](r3-search-best.md) | Which top program is best on unseen mice, how much of the HB-CTT-to-GRU gap does it close on test, and does it behave like the mice? | `satiety_engagement` closes 0.59 of the gap (CI 0.53–0.64), half of it from the fitting regime; better mouse-average curves than Miller-RHG, but the GRU stays best on 8 of 9 |
| [r4 — Simplification at three floors: fit against complexity on the test subjects](r4-simplification.md) | How simple can the programs become while keeping their gain, and how do they score on test? | Effort 2,634 → 403 and 107 keeps 0.55 and 0.53 of the gap (search's best 0.59); the 75% program's experience-dependent habit matches the GRU's rollout curves; no 90% program |
