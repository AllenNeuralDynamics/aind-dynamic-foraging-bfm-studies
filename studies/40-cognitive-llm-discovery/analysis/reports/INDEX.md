# 40-cognitive-llm-discovery — reports

| report | question | verdict |
|---|---|---|
| [r1 — Gate 4: program recovery on simulated mice](r1-gate4-program-recovery.md) | Does the fitter recover simulated parameters, and does a short search recover a planted program, with and without seeds? | Pass: Hattori2019 recovered; both searches reach the target (seeded at proposal 3, no seeds at 71) |
| [r2 — The first production search: curve, mechanisms and gains on unseen mice](r2-search-curve.md) | How far did the first production search climb, what did it converge on, and does the climb hold on unseen subjects? | +0.167 over Miller-RHG on validation, nearly all in the first 500 proposals; a shared core; the early gain holds on 100 monitor subjects, the late gain does not |
