# Four cohorts, one fit direction, and a headline on the shared test subjects

Every subject is fitted on its even sessions and scored on its odd sessions, as every comparator was,
and four disjoint cohorts play four roles. Generation subjects feed the LLM's feedback; validation
subjects give the score that ranks programs; selection subjects are used once, at the end, to choose
the pre-registered programs; test subjects are scored once. Picking the best of thousands of programs
on the validation subjects inflates its score (the winner's curse), so a separate cohort chooses and
the test set only measures (decided with Han, 2026-10-31).

- **Sizes.** Generation and validation start at 48 subjects each and are resized after the timing gate
  so that one full evaluation stays under about three minutes; selection takes about 100 more.
- **Drawing.** All three come from the 614-subject training pool, stratified by curriculum and by
  session count in thirds, with a fixed seed, so each resembles the test subjects, which were chosen
  by session-count rank within curriculum.
- **Test subjects** are the 149 held-out subjects of the per-subject MLE and HB baselines. The GRU's
  held-out list differs by three subjects each way, so the headline is pooled over the 146 shared test
  subjects for every model, re-pooled from committed per-subject values; the published 149-subject
  numbers are reported beside it. The three subjects on the GRU's held-out list that sit in the
  training pool (`cross_study.gru_split`) are kept out of the generation, validation and selection
  cohorts, so the final programs can also be scored on the GRU's own 149 test subjects (decided
  2026-11-01).

## Amendment (2026-11-04): the seed-0 draft is adopted

The cohorts are the seed-0 draft, adopted as drawn: generation 48, validation 48, selection 100
(Han, 2026-11-04, after a second opinion; issue #103). The lists are in
[`reference/cohorts_seed0.json`](../reference/cohorts_seed0.json), sha256
`cbe2cd029611a713043afdd4312aedd5fa31d10473b04e9402134cb994c4d7cb`, the file gate 2 timed. It was
drawn by `mevolve.cohorts --seed 0` from the MEvolve freeze whose manifest has sha256
`cfea3d1e21f49ed7baa42f11e1fdda66fd5901c519a47773c08716e062e7eef0`, and MEvolve at 2d2e406 redraws the
same lists. They are never redrawn, so that no result can choose its own subjects. The seed order of
ADR 0006, ranked on these validation subjects, is final with them.

| check | result |
|---|---|
| composition against the 149 test subjects | curriculum shares within 2 points; sessions per subject (medians 26 to 28, test 27), choices per subject and task mix alike |
| size | the gap from HB-CTT to the GRU (`analysis/explainers/explainer_refs.json`) is about ten times the paired standard error between two models on the 48 validation subjects at gate 2 |
| difficulty | literature models score slightly higher on the validation subjects than on the test subjects, and rank the same |
| unknown stage | 34% of validation sessions have no recorded stage, against 26% of test sessions. Stage is not a stratum, and 19% of 1,000 random 48-subject draws are as high. A program that uses the stage input gets slightly less credit on validation than it would on test. |

Timing does not depend on these lists. One evaluation of the 96 generation and validation subjects
takes about 3 minutes for literature models with the current fitter (MEvolve ADR 0004, corrected
2026-11-04), and longer for the largest programs; EM iterations and workers set the pace, not the
cohorts.

## Amendment (2026-11-07): monitor subjects

A fourth cohort of 100 **monitor subjects** rescores programs the search has already found, to see
whether gains on the 48 validation subjects carry to subjects the search never used (Han, 2026-11-07;
issue #128). They are drawn by the same rule, after the other three: `mevolve.cohorts --seed 0` with
sizes generation 48, validation 48, selection 100, monitor 100 returns the first three exactly as in
`cohorts_seed0.json`, and 100 more from the rest of the pool, sharing no subject with them. The lists
are in [`reference/cohorts_seed0_monitor.json`](../reference/cohorts_seed0_monitor.json), sha256
`d4f442555667e53ba16d34cef558fb15847bf3cc00d24a23674e3ee160229241`, committed before any program was
scored on them. Monitor scores never feed back into the search, the simplification or the choice of
ADR 0007, which stays on the selection subjects; the test subjects stay untouched.

## Considered options

- **Both fit directions, averaged**, as CogFunSearch did: less noise at twice the cost. Revisited only
  if the timing gate shows validation noise too large to rank programs.
- **The GRU's held-out list as the test set**: the headline's floor, HB-CTT, was scored on the other list.
