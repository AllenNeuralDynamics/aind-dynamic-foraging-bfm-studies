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
  numbers are reported beside it.

## Considered options

- **Both fit directions, averaged**, as CogFunSearch did: less noise at twice the cost. Revisited only
  if the timing gate shows validation noise too large to rank programs.
- **The GRU's held-out list as the test set**: the headline's floor, HB-CTT, was scored on the other list.
