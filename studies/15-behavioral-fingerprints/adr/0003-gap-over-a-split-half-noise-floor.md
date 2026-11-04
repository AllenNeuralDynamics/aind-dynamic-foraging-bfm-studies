# The headline is the gap as a multiple of a split-half noise floor

**Status:** accepted; its replay floor is superseded by ADR 0008 (replay uses a binomial floor).

For each subject and statistic, the gap is the mean of |model − subject| over the statistic's
bins, weighted by the subject's own trial count in each bin, in replay and rollout alike; bins
with fewer than 5 of the subject's trials are dropped, as the wrapper does. A rollout may never
produce some of the subject's histories (Win-Stay-Lose-Shift never makes `S+O+S+`); those bins
have no rate and are left out of that model's gap, and the share of the subject's kept-bin
weight the rollout does visit, its coverage, is reported beside every floor ratio.
Gaps are averaged over subjects, so every subject counts once (subject-balanced). The headline is
the floor ratio: that gap divided by the noise floor.

The noise floor is a test–retest reference: how far the subject is from itself, given its finite
trials. It is measured from the data, not from a model: split each subject's sessions into two
random halves, compute the gap between the halves, and average over 20 splits. A half has fewer
trials than the whole, so each bin's difference is rescaled to the full data by
√(n_a·n_b)/(n_a + n_b), the binomial ratio of the two gaps (½ when the halves are equal; a fixed
½ overstated the floor by 70% on unequal sessions). A rollout's own sampling noise adds about 5%
with 10 rollouts per session. The same floor serves replay and rollout (decided with Han,
2026-11-02 and 2026-11-03).

## Considered options

- **Study 01 r9's subject-mean correlation and RMSE.** Correlation is meaningless with 2 to 4 bins
  (r9 shows ±1), and neither says how close to the data's own limit a model is.
- **MEvolve's pooled trial-weighted gap.** Pooling lets subjects with many trials dominate.
- **A binomial floor**, computed from each bin's trial count. It is cheap, but it ignores
  session-to-session variation, so for rollouts the floor is too low and every model looks worse
  than it is.

## Consequences

- The floor includes differences between the subject's sessions. A replayed model can predict
  those differences from each session's own history, so its replay gap can fall below the floor
  (a ratio under 1). Rollouts draw new sessions, so for them the floor is the fair reference.
  Replay therefore gets its own floor, the binomial one: ADR 0008.
