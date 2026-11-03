# The headline is the gap as a multiple of a split-half noise floor

For each subject and statistic, the gap is the mean of |model − subject| over the statistic's
bins, weighted by the subject's own trial count in each bin, in replay and rollout alike; bins
with fewer than 5 of the subject's trials are dropped, as the wrapper does.
Gaps are averaged over subjects, so every subject counts once (subject-balanced). The headline is
the floor ratio: that gap divided by the noise floor.

The noise floor is the gap a perfect model would still have, because each subject has a finite
number of trials. It is measured from the data, not from a model: split each subject's sessions into
two random halves, compute the gap between the halves, and average over 20 splits. Each half has
half the trials, so the gap between halves is about twice the floor of the full data; the floor is
half of it. A rollout's own sampling noise adds about 5% with 10 rollouts per session. The same
floor serves replay and rollout (decided with Han, 2026-11-02 and 2026-11-03).

## Considered options

- **Study 01 r9's subject-mean correlation and RMSE.** Correlation is meaningless with 2 to 4 bins
  (r9 shows ±1), and neither says how close to the data's own limit a model is.
- **MEvolve's pooled trial-weighted gap.** Pooling lets subjects with many trials dominate.
- **A binomial floor**, computed from each bin's trial count. It is cheap, but it ignores
  session-to-session variation, so the floor is too low and every model looks worse than it is.
