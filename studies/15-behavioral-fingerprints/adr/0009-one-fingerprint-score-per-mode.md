# One fingerprint score per mode, for leaderboards, beside the grid

**Status:** accepted; amends ADR 0001 (the grid stays the primary result).

Each model gets one number per mode, its fingerprint score: the geometric mean of its floor ratios
over that mode's statistics. Replay uses the four rate statistics (losses, last 1–3 trials, hazard,
after a switch); rollout adds the run-length distribution. A score of 1 is the aggregate reference:
what a model at the floor on every statistic would score. It does not show that each statistic is
at its floor (ratios of 2 and 0.5 also give 1), which is why the grid stays beside it. Replay and rollout keep separate scores, never one mixed score, because
they test different things and a model can pass one and fail the other (ADR 0001). A leaderboard
can then plot held-out likelihood, replay score and rollout score side by side (decided with Han,
2026-11-04).

Every floor ratio is already a multiple of the subjects' own limit, so the statistics share one
scale and need no weights. A geometric mean suits ratios: twice the floor on one statistic and half
on another cancel. Reports still show the grid and name each model's worst statistic beside its
score.

## Considered options

- **The worst statistic's floor ratio.** Conservative and close to ADR 0001's wording, but one
  sparse statistic (after a switch has few trials per bin) can decide a model's rank.
- **One gap pooled over every bin of every statistic, over one pooled floor.** It weights bins by
  trials, so the history patterns would dominate: each trial counts once at every depth.
- **One score mixing replay and rollout.** It would hide the case P2 exists to find.

## Consequences

- The hazard and the 1-trial pattern partly measure the same thing, so local switching weighs a
  little more than its share. The score is a summary for ranking and demonstration; claims about
  where a model fails come from the grid.
- A model with no finite floor ratio on a statistic (a cell left out, ADR 0008's gate) has no score
  in that mode.
