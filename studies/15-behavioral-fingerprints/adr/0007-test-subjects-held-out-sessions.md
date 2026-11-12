# Fingerprints use study 12's test subjects and their held-out sessions only

Every model's replay and rollout fingerprints, and the noise floor, use study 12's 149 test
subjects and only their held-out (odd) sessions. The 146 that are also on the GRU's held-out list
are marked. The grid then sits beside study 12 r3's leaderboard, study 13 and study 40's test
scores on the same subjects and sessions (decided with Han, 2026-11-03).

## Considered options

- **Study 01 r9's choice: the 614 training-pool subjects, all their sessions.** Four times the
  subjects, and continuity with r9. But half of those sessions are the ones every model was fit or
  trained on, which flatters replay in particular.
- **The 614 training-pool subjects, odd sessions only.** Every model has seen the subject but not
  the session. It is a possible later variant if the test subjects leave too many bins empty.

## Consequences

- The split-half floor divides only the odd sessions, so more of the 32 history bins and the 6+
  bins fall below 5 trials per subject than in r9; the grid will have more empty cells.
