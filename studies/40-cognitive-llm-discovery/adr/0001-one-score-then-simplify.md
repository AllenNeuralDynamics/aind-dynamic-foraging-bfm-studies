# Hill-climb one score, then simplify; behaviour is feedback, never score

The search maximizes one number: a program's pooled held-out likelihood per trial on the validation
subjects, the same axis the GRU ceiling and the cognitive baselines are measured on. A second stage
then minimizes code complexity while keeping the score above a fit floor, as DataDIVER did.
Behaviour is checked two ways, and never added to the score (decided with Han, 2026-10-31 and
2026-11-01):

- **Replay fingerprints**, cheap, for every program: on the subjects' real sessions, the program's
  predicted switch probability against the subjects' actual switch rate after the same history:
  after k consecutive unrewarded trials on the current side (k = 1 to 5, then 6 or more), by the last
  1 to 3 trials' choice and reward pattern, and by reward and run length. Each is summarized as the
  trial-weighted gap between predicted and observed rates, split by stage group and task variant, and
  goes into the feedback the LLM sees.
- **Rollout fingerprints**, costly, at pre-registration only: the program runs as an agent in the task
  simulator, and its switch curve by reward and run length (study 01's headline statistic) is compared
  with the subjects'.

Both gate pre-registration, by a rule fixed before any search (the **fingerprint gate**). On the
selection subjects, a candidate gets two numbers: its replay gap, the mean of the three replay
statistics' trial-weighted gaps, and its rollout gap, the trial-weighted gap of the rollout switch
curve. The reference is the literature model with the best 2-level EM validation score, the same
one the simplification floors start from (ADR 0005), computed the same way on the same subjects. A
candidate whose replay gap or rollout gap exceeds the reference's is skipped for the next candidate
(ADR 0007). Aggregates are compared rather than single bins, so noise in one sparse bin cannot reject
a program.

## Amendment (2026-11-08): the gate uses each mode's fingerprint score

The gate's statistics are now study 15's fingerprint scores, one per mode (study 15 ADR 0009): replay
over `losses`, `pattern`, `hazard` and `post_switch`; rollout over the same four and the `run_length`
distribution; each a geometric mean of gaps over the subjects' noise floor. A candidate passes if both
scores are no larger than the reference's. The rollout `hazard` curve alone barely separates good models
(study 15 r3), so Han chose more fingerprints, including those that differ (2026-11-08); fixed before any
selection subject was replayed or rolled out. Details in ADR 0008.

## Considered options

- **A weighted sum of likelihood and behavioural distances.** The weights would be arbitrary, summary
  statistics are easy to game, and rollouts for every candidate would multiply evaluation cost.
- **Likelihood only, behaviour checked after the search.** Simplest, but a program that fits choices
  while generating unrealistic behaviour would only be caught at the end.
- **Fingerprint gates during the search.** Rollouts for many programs would multiply the cost and
  invite gaming the gate.
