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

Both gate pre-registration: a candidate whose replay or rollout fingerprints are worse than the best
published model's is skipped for the next candidate (ADR 0007).

## Considered options

- **A weighted sum of likelihood and behavioural distances.** The weights would be arbitrary, summary
  statistics are easy to game, and rollouts for every candidate would multiply evaluation cost.
- **Likelihood only, behaviour checked after the search.** Simplest, but a program that fits choices
  while generating unrealistic behaviour would only be caught at the end.
- **Fingerprint gates during the search.** Rollouts for many programs would multiply the cost and
  invite gaming the gate.
