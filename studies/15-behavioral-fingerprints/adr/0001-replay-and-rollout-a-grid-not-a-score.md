# Replay and rollout for every model, reported as a grid, never one score

Every model gets both kinds of fingerprint. A replay fingerprint runs the model on the subjects'
real choices and rewards and compares its predicted switch probability in each history bin with
the subjects' observed rate. A rollout fingerprint lets the fitted model play the task and
compares the statistics of its sessions with the subjects'. The two test different things: replay
checks the model's predictions on real histories, and rollout checks whether small errors per
trial add up over a session (a model 2% too sticky per trial makes runs much too long). A model can
pass one and fail the other, so neither replaces the other.

The result is a model × statistic grid of floor ratios (ADR 0003). The study does not combine the
statistics into one behavioural score, and its text names each model's worst statistic instead of
ranking models on behaviour (decided with Han, 2026-11-02 and 2026-11-03).

## Considered options

- **A weighted mean of the statistics.** Any weights would be arbitrary, and one number hides
  where a model fails, for example compare-to-threshold saturating near p(switch) 0.40 in study 01
  r9. Study 40 rejected a combined score for the same reason (its ADR 0001).
- **Replay only.** It is cheap, but it cannot see errors that compound over a session.
- **Rollout only**, as studies 01 and 05 did. It mixes a model's errors with simulator error and
  sampling noise, and it cannot say on which real histories the model is wrong.
