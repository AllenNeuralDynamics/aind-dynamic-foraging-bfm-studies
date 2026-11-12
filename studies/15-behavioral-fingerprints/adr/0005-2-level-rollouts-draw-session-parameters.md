# 2-level EM fits roll out with session parameters drawn from the subject's distribution

A model fit as 2-level EM gives each subject a distribution of session parameters, not one set.
Each rollout of a scored session draws its own parameter set from that distribution. This is the
quantity the session marginal scores, so rollout and score describe the same model, and it keeps
the real spread between sessions. Replay uses the session marginal's predicted probabilities, as
MEvolve's evaluator does (decided with Han, 2026-11-03).

## Considered options

- **The subject's typical values.** Every session then gets the same parameters, so rollouts show
  less session-to-session variation than the subjects do.
- **Each session's MAP parameters.** Held-out sessions are not fit, so they have none.
