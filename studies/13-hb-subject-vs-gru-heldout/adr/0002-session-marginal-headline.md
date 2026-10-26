# The headline score is the session-marginal likelihood, not study 08's trial-marginal

A held-out session's session-level parameters are unseen, so the score must average over them.
Study 08 averaged each trial's choice probability over draws and then multiplied across trials
(trial-marginal). That lets every trial use a different session's parameters: it is not the
predictive probability of any session the model could generate, and its bias has no fixed sign.
Because it cannot learn within a session which parameters fit, it most likely understates the
model when draws that fit one trial well also fit the rest, which is the usual case. We
average the whole session's likelihood over draws instead (session-marginal, a log-mean-exp
per session). That equals predicting trial by trial while updating belief about the session's
parameters as trials arrive (chain rule), so each trial is predicted from that session's
earlier trials only. The GRU's hidden state also uses only earlier trials, though not by
Bayesian inference, so the two models have the same information, which makes this the fair
comparison. The trial-marginal is still reported, beside it, for continuity
with 08.

## Consequences

- The estimator is a plain Monte Carlo average over draws, which can starve: on long sessions
  a few draws carry the weight, and a starved average is biased low. Measured on
  model-generated 500-trial sessions, the weight ESS grows in proportion to the draws (median
  14 → 208 → 831 at 400 → 6,400 → 25,600 draws) and the pooled score is stable from 400 draws,
  so the headline uses plain draw-averaging with many session draws per posterior draw
  (default 64), not importance sampling. Each session is gated on its Pareto k̂ (< 0.7;
  Vehtari et al. 2024), used as a diagnostic only: the score and its SE stay unsmoothed.
  Importance sampling with a Laplace proposal is built only if real sessions fail the gate.
- The scorer is `batched_heldout_scores` (foraging-models#78), with its SE and pooled error.
