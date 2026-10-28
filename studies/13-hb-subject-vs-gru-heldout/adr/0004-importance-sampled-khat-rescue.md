# Sessions failing Pareto k̂ are rescored by importance sampling; failing subjects are refit once, centred

ADR 0002 headlined the plain draw-average and promised importance sampling "only if real sessions
fail the gate". In production they did: 262 of 2,138 scored sessions had k̂ ≥ 0.7 (none unjudged;
median failing k̂ 0.99, failing sessions a little longer than the rest, median 521 vs 481 trials).
A starved average is biased low, so those sessions are rescored. Separately, 46 of 149 subjects
failed the convergence gates (7 on R-hat/ESS, the rest on divergences alone), and PLAN.md allows
one refit.

**Rescore.** Each failing session's session-marginal is the integral of its likelihood against
the predictive distribution of a fresh session's parameters, `(1/D) Σ_d N(θ; mu_d, σ_d)` over
the subject's posterior draws. That mixture's density is exact, so the integral can be estimated
from any proposal. The proposal is a multivariate t (ν = 4) at the session's mode under
likelihood × mixture, scaled by the inflated inverse Hessian there, mixed defensively with the
mixture itself (20% of the draws), so every weight is at most likelihood / 0.2. Its k̂ is gated
like the plain one. Both estimators are unbiased for the same quantity, so the headline takes
the importance-sampled value for the sessions that failed and the plain value elsewhere; the
all-plain total is reported beside it (`*_plain`). Code: foraging-models'
`importance_sampled_session_marginal`; the wrapper applies it with `importance_rescore`.

**Refit.** Subjects failing any gate (R-hat ≥ 1.01, bulk ESS < 400, or any divergence) are
refit once: centred session level (the non-centred form diverged where sessions are well
constrained), target acceptance 0.95, 3000 warmup + 4 × 3000 draws, 12 session draws per
posterior draw (144,000 per session, as before). A subject still failing after the refit keeps
the refit and is reported as failing; there is no second refit.

**Passing subjects** keep their production fit: their saved draws are scored again through the
same data path (`draws_path`), which reproduces the production plain scores exactly, then the
failing sessions are rescored.

## Considered options

- **More plain draws.** Rejected: weight ESS grows only linearly with draws, and at k̂ near 1
  the estimate's variance is barely finite; tens of millions of draws per session would be
  needed.
- **PSIS-smoothing the plain weights.** Rejected: it trades the low bias for a different bias
  and would leave the score depending on the smoother rather than on the model.
- **One proposal per posterior draw.** Rejected: 6,000 mode searches per session for no gain over
  one proposal against the exact mixture.

## Consequences

- The headline is a mix of two unbiased estimators of the same integral. Its SE keeps the error
  from the finite posterior draws whole (the plain joint error over every session) and adds the
  importance-sampled sessions' proposal error in quadrature.
- The converged-only number and the headline are now computed after the refit, so they move
  together; the production numbers stay in r1's history.
