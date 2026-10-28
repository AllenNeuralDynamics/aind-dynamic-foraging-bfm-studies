# Sessions failing Pareto k̂ are rescored by importance sampling; the centred refit was rejected

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

**Refit.** The 46 subjects failing any gate (R-hat ≥ 1.01, bulk ESS < 400, or any divergence)
were refit once: centred session level, target acceptance 0.95, 3000 warmup + 4 × 3000 draws
(`hattori@20261027-181415`). It was worse on every gate: divergences rose from 304 to 11,007
(median 3 → 170 per subject), and subjects passing R-hat and ESS fell from 39 to 24; one subject
passed every gate. Divergences rising 36-fold under the centred form is a funnel's signature: the
sessions are weakly constrained, so the non-centred production geometry was the right one. The
refit is not used (record: `variants/hattori/launch_record/refit/README.md`). The untried remedy
is a non-centred refit at higher target acceptance; PLAN.md allowed one refit, so a second is a
new decision, not part of this one.

**Every subject keeps its production fit.** Its saved draws are scored again through the same
data path (`draws_path`), which reproduces the production plain scores exactly, then the
sessions failing k̂ are rescored (`hattori@20261027-181430` for the 103 subjects passing the
gates, `hattori@20261027-201810` for the other 46). The headline covers all 149; the
converged-only number beside it excludes the 46.

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
- The production headline is shown beside the final one in r1, so the effect of the rescue is
  read directly.
