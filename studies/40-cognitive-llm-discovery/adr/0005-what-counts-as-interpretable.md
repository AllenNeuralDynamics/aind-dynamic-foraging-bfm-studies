# What counts as interpretable: hard limits during search, then simplification

During the search a program may have at most 8 parameters and 6 latent variables, all named; a
program with discrete states counts its state probabilities among its latent variables. Static checks
reject unused parameters and unused latent variables, and the feedback flags parameters stuck at a
bound or with negligible effect. After the search, the simplification stage minimizes Halstead effort
while keeping the score above a fit floor, reported beside the parameter and latent-variable counts.
There are three floors, as in DataDIVER: the simplified program keeps 90%, 75% or 50% of the gain from
the best literature model to the best discovered program, each floor a separate run; together they
trace a fit-versus-complexity curve. Both ends of the gain are validation scores computed by MEvolve's
evaluator at the session level, so they are fitted and scored the same way; the literature models'
session-level scores come from study 12, which computes them with the same evaluator (decided with Han, 2026-10-31 and 2026-11-01).

The literature models use 4 to 7 parameters. CogFunSearch's best rat program used 18, carried
latent variables it never read and parameters that fit to zero. At the session level every parameter
also varies from session to session, which adds freedom, so the limits are tighter than the 12 and 8
first planned.

## Considered options

- **Halstead difficulty**, as CogFunSearch used, or plain counts of parameters, latent variables and
  lines: Halstead effort, as DataDIVER used, combines size and difficulty in one deterministic number.
- **An LLM ranking pairs of programs by readability**, as CogFunSearch also tried: not deterministic;
  deferred.
