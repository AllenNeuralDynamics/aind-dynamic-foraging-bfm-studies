# What counts as interpretable: hard limits during search, then simplification

During the search a program may have at most 8 parameters and 6 latent variables, all named; a
program with discrete states counts its state probabilities among its latent variables. Static checks
reject unused parameters and unused latent variables, and the feedback flags parameters stuck at a
bound or with negligible effect. After the search, the simplification stage minimizes Halstead effort
while keeping the score above a fit floor, reported beside the parameter and latent-variable counts
(decided with Han, 2026-10-31).

The lab's published models use 4 to 7 parameters. CogFunSearch's best rat program used 18, carried
latent variables it never read and parameters that fit to zero. At the session level every parameter
also varies from session to session, which adds freedom, so the limits are tighter than the 12 and 8
first planned.

## Considered options

- **Halstead difficulty**, as CogFunSearch used, or plain counts of parameters, latent variables and
  lines: Halstead effort, as DataDIVER used, combines size and difficulty in one deterministic number.
- **An LLM ranking pairs of programs by readability**, as CogFunSearch also tried: not deterministic;
  deferred.
