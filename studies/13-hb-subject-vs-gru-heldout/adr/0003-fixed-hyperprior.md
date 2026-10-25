# A fixed weak hyperprior, the one `hattori2019_two_level` already uses

Each subject's subject-level hyperparameters get the same fixed prior: `mu_p ~ Normal(0, 1)` on
the unconstrained scale (uniform on each bounded parameter) and a log-normal prior on the
between-session spread, as in foraging-models' `hattori2019_two_level`. All of the forager's
parameters vary by session.

## Considered options

- **Empirical Bayes**, setting the hyperprior from the training subjects. Rejected: it pools
  across subjects again, which ADR 0001 removed on purpose.
- **Sue's original priors** (`stan_qLearning_5params.stan`, kept verbatim in foraging-models):
  a half-Cauchy(0.2) on the spread, and a bias that varies by session but has no subject-level
  mean. Rejected as the default: per `hattori2019_stan_reference`'s docstring,
  `hattori2019_two_level` differs from that reference in exactly these two ways, and one
  structure shared by Hattori and CTT keeps the two variants comparable. Recorded as a known deviation from the original approach.

## Consequences

With about five context sessions per subject, the posterior on the between-session spread may
be dominated by this prior. If it is, a prior-sensitivity variant follows before any verdict.
