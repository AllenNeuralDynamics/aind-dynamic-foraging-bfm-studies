# Refit (rejected)

Group [`hattori@20261027-181415`](https://wandb.ai/AIND-disRNN/mice_data_scaling/sweeps/e74d39z2),
SLURM array 27514168, launched 2026-10-27 18:14 PT. The 46 subjects failing a gate in production
were refit once: centred session level, target acceptance 0.95, 3000 warmup + 4 x 3000 draws.

It was worse on every gate, so it is not used; these subjects keep their production fits
(rescored in `../rescore-refit-subjects/`). Over the 46 subjects:

| | production | centred refit |
|---|---|---|
| divergences (total) | 304 | 11,007 |
| divergences (median per subject) | 3 | 169.5 |
| pass R-hat < 1.01 and bulk ESS >= 400 | 39 | 24 |
| pass every gate | 0 | 1 |

Divergences rising 36-fold under the centred form is the signature of a funnel: the sessions are
weakly constrained, so the non-centred production geometry was the right one. The untried remedy
is a non-centred refit at higher target acceptance (ADR 0004).
