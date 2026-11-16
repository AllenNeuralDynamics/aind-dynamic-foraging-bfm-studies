# Variant `half-refit`

**What.** Study 40 r5's 17 programs, each fitted per mouse with the 2-level Laplace-EM (study 40's
ADR 0008 settings), but on each half of a monitor mouse's sessions separately (odd and even sessions,
as in r5). Each half's parameters are then scored on the other half. Keeps per-mouse and per-session
parameters and per-trial predictions, which r5 did not.

**Why.** r5 scored both halves with one fitted parameter set per mouse, so a fit that happens to suit a
mouse counts as mouse-specific in both halves (r5's caveat). If the program × mouse interaction
survives separate fits, it reflects mechanism; if it shrinks, much of it was estimation. Discovery
tier only (monitor mice).

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group `half-refit@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Status.** Not launched. Design to be fixed on an issue before scoring.
