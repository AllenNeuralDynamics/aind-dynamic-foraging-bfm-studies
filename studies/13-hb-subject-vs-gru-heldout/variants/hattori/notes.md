# Variant `hattori`

**What.** HB-Hattori2019 as a two-level HB (`hattori2019_two_level` in foraging-models): subject-level mean and between-session spread over all 5 session-level parameters, under the fixed hyperprior, conditioned on each held-out subject's context sessions.

**Why.** Nests the per-subject MLE Hattori (`cross_study.rl_baseline("hattori")`), which is this model with zero between-session SD, so the gap measures what session-to-session variability is worth.

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group `hattori@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Status.** Not launched.
