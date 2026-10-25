# Variant `hattori`

**What.** Hattori2019 two-level model (`hattori2019_two_level` in foraging-models): mouse-level mean and between-session SD over all 5 session-level parameters, fixed weak hyperpriors, fit per held-out mouse on its conditioning sessions.

**Why.** Nests the per-mouse MLE Hattori (0.71267), which is this model with zero between-session SD, so the gap measures what session-to-session variability is worth.

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group `hattori@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Status.** Not launched.
