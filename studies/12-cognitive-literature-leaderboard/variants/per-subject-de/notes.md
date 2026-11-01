# Variant `per-subject-de`

**What.** One W&B grid sweep over 19 literature models (`model=lit_<name>`, one dispatcher config
per model), one run per model. Each run fits all 763 subjects (614 training pool + 149
held-out) with differential evolution, one parameter set per subject, on even sessions, and
scores odd sessions; study 01 r8's protocol and data config. Sweep and launch record: TODO.

**Why.** The leaderboard of literature models on our subjects, for study 40's seed choice and
MEvolve's parity reference. The re-fit Hattori2019, Bari2019 and CompareToThreshold must
reproduce study 01 r8's per-subject values.

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group `per-subject-de@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Status.** Not launched.
