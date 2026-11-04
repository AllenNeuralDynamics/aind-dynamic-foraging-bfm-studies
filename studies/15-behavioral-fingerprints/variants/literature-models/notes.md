# Variant `literature-models`

**What.** Replay and rollout fingerprints of study 12's 19 ranked literature models, per subject
and as 2-level EM (fitted parameters read through `studies/util/cross_study_literature.py`), of
Win-Stay-Lose-Shift as a strawman, and of a per-subject GLM (Su2022 design) fit here on the even
sessions.

**Why.** Study 12 ranks these models by held-out likelihood only. This variant shows where each
one's behaviour departs from the subjects', and whether the likelihood ranking holds on behaviour.

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group `literature-models@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Status.** Not launched.
