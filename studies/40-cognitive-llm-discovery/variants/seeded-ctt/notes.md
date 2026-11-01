# Variant `seeded-ctt`

**What.** The search starts from CompareToThreshold, the lab's best per-subject model on this split,
written as a program and checked against its original. Budget: half of the pilot's search budget.

**Why.** CogFunSearch found that seeding from the strongest published model gave the best programs
for rats and flies. Comparing with `base` shows what the head start is worth.

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group `seeded-ctt@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Status.** Not launched.
