# Variant `base`

**What.** The search starts from a near-empty program template; the published models appear in
the prompt only as inspirations. Budget: half of the pilot's search budget.

**Why.** Tests discovery from scratch: whether the search reaches HB-CTT and beyond without being
handed the best known model.

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group `base@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Status.** Not launched.
