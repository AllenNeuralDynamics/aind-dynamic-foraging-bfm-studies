# Variant `seeded-islands`

**What.** One search with a different seed program on each island (ADR 0006): a near-empty template
plus literature models from the seed library, chosen by a rule fixed in advance. The whole library
appears in the prompt pool as inspirations. Budget: the pilot's $450 search budget.

**Stops** when the budget runs out, or when 2,000 proposals pass without the best validation score
rising by more than one standard error of a paired difference between two programs (the standard
error is measured at the timing gate).

**Why.** The study's goal is the best interpretable model rather than a comparison of starting
points, and seeding from strong models from the literature gave CogFunSearch its best programs. Islands keep
the seeds' lineages apart long enough to develop, and migration lets their mechanisms combine.

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group `seeded-islands@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Status.** Not launched.
