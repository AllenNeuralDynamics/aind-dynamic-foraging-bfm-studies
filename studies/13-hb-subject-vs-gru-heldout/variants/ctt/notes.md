# Variant `ctt`

**What.** Compare-to-threshold forager in the same two-level structure, with the configuration behind the per-mouse MLE CTT baseline (0.71704). Needs a JAX CTT likelihood and parity test in foraging-models first.

**Why.** CTT is the best per-mouse MLE family, so this is the strongest classical baseline the subject-level approach can give against the GRU.

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group `ctt@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Status.** Not launched.
