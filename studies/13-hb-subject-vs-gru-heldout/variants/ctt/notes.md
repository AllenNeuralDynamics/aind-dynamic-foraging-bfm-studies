# Variant `ctt`

**What.** HB-CompareToThreshold in the same two-level structure, with the configuration of the per-subject MLE CTT baseline (`cross_study.rl_baseline("ctt")`): one learning rate, threshold, softmax inverse temperature, bias, no choice kernel. Needs a JAX CTT likelihood with a parity test in foraging-models first.

**Why.** CTT is the best per-subject MLE family, so this is the strongest classical baseline the subject-level approach can give against the GRU.

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group `ctt@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Status.** Done: production `ctt@20261028-003745` (149 subjects) and refit `ctt@20261028-020002` (42 subjects failing a gate, non-centred at target acceptance 0.95). Result: r5.
