# Variant `test-fold-ref`

**What.** The test fold: the 149 held-out test mice, which study 01's reference GRU (v2 H128,
ratio 1.0, snapshot `20260603`) never trained on. Open choice (PLAN.md, D1): reuse the study-01
run as it is, or retrain the recipe here.

**Why.** It ties the map to every earlier held-out result. It must reproduce study 01's held-out
score (read from `studies/01-gru-scaling-law/analysis/per_subject_rows.json`, never retyped):
exactly if the run is reused, within seed noise if it is retrained.

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group `test-fold-ref@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Status.** Not launched.
