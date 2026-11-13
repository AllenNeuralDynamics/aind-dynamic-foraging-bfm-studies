# Variant `cedar-smoke`

**What.** The first launch through the Cedar launcher (dispatcher `launch_cedar.py`, #219): one
A100 session running study 02's `nxd-3way` recipe at H128 (3-way), cut to 300 steps, session
conditioning on from step 200, held-out fine-tune off, default subject selection (the 614 pool
mice). Image `dfbfm-wrapper:20261109-wf390d4f`, cached on the node.

**Why.** Tests Cedar end to end before study 17's real fold trainings could use it: session
start, pinned SHAs, data snapshot read, GPU training, checkpoints in `$CEDAR_OUTPUTS_DIR`, W&B
logging (secret → `WANDB_API_KEY` → this study's project, group and `meta.*`), and the
launch-record v2 `platform: cedar` record (studies #153). Not a result; no report reads it.

**Launches.** Each launch writes its record to `launch_record/<label>/` and mints one W&B group `cedar-smoke@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Status.** Not launched.
