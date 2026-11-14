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

**Status.**
- `first-cedar-launch` (2026-11-12 17:53 PT, group `cedar-smoke@20261112-175329`, run
  `cedar-smoke-20261112-175329-89c9c8f7`): **failed**. Session start, pinned SHAs, snapshot read,
  GPU and W&B logging worked; about a minute into training the container was killed for running
  out of memory (Cedar event RESTARTED, exit 137; peak sampled 45 GiB against the 64 GiB
  request). The restarted container does not resume the run. Session stopped.
- `second-cedar-launch` (2026-11-12 18:28 PT, group `cedar-smoke@20261112-182835`, run
  [`cedar-smoke-20261112-182835-21b19cb6`](https://wandb.ai/NeuralDynamics/dfbfm_17-gru-crossfit-predictability/runs/cedar-smoke-20261112-182835-21b19cb6)):
  `ram` 64Gi → 256Gi and `cpu` 8 → 16, as study 02 `nxd-3way` requested on Beaker. **Finished**
  (W&B state finished, 19:08 PT). Startup and compilation took about 15 min, then about
  2.5 min per 100 steps; post-training work took about 14 min at up to 178 GiB resident memory,
  so this recipe needs about 190 GB of host RAM. The 300-step numbers are not results.

**Learned.** Cedar runs the study 17 recipe end to end, including W&B logging through the
launcher. Real fold trainings there need 256Gi per session and resume after a container restart
(dispatcher #219, still open): a restarted container loses the run.
