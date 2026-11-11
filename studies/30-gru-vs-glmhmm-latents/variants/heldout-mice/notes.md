# Variant `heldout-mice`

**What.** r1's decoding, repeated on the 149 held-out mice (snapshot 20260603, every 5th ranked
subject), which neither model trained on. Plan and steps: [`PLAN.md`](../../PLAN.md).

- **GRU.** Study 02's H256 / D614 / seed 0 3-way GRU
  ([`nxd-3way-20260707-075401-80a6d102`](https://wandb.ai/AIND-disRNN/mice_ignore_scaling/runs/nxd-3way-20260707-075401-80a6d102)),
  network frozen. One new embedding row per held-out mouse is trained for 500 steps (lr 10⁻³) on the
  mouse's even sessions; odd sessions are scored. This is study 02's own end-of-training held-out
  fine-tune, redone offline because the run's fine-tuned embeddings were never saved.
  `heldout_offline.yaml` is generated from the run's own config by the wrapper's
  `resume_heldout_beaker._build_finetune_config`. That script's `main()` resumes the study 02 run and
  overwrites its results, so it is never run.
- **Source checkpoint.** Artifact `gru-output-nxd-3way-20260707-075401-80a6d102:v0`, the same bytes
  r1 read (`step_90000/params.json` sha256 `4a0b8124…`). `best_eval` resolves to step 90000.
- **GLM-HMM target.** Each mouse's MAP refit from study 14 run `u4j7ukti` (hurdle K = 3), labels
  matched to the global states.

**Why.** r1 is an internal validation: the GRU and the GLM-HMM global fit had both seen r1's mice.
This variant is the external one. If the held-out within-session R² falls far below r1's, part of
r1's score came from the GRU's familiarity with the training mice.

**Reproduction gate.** The fine-tune must reproduce the run's logged `heldout/final/eval_likelihood`,
read from study 02's `analysis/frozen_runs.json` (`check_reproduction.py`). The fine-tune starts
from the mean embedding and uses full batches, so it is deterministic: tolerance 10⁻⁴. A miss stops
the variant. Retry first with the source run's wrapper (`4530d04`).

**W&B.** Decision (b) in `PLAN.md` (Han, 2026-11-11). The job itself logs nothing. After the gate
passes, `upload_wandb.py` logs one run to
[`NeuralDynamics/dfbfm_30-gru-vs-glmhmm-latents`](https://wandb.ai/NeuralDynamics/dfbfm_30-gru-vs-glmhmm-latents)
in group `heldout-mice@<launch_id>`, with the fine-tuned parameters, embeddings and per-trial
predictions as an artifact.

**Launches.** Each launch writes its record to `launch_record/<label>/` and mints one W&B group
`heldout-mice@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that
group.

**Status.** Not launched.
