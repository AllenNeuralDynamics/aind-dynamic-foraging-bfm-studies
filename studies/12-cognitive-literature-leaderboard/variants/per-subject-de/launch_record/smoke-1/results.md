# per-subject-de@20261101-004845 (smoke, failed)

- W&B group: per-subject-de@20261101-004845
- W&B project: https://wandb.ai/NeuralDynamics/aind-dynamic-foraging-model (sweep [x6715703](https://wandb.ai/NeuralDynamics/aind-dynamic-foraging-model/sweeps/x6715703))
- SLURM: array 27565000 (2 tasks, 112-core 6238R nodes), autostop 27565001
- Submitted: 2026-11-01 00:48 PT
- Settled: 2026-11-01 00:50 PT
- Status: failed (2/2 runs died in Hydra composition; SLURM reports COMPLETED)
- Key numbers: none
- Feeds reports: none
- Notes: `launch_hpc.py` read the model only from `model=` in the command, so with the
  model swept as a parameter it injected `model.training.auto_heldout_finetune.output_root`,
  which the baseline_rl configs do not have. Fixed in [dispatcher PR 214](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/pull/214) (commit 6450af3),
  which this launch predates (it ran dispatcher 2619e18). Runs reached W&B (`8limngac` and
  its sibling) only as crashed stubs.
