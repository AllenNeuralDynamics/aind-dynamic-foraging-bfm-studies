# ctt-credit@20261102-132626 (production)

- W&B group: ctt-credit@20261102-132626
- W&B project: https://wandb.ai/NeuralDynamics/aind-dynamic-foraging-model (sweep [2f26d1qa](https://wandb.ai/NeuralDynamics/aind-dynamic-foraging-model/sweeps/2f26d1qa))
- Runs: CompareToThreshold-credit [vr3ba0lv](https://wandb.ai/NeuralDynamics/aind-dynamic-foraging-model/runs/vr3ba0lv), CompareToThreshold [fb053fom](https://wandb.ai/NeuralDynamics/aind-dynamic-foraging-model/runs/fb053fom)
- SLURM: array 27572010 (2 tasks, one 112-core 6238R node each, 400G), autostop 27572011
- Submitted: 2026-11-02 13:26 PT
- Settled: 2026-11-02 15:02 PT
- Status: complete (2/2 runs finished; 614 training-pool + 149 held-out subjects each)
- Wall time: CompareToThreshold-credit 1 h 36 min, CompareToThreshold 53 min (array tasks 0 and 1)
- Key numbers: pooled held-out likelihood on the 149 test subjects, 0.71705 (credit) and 0.71704 (CTT); see r2
- Feeds reports: [r2](../../../../analysis/reports/r2-ctt-credit.md), through `analysis/ctt_credit_fits.csv` (`make freeze-ctt-credit`)
- Notes: foraging-models v0.29.0 (4574b4b) was installed from git into the shared `dynamic-foraging-bfm-cpu` env for this launch; the env had 2ce1f70 (v0.28.0) before.
