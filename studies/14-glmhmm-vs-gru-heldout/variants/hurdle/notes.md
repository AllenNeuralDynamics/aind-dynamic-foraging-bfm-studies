# Variant `hurdle`

**What.** K = 1 … 10 with the `hurdle` emission: per state, an engagement GLM on unsigned history times a side GLM on signed history (`sweep_hpc.yaml`). L comes from
`lag-scan`; `model.n_lags=5` in the manifest is a placeholder to set before launching.
10 restarts per K > 1, each from the K = 1 weights plus noise.

**Why.** The held-out K curve against study 02's 3-way GRU. Hurdle maps one-to-one onto study 02's decomposition: the side GLM against LR_engaged, the engagement GLM against engage and ignore PR-AUC. K is chosen on
`glmhmm/select/eval_likelihood` (training mice), and the full held-out curve is reported.

**Launch** from the studies repository root, so `$PWD` is this checkout (`launch_hpc.py`
resolves a relative `--sweep-yaml` against the dispatcher, and writes the record to
`--output-dir`) (after `lag-scan` fixes L):

    python ../aind-dynamic-foraging-bfm-dispatcher/code/launch_hpc.py \
      --sweep-yaml "$PWD/studies/14-glmhmm-vs-gru-heldout/variants/hurdle/sweep_hpc.yaml" \
      --mode cpu --sbatch-extra "--cpus-per-task=8 --mem=64G --time=48:00:00" \
      --output-dir "$PWD/studies/14-glmhmm-vs-gru-heldout/variants/hurdle/launch_record/hurdle-k1-10" \
      --label hurdle-k1-10 --note "Study 14: hurdle GLM-HMM, K=1..10 at L=<from lag-scan>, held-out 3-way score vs study 02's GRU"

**Or on Beaker**, which needs the GLM-HMM image built first (wrapper `beaker/glmhmm/Dockerfile`,
AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper#111; see `experiment_beaker.yaml`). Same
grid, one non-preemptible single-GPU task per cell:

    python ../aind-dynamic-foraging-bfm-dispatcher/code/launch_beaker_resumable.py \
      --sweep      "$PWD/studies/14-glmhmm-vs-gru-heldout/variants/hurdle/sweep_beaker.yaml" \
      --experiment "$PWD/studies/14-glmhmm-vs-gru-heldout/variants/hurdle/experiment_beaker.yaml" \
      --workspace  ai1/aind-dynamic-foraging-foundation-model \
      --output-dir "$PWD/studies/14-glmhmm-vs-gru-heldout/variants/hurdle/launch_record/hurdle-k1-10" \
      --label hurdle-k1-10 --note "<why this run exists and what it should teach us>"

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group
`hurdle@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that
group.

**Status.** Launched 2026-10-27 21:29–21:30 PT as `hurdle@20261027-212905`: K = 1–10 at L = 10, 20
restarts, 10 workers on 32 cores and 96 GB, HPC CPU. Settled 2026-10-28 07:02 PT: all 10 tasks
completed, and K = 5–10 were killed after scoring by the memory-map limit
(AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper#119), so their scores are intact but their
per-trial posteriors were not written. The refit selection score peaks at K = 3. Results:
[`launch_record/hurdle-k1-10/results.md`](launch_record/hurdle-k1-10/results.md).
