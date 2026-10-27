# Variant `hurdle`

**What.** K = 1 … 10 with the `hurdle` emission: per state, an engagement GLM on unsigned history times a side GLM on signed history (`sweep_hpc.yaml`). L comes from
`lag-scan`; `model.n_lags=5` in the manifest is a placeholder to set before launching.
10 restarts per K > 1, each from the K = 1 weights plus noise.

**Why.** The held-out K curve against study 02's 3-way GRU. Hurdle maps one-to-one onto study 02's decomposition: the side GLM against LR_engaged, the engagement GLM against engage and ignore PR-AUC. K is chosen on
`glmhmm/select/eval_likelihood` (training mice), and the full held-out curve is reported.

**Launch** (after `lag-scan` fixes L):

    python ../aind-dynamic-foraging-bfm-dispatcher/code/launch_hpc.py \
      --sweep-yaml studies/14-glmhmm-vs-gru-heldout/variants/hurdle/sweep_hpc.yaml \
      --mode cpu --sbatch-extra "--cpus-per-task=8 --mem=64G --time=48:00:00" \
      --label hurdle-k1-10 --note "Study 14: hurdle GLM-HMM, K=1..10 at L=<from lag-scan>, held-out 3-way score vs study 02's GRU"

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group
`hurdle@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that
group.

**Status.** Not launched.
