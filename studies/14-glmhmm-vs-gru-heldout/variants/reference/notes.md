# Variant `reference`

**What.** K = 1 … 10 with the `reference` emission: per state, one multinomial GLM over every regressor with ignore as the zero logit (Faeze's and `ssm`'s parameterisation) (`sweep_hpc.yaml`). L comes from
`lag-scan`; `model.n_lags=5` in the manifest is a placeholder to set before launching.
10 restarts per K > 1, each from the K = 1 weights plus noise.

**Why.** The held-out K curve against study 02's 3-way GRU. The reference parameterisation is the one prior work used. Run beside `hurdle` on the same grid, it says whether the emission's form, rather than K, moves the score. K is chosen on
`glmhmm/select/eval_likelihood` (training mice), and the full held-out curve is reported.

**Launch** (after `lag-scan` fixes L):

    python ../aind-dynamic-foraging-bfm-dispatcher/code/launch_hpc.py \
      --sweep-yaml studies/14-glmhmm-vs-gru-heldout/variants/reference/sweep_hpc.yaml \
      --mode cpu --sbatch-extra "--cpus-per-task=8 --mem=64G --time=48:00:00" \
      --label reference-k1-10 --note "Study 14: reference GLM-HMM, K=1..10 at L=<from lag-scan>, held-out 3-way score vs study 02's GRU"

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group
`reference@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that
group.

**Status.** Not launched.
