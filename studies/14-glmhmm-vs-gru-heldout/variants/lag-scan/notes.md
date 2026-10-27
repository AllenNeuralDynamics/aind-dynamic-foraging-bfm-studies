# Variant `lag-scan`

**What.** K = 1 on the full training pool: history length L ∈ {3, 5, 7, 10, 15} × emission
∈ {hurdle, reference}, 10 cells (`sweep_hpc.yaml`). Everything else is the study default
(`PLAN.md`). L = 7 is Faeze's choice in her `ssm` fit.

**Why.** To fix L for the `hurdle` and `reference` K grids, on
`glmhmm/select/eval_likelihood`, the per-mouse refit score of the training mice. Held-out keys
are logged but never used to choose. At K = 1 the hurdle model is two history logistic
regressions, so this is also the history-GLM baseline row of the K curve.

**Launch** from the studies repository root, so `$PWD` is this checkout (`launch_hpc.py`
resolves a relative `--sweep-yaml` against the dispatcher, and writes the record to
`--output-dir`) (after the code PRs merge and the models package is released):

    python ../aind-dynamic-foraging-bfm-dispatcher/code/launch_hpc.py \
      --sweep-yaml "$PWD/studies/14-glmhmm-vs-gru-heldout/variants/lag-scan/sweep_hpc.yaml" \
      --mode cpu --sbatch-extra "--cpus-per-task=8 --mem=64G --time=48:00:00" \
      --output-dir "$PWD/studies/14-glmhmm-vs-gru-heldout/variants/lag-scan/launch_record/lag-scan" \
      --label lag-scan --note "Study 14: K=1 GLM-HMM over L x emission to fix the history length for the K grids, chosen on the training-mouse refit score"

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group
`lag-scan@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join
that group.

**Status.** Not launched.
