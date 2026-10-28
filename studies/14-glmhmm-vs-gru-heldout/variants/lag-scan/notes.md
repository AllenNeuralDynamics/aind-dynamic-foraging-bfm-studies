# Variant `lag-scan`

**What.** K = 1 on the full training pool: history length L ∈ {3, 5, 7, 10} × emission
∈ {hurdle, reference}, 8 cells (`sweep_hpc.yaml`). Everything else is the study default
(`PLAN.md`). L = 7 is Faeze's choice in her `ssm` fit.
L stops at 10: Su2022 history-regression weights are near zero by lag 10, so longer histories buy little and cost compute.

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

**Or on Beaker**, which needs the GLM-HMM image built first (wrapper `beaker/glmhmm/Dockerfile`,
AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper#111; see `experiment_beaker.yaml`). Same
grid, one non-preemptible single-GPU task per cell:

    python ../aind-dynamic-foraging-bfm-dispatcher/code/launch_beaker_resumable.py \
      --sweep      "$PWD/studies/14-glmhmm-vs-gru-heldout/variants/lag-scan/sweep_beaker.yaml" \
      --experiment "$PWD/studies/14-glmhmm-vs-gru-heldout/variants/lag-scan/experiment_beaker.yaml" \
      --workspace  ai1/aind-dynamic-foraging-foundation-model \
      --output-dir "$PWD/studies/14-glmhmm-vs-gru-heldout/variants/lag-scan/launch_record/lag-scan" \
      --label lag-scan --note "<why this run exists and what it should teach us>"

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group
`lag-scan@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join
that group.

**Status.** Done. Launched 2026-10-27 as `lag-scan@20261027-200129`, after the smoke
`lag-scan@20261027-194921`. L = 10 maximises the selection score for both emissions, at the
scan's upper bound (`launch_record/lag-scan/results.md`).

**L = 15 check** (`sweep_hpc_l15.yaml`). The first launch peaked at its cap, L = 10, for both
emissions, so two K = 1 cells at L = 15 test whether the selection score turns down (overfit)
or keeps rising. The K grids were already launched at L = 10 and keep running. If L = 15 scores
higher, that is reported, and whether to rerun the grids at a longer L is decided then.
