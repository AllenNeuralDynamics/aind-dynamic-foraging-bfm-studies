# Variant `refit-prior`

**What.** Round 2 of the K grid. The per-mouse refits use a Gaussian weight prior centred on
the global fit's weights (`model.subject_prior_center=global`) instead of on zero, at prior SD
σ ∈ {0.25, 0.5, 1, 2}, for K = 1–10 and both emissions: 80 cells (`sweep_hpc.yaml`).
- **No global fit runs.** Each cell loads round 1's global fit for its (emission, K) through
  `model.global_fit_dir` (AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper#121), so the
  global model, the zero-shot score and the data are round 1's. Only the refits and their
  scores change.
- **The saved fits** are a read-only copy of round 1's `fit.json` files (`hurdle@20261027-212905`,
  `reference@20261027-213006`) in
  `/allen/aind/scratch/han.hou/outputs/glmhmm_global_fits/study14-round1/`. Their sha256s, source
  runs and paths are in `global_fits_manifest.json` (a copy of that directory's
  `MANIFEST.json`); each equals the digest frozen in `analysis/frozen_round1.json`.
- **Everything else is round 1's:** snapshot 20260603, `ignore_policy=include`, L = 10, transition
  prior α = 2, the same split and scoring. The zero-centred σ = 2 arm is round 1 itself and is
  not rerun; the report reads it from `analysis/frozen_round1.json`.
- Needs `aind-dynamic-foraging-models` >= 0.27.0 (`Prior.mean`,
  AllenNeuralDynamics/aind-dynamic-foraging-models#89) in the runtime env.

**Why.** In round 1 (r1) the zero-centred refit scored *below* the global fit alone on held-out
mice from K = 4 on, and its selection score peaked at K = 3 (hurdle) and K = 2 (reference),
while the global fit alone kept improving to K = 10. That pattern says the refit overfits each
mouse as K grows. A prior centred on the global fit lets a mouse move only as far from the
population as its data require. The variant asks: does a centred refit stop losing to no
refit, and where does its K curve peak then?

**Selection rule (stated before launch).**
1. K, σ and the centre are chosen together by `glmhmm/select/eval_likelihood` (refits of the
   training mice, scored on their eval sessions), over this variant's 80 cells plus round 1's
   zero-centred σ = 2 cells. The held-out scores are reported for every cell and never used to
   choose.
2. Headline question: for each (emission, K), does the best centred cell's matched held-out
   score (`heldout/eval_likelihood`) exceed the zero-shot score (`heldout/zero_shot/eval_likelihood`)?
3. Small σ is close to "no refit for the weights", but each mouse's transitions are still
   refitted, so the small-σ score need not equal the zero-shot score or follow its curve. A
   later peak at small σ is partly the prior's doing; the report says so beside the curve.

**Checks.**
- The saved fits predate `glmhmm/global/train_data_sha256`, so the loader checks their data only
  by training-trial count (4,887,818) and warns, as designed in wrapper#121.
  `glmhmm/global/loaded_data_check` is `num_train_trials` in every cell.
- The real check that the right fit loaded: every cell's `heldout/zero_shot/eval_likelihood`
  equals round 1's for the same (emission, K) to within 1e-5, and `glmhmm/global/loaded_sha256`
  equals the manifest's digest. The report's freeze asserts both.
- Before the fan-out, a diagnostic smoke (W&B disabled, not a launch) ran three cells: hurdle
  K = 3 zero-centred σ = 2 (must reproduce round 1's scores), hurdle K = 3 centred σ = 1, and
  reference K = 10 centred σ = 0.25 (the slowest cell), to measure run time and memory.

**Launch** from the studies repository root, after the shared env has models >= 0.27.0 from git:

    python ../aind-dynamic-foraging-bfm-dispatcher/code/launch_hpc.py \
      --sweep-yaml "$PWD/studies/14-glmhmm-vs-gru-heldout/variants/refit-prior/sweep_hpc.yaml" \
      --wrapper-root <detached wrapper worktree at the launch SHA> \
      --mode cpu --sbatch-extra "--cpus-per-task=32 --mem=96G --time=24:00:00" \
      --output-dir "$PWD/studies/14-glmhmm-vs-gru-heldout/variants/refit-prior/launch_record/refit-prior-r2" \
      --label refit-prior-r2 --note "<why this run exists and what it should teach us>"

Resources and array concurrency are set from the smoke's measurements.

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group
`refit-prior@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join
that group.

**Status.** Not launched.
