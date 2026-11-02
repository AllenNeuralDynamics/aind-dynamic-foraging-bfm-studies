# Variant `refit-prior`

**What.** Round 2 of the K grid. Two pooling knobs of the per-mouse refits, crossed, for
K = 1–10 and both emissions: 2 × 10 × 4 × 4 = 320 cells (`sweep_hpc.yaml`).
- **Weights:** a Gaussian prior centred on the global fit's weights
  (`model.subject_prior_center=global`) instead of on zero, at SD σ ∈ {0.25, 0.5, 1, 2}.
- **Transitions:** `model.transition_strength_subject` = c ∈ {off, 10, 100, 1000}. Off is round 1's
  Dirichlet(α = 2) on each row, centred on uniform rows. A number c centres the rows on the global
  transition matrix: concentration 1 + c · A_global, i.e. c pseudo-transitions per row, so the
  update is (counts + c · A_global) / (N + c). A mouse has about 8,000 training trials
  (4,887,818 over 614 mice), about 800 per row at K = 10, so c = 1000 weighs the global row about
  as much as the mouse's data. AllenNeuralDynamics/aind-dynamic-foraging-models#93,
  AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper#123.
- At K = 1 the transition matrix is fixed at 1, so the four c values give identical cells.
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
  not rerun; the report reads it from `analysis/frozen_round1.json`, which studies#69 adds
  (a prerequisite of the round-2 report, not of the launch).
- Needs `aind-dynamic-foraging-models` with `Prior.mean` (#89, v0.27.0) and
  `Prior.transition_concentration` (#93) in the runtime env.

**Why.** In round 1 (r1) the zero-centred refit scored *below* the global fit alone on held-out
mice from K = 4 on, and its selection score peaked at K = 3 (hurdle) and K = 2 (reference),
while the global fit alone kept improving to K = 10. That pattern says the refit overfits each
mouse as K grows. A prior centred on the global fit lets a mouse move only as far from the
population as its data require. Round 1 centred neither knob: the weights were pulled toward
zero, and each mouse's K × K transition matrix was nearly free (90 values at K = 10). The variant
asks: does a refit centred on the global fit stop losing to no refit, which knob matters, and
where does its K curve peak then?

**Selection rule (stated before launch).**
1. K, σ, c and the weight centre are chosen together by `glmhmm/select/eval_likelihood` (refits
   of the training mice, scored on their eval sessions), over this variant's 320 cells plus
   round 1's zero-centred σ = 2 cells. The held-out scores are reported for every cell and never
   used to choose.
2. Headline question: for each (emission, K), does the best centred cell's matched held-out
   score (`heldout/eval_likelihood`) exceed the zero-shot score (`heldout/zero_shot/eval_likelihood`)?
3. Small σ with large c is close to "no refit": the weights and the transitions both stay near
   the global fit (the initial-state distribution is still refitted). A later peak there is
   partly the prior's doing; the report says so beside the curve.

**Checks.**
- The saved fits predate `glmhmm/global/train_data_sha256`, so the loader checks their data only
  by training-trial count (4,887,818) and warns, as designed in wrapper#121.
  `glmhmm/global/loaded_data_check` is `num_train_trials` in every cell.
- The real check that the right fit loaded: every cell's `heldout/zero_shot/eval_likelihood`
  equals round 1's for the same (emission, K) to within 1e-5, and `glmhmm/global/loaded_sha256`
  equals the manifest's digest. The report's freeze asserts both.
- **Diagnostic smoke** before the fan-out (2026-11-01; W&B disabled, not a launch; models and
  wrapper from checkouts on the PR branches, the shared env untouched). Selection and held-out
  scores, against round 1's zero-centred σ = 2 refit and the zero-shot score:

  | cell | selection | held-out matched | held-out zero-shot | wall | max RSS |
  |---|---|---|---|---|---|
  | hurdle K = 3, zero-centred σ = 2, c off (= round 1) | 0.628866 | 0.614779 | 0.613862 | 11 min | 16 GB |
  | hurdle K = 3, centred σ = 1, c off | 0.629706 | 0.615592 | 0.613862 | 11 min | 15 GB |
  | hurdle K = 3, centred σ = 1, c = 100 | 0.629662 | 0.615560 | 0.613862 | 10 min | 16 GB |
  | reference K = 10, centred σ = 0.25, c off | 0.632751 | 0.618102 | 0.618986 | 48 min | 15 GB |

  - The first row equals round 1's scores exactly (difference 0), and every cell's zero-shot
    score and loaded-file sha256 match round 1 and the manifest.
  - At reference K = 10, centring the weights lifts the held-out score from round 1's 0.6006 to
    0.6181, but it stays below the zero-shot 0.6190 with the transitions free; the c axis tests
    whether centring them closes that gap.
  - Resources: 32 cores and 32 GB per cell; about 10–50 min per cell, growing with K.

**Launch** from the studies repository root, after the shared env has a models release with #93, installed from git:

    python ../aind-dynamic-foraging-bfm-dispatcher/code/launch_hpc.py \
      --sweep-yaml "$PWD/studies/14-glmhmm-vs-gru-heldout/variants/refit-prior/sweep_hpc.yaml" \
      --wrapper-root <detached wrapper worktree at the launch SHA> \
      --mode cpu --sbatch-extra "--cpus-per-task=32 --mem=32G --time=04:00:00 --array=0-319%30" \
      --output-dir "$PWD/studies/14-glmhmm-vs-gru-heldout/variants/refit-prior/launch_record/refit-prior-r2" \
      --label refit-prior-r2 --note "<why this run exists and what it should teach us>"

One array task per cell (320 tasks, so the launcher sets AGENT_COUNT = 1), at most 30 at once;
the 4-hour limit then covers one cell, the slowest of which took 48 min. Lower or raise the `%30`
cap from the partition's free capacity at launch.

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group
`refit-prior@<launch_id>` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join
that group.

**Status.** Launched 2026-11-01 13:02 PT as `refit-prior@20261101-130226` (W&B sweep
[`6mzqt91f`](https://wandb.ai/AIND-disRNN/mice_glmhmm/sweeps/6mzqt91f), SLURM array 27565930):
320 cells, 30 at a time, HPC CPU, 32 cores and 32 GB each. Runtime: studies `cb21327`, wrapper
`f9ea875`, dispatcher `ea2d6de`, foraging models `2ce1f70` (v0.28.0, installed from git in the
shared CPU env). Settled 2026-11-01 17:59 PT: all 320 cells finished; results in
[`launch_record/refit-prior-r2/results.md`](launch_record/refit-prior-r2/results.md) and r2.
