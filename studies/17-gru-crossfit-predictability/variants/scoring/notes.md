# Variant `scoring`

**What.** Held-out scoring of every study-17 model (PLAN.md, "Design"): for each model, its held-out
fold's mice are fine-tuned (subject embedding only, 500 steps, lr 1e-3, as studies 01/02) on one half
of their sessions and scored on the other, with the freeze's fit/score roles (`reference/roles.csv`):
pass A tunes on `fit` and scores `score`, pass B swaps. Tool: wrapper
`post_training_analysis.heldout_rescore` (wrapper PR #128, merged as `e863ce3`), which reproduced
study 02's held-out scores for model 0 to 1.7e-5 per mouse (check 1).

**Checkpoint and lambda.** Each model is scored at its `best_eval` checkpoint and at that
checkpoint's own session lambda (Han, 2026-11-13). Every D=10 and D=30 model's best_eval checkpoint
has lambda 0 or 0.5, so those 10 models are also scored at lambda 1 (the study 01/02 way) for the
comparison: 20 extra jobs.

**Files.**
- `models.yaml`: the 28 models (source W&B run, held-out fold, excluded mice). The 4 pool-fold
  2-way models are added when they finish training.
- `prepare.py`: downloads the models to `/allen/aind/scratch/han.hou/s17_scoring/models/`, writes
  the held-out mouse lists and `jobs.csv` (76 rows: 28 models x 2 passes + 10 x 2 at lambda 1).
- `score.slurm`: one array task per `jobs.csv` row, HPC CPU (56 CPUs, 120G, 10 h).

**Why HPC CPU.** One scoring job on CPU takes ~3.5 h with 56 CPUs and ~65 GB (check 1). The full
held-out batch is as large as a training batch, which ran out of GPU memory on a 40 GB A100; the
Cedar H200 node had no free GPU; HPC had ~60 free 56-CPU slots (2026-11-14 03:30 PT).

## Launches
- `score-28` (2026-11-14 03:49 PT, group `scoring@20261114-034950`, SLURM array 27839436, 76 tasks,
  at most 39 at once): the 28 finished models. Wrapper `e863ce3` from the detached worktree
  `~/code/wt/17-score`, conda env `dynamic-foraging-bfm-cpu` (its foraging-models is `4574b4b`;
  the GRU path does not import it). Outputs: `/allen/aind/scratch/han.hou/s17_scoring/out/<model>/pass<A|B>_lam-<mode>/`
  plus one W&B run each in group `scoring@20261114-034950`. Record: `launch_record/score-28/hpc_launch.json`.
- 04:40 PT, `score-28`: **for a lambda-0 checkpoint, lambda 1 changes nothing.** At step 20000 of
  `test-3way-d10` (lambda 0) the session-delta output layer (`session_delta_out` w and b) is exactly
  zero: it starts at zero and gets no gradient while lambda is 0, and the held-out fine-tune tunes
  only subject embeddings. The session delta is then 0 at any lambda, so scoring at lambda 0 and at
  lambda 1 is identical by construction (tasks 6 and 8 log the same losses). So study 01's small-D
  held-out scores, made at lambda 1, are unaffected. Only the two lambda-0.5 checkpoints
  (`pool1-3way-d30`, `pool3-3way-d30`) can differ. Cancelled the 10 not-yet-started lambda-1 tasks
  of lambda-0 models (42, 43, 46, 47, 54, 55, 66, 67, 70, 71); the 6 already running (8, 9, 12, 13,
  30, 31) are kept as the empirical check, and tasks 34, 35, 58, 59 (lambda 0.5) run as planned.
  (The fine-tune log line "Session Curriculum Lambda: 1.000" is the trainer's runtime value; the
  network is built with the requested lambda.)
- `score-2way-124` (2026-11-14 08:19 PT, group `scoring@20261114-081940`, SLURM array 27839669, 6
  tasks): the 2-way pool-fold models 1, 2 and 4 (best_eval step 90000, lambda 1). Same wrapper,
  env and script as `score-28`; record written by `write_record.py`. Fold 3 follows when it finishes.
- `score-2way-3` (2026-11-14, group `scoring@20261114-092226`, SLURM array 27839720, 2 tasks): the 2-way
  pool-fold 3 model (best_eval step 90000, lambda 1), launched when it finished training.
