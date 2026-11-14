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
