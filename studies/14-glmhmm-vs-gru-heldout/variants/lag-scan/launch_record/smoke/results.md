# Launch record — lag-scan smoke

- **W&B project:** [`mice_glmhmm`](https://wandb.ai/AIND-disRNN/mice_glmhmm)
- **W&B group:** `lag-scan@20261027-194921`, sweep [`p1agwczg`](https://wandb.ai/AIND-disRNN/mice_glmhmm/sweeps/p1agwczg), run [`xl12oo0r`](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/xl12oo0r)
- **SLURM:** array 27516499, task 0 only (`--array=0-0 --agent-count 1`): the first grid
  point, hurdle at L = 3, on the full cohort
- **Runtime refs:** as `../lag-scan/results.md`
- **Resources:** HPC CPU, 8 cores, 64 GB; ran 9.5 min, 14 GB peak
- **Status:** success: `COMPLETED`, exit 0

**Validation.**
- The global fit saw the benchmark cohort exactly (614 subjects, 9,227 sessions, 4,887,818
  trials) and converged.
- Every expected key was logged: `glmhmm/select/*`, `heldout/zero_shot/*`,
  `heldout/final/*` (LR engaged, engage, engage-ignore PR-AUC, mature / non-mature), and the
  pooling pair, with 149 held-out subjects and the per-subject table. There was no
  `heldout/test_likelihood`, by design.
- Its artifacts were written and logged (`glmhmm-output-xl12oo0r`).
- Gap: `fit.json` records `models_git_sha: null`; the launch record holds the models commit.

**Feeds.** Infrastructure gate only; the full launch reran this cell.
