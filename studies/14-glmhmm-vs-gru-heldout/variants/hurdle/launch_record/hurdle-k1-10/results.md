# Launch record — hurdle-k1-10

- **W&B project:** [`mice_glmhmm`](https://wandb.ai/AIND-disRNN/mice_glmhmm)
- **W&B group:** `hurdle@20261027-212905`, sweep [`p4t3u2zf`](https://wandb.ai/AIND-disRNN/mice_glmhmm/sweeps/p4t3u2zf)
- **SLURM:** array 27518375, tasks 0–9
- **Runtime refs:** studies `41e85aab9d006dd5b62df32f586c538be5a40f00` (#53); dispatcher
  `f3109058e8106181917110789f1c11ddfb3f1ba1`; wrapper `5f5301c738898e1289060635e8b656222ba32912`;
  foraging models `cf38f45d38eddaade38434bd9dbecfa4a98f7388` (0.25.0, shared CPU env). Runs started
  after 23:44 PT on 2026-10-27 would record 0.26.0 (study 13's upgrade); every task started at
  21:30–21:31, and glm_hmm is identical in both.
- **Resources:** HPC CPU, 32 cores and 96 GB per cell; K = 1–10 at L = 10, 20 restarts, 10 workers
- **Launched / settled:** submitted 2026-10-27 21:29 PT, tasks started 21:30 PT / last task ended 2026-10-28 07:02 PT
- **Status:** all 10 tasks `COMPLETED`. K = 5–10 were killed after scoring by the kernel's memory-map limit (AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper#119): W&B marks them `crashed`, but every score is in their W&B summary and `fit.json`; only `heldout_trials.parquet` and the output artifact are missing, and r1's freeze recomputes the posteriors from `fit.json`.

| K | W&B run | end | run time (h) | selection score (refit) |
|---|---|---|---|---|
| 1 | [tqxqq6ek](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/tqxqq6ek) | finished | 0.1 | 0.62266 |
| 2 | [bax62c47](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/bax62c47) | finished | 0.3 | 0.62778 |
| 3 | [u4j7ukti](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/u4j7ukti) | finished | 0.4 | **0.62887** |
| 4 | [8crhqlqk](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/8crhqlqk) | finished | 1.3 | 0.62841 |
| 5 | [gpqv6sub](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/gpqv6sub) | crashed after scoring | 1.9 | 0.62782 |
| 6 | [nxvyn933](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/nxvyn933) | crashed after scoring | 2.9 | 0.62656 |
| 7 | [qqgkv2yu](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/qqgkv2yu) | crashed after scoring | 4.1 | 0.62616 |
| 8 | [xdpnnxs9](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/xdpnnxs9) | crashed after scoring | 5.2 | 0.62518 |
| 9 | [8r9ltf6k](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/8r9ltf6k) | crashed after scoring | 7.8 | 0.62431 |
| 10 | [3gknqru4](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/3gknqru4) | crashed after scoring | 9.5 | 0.62411 |

**Result.** The refit selection score peaks at K = 3. Scores under the other protocols, the GRU comparison and the states are in r1 (`analysis/reports/r1-round-1.md`, added by #69; planned until it merges).

**Feeds.** r1.
