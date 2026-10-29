# Launch record — reference-k1-10

- **W&B project:** [`mice_glmhmm`](https://wandb.ai/AIND-disRNN/mice_glmhmm)
- **W&B group:** `reference@20261027-213006`, sweep [`51jf7e2d`](https://wandb.ai/AIND-disRNN/mice_glmhmm/sweeps/51jf7e2d)
- **SLURM:** array 27518378, tasks 0–9
- **Runtime refs:** studies `41e85aab9d006dd5b62df32f586c538be5a40f00` (#53); dispatcher
  `f3109058e8106181917110789f1c11ddfb3f1ba1`; wrapper `5f5301c738898e1289060635e8b656222ba32912`;
  foraging models `cf38f45d38eddaade38434bd9dbecfa4a98f7388` (0.25.0, shared CPU env). Runs started
  after 23:44 PT on 2026-10-27 would record 0.26.0 (study 13's upgrade); every task started at
  21:30–21:31, and glm_hmm is identical in both.
- **Resources:** HPC CPU, 32 cores and 96 GB per cell; K = 1–10 at L = 10, 20 restarts, 10 workers
- **Launched / settled:** submitted 2026-10-27 21:30 PT, tasks started 21:31 PT / last task ended 2026-10-28 20:18 PT
- **Status:** all 10 tasks `COMPLETED`. every run finished normally, with its artifacts.

| K | W&B run | end | run time (h) | selection score (refit) |
|---|---|---|---|---|
| 1 | [ku5cosnd](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/ku5cosnd) | finished | 0.1 | 0.62277 |
| 2 | [74wl3w23](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/74wl3w23) | finished | 1.2 | **0.62618** |
| 3 | [fhys1r3s](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/fhys1r3s) | finished | 1.4 | 0.62612 |
| 4 | [g934dzpw](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/g934dzpw) | finished | 2.6 | 0.62431 |
| 5 | [imhowo92](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/imhowo92) | finished | 4.4 | 0.62308 |
| 6 | [0omy97mc](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/0omy97mc) | finished | 7.8 | 0.62089 |
| 7 | [vf1y44zx](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/vf1y44zx) | finished | 9.4 | 0.61943 |
| 8 | [bxtozep8](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/bxtozep8) | finished | 12.9 | 0.61816 |
| 9 | [ku3c1uvb](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/ku3c1uvb) | finished | 16.9 | 0.61696 |
| 10 | [v8zjymez](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/v8zjymez) | finished | 22.7 | 0.61585 |

**Result.** The refit selection score peaks at K = 2. Scores under the other protocols, the GRU comparison and the states are in r1 (`analysis/reports/r1-round-1.md`, added by #69; planned until it merges).

**Feeds.** r1.
