# Launch record — lag-scan

- **W&B project:** [`mice_glmhmm`](https://wandb.ai/AIND-disRNN/mice_glmhmm)
- **W&B group:** `lag-scan@20261027-200129`, sweep [`oc3seen8`](https://wandb.ai/AIND-disRNN/mice_glmhmm/sweeps/oc3seen8)
- **SLURM:** array 27516851, tasks 0–7. The script's default array of 12 was 4 tasks too
  many for the 8-point grid; tasks 8–11 were cancelled before they started.
- **Runtime refs:** studies `bdbdc536d2ec606bfad3cdd1f54a7664eafa61df`; dispatcher
  `90f9b6cfc2817a0f28b497f48666c769ffb36604`; wrapper `a4b018b978c3f9c7aff4c6be8b873d3f9eb87e43`;
  foraging models `13f7e9c3af6e80bcaeaae65a970a410e0539077d` (0.24.0, shared CPU env)
- **Resources:** HPC CPU, 8 cores and 64 GB per cell; each cell ran 6–10 min
- **Launched / settled:** 2026-10-27 20:01 PT / 20:12 PT
- **Status:** success: all 8 tasks `COMPLETED`, every global fit converged, 149 held-out
  subjects scored in each

**Result** (K = 1; `select` is `glmhmm/select/eval_likelihood`, the training-mouse refit
score that chooses L; the held-out columns are reported, never used to choose):

| emission | L | select | held-out matched | held-out zero-shot | LR engaged | engage | run |
|---|---|---|---|---|---|---|---|
| hurdle | 3 | 0.61520 | 0.60059 | 0.59015 | 0.71536 | 0.81215 | [d8dizic2](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/d8dizic2) |
| hurdle | 5 | 0.62013 | 0.60540 | 0.59588 | 0.71776 | 0.81619 | [s5zw7whi](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/s5zw7whi) |
| hurdle | 7 | 0.62175 | 0.60709 | 0.59804 | 0.71843 | 0.81779 | [l281kpdw](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/l281kpdw) |
| hurdle | 10 | **0.62266** | 0.60812 | 0.59968 | 0.71875 | 0.81885 | [he001yfr](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/he001yfr) |
| reference | 3 | 0.61641 | 0.60140 | 0.59024 | 0.71622 | 0.81237 | [s3hc2pdc](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/s3hc2pdc) |
| reference | 5 | 0.62100 | 0.60594 | 0.59587 | 0.71855 | 0.81610 | [jvbjwpx6](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/jvbjwpx6) |
| reference | 7 | 0.62232 | 0.60734 | 0.59798 | 0.71907 | 0.81747 | [cik2byvs](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/cik2byvs) |
| reference | 10 | **0.62277** | 0.60792 | 0.59957 | 0.71912 | 0.81819 | [8qlkfw01](https://wandb.ai/AIND-disRNN/mice_glmhmm/runs/8qlkfw01) |

- **L = 10 maximises the selection score for both emissions**, so by the variant's rule the K
  grids use L = 10.
- **It is the scan's upper bound**, and the gain is shrinking. From L = 7 to 10, select rises
  0.0009 (hurdle) and 0.0005 (reference), against 0.0049 / 0.0046 from L = 3 to 5. The cap
  at 10 stands (`notes.md`).
- **The emissions are close at K = 1:** reference is ahead by 0.0001–0.0012 on select, with
  twice the weights.
- Numbers were read from the runs' synced `wandb-summary.json`. No report reads them yet:
  r1 will, through a producer.

**Feeds.** Fixes L for `hurdle` and `reference`. Also the K = 1 row of r1.
