# Launch record — refit-prior-r2

- **W&B project:** [`mice_glmhmm`](https://wandb.ai/AIND-disRNN/mice_glmhmm)
- **W&B group:** `refit-prior@20261101-130226`, sweep [`6mzqt91f`](https://wandb.ai/AIND-disRNN/mice_glmhmm/sweeps/6mzqt91f)
- **SLURM:** array 27565930, tasks 0–319 (one cell each, at most 30 at once); sweep stop job 27565931
- **Runtime refs:** studies `cb21327b77aa7423cb160ce7e01a67853a585130` (#83); dispatcher
  `ea2d6deaf308c0c04cde5b925f3444631415d754`; wrapper `f9ea87594215c4ce8dbb772de2b2af6968d38222`;
  foraging models `2ce1f700b445823b19e3b7033704eb54c5ef746e` (v0.28.0, installed from git in the
  shared CPU env at 13:01 PT, unchanged until the array drained).
- **Global fits:** round 1's, loaded read-only from
  `/allen/aind/scratch/han.hou/outputs/glmhmm_global_fits/study14-round1/` (`global_fits_manifest.json`).
- **Resources:** HPC CPU, 32 cores and 32 GB per cell; peak RSS 15 GB; 6–110 min per cell, growing with K.
- **Launched / settled:** submitted 2026-11-01 13:02 PT, first task started 13:02 PT / last task ended 2026-11-01 17:59 PT
- **Status:** all 320 tasks `COMPLETED`, all 320 runs finished with their scores. Every cell loaded
  the manifest's global fit (sha256) and reproduced round 1's zero-shot score (largest gap 6e-8),
  checked by `analysis/freeze_round2.py`.

**Result.** With the refit pooled toward the global fit, the selection score rises through K = 10
for both emissions; the rule picks K = 10, weight prior σ = 0.25 and transition strength c = 1000,
each the edge of the grid. The picked refit beats the global fit alone on held-out mice at every
K. Details in r2 (`analysis/reports/r2-refit-prior.md`).

**Feeds.** r2.
