# Variant `training-mice-pilot`

**What.** Post-hoc analysis of existing runs, no training and no W&B launch. GRU: study 02 `nxd-3way` H256 / D614 /
seed 0 (run `nxd-3way-20260707-075401-80a6d102`). GLM-HMM: study 14 groups `hurdle@20261027-212905` and
`reference@20261027-213006` (K = 2–10, L = 10), global parameters for every mouse. Data: 50 GRU-training mice
(stratified by curriculum, seed 30), their odd (eval) sessions, 378,624 trials. Readouts cross-validated by
mouse, 5 folds of 10.

**Why.** A cheap first test of whether the GRU latents carry the GLM-HMM state, before the held-out mice (which
need the GRU's embedding fine-tune redone offline).

**Launches.** None: the work ran as Slurm jobs on the `aind` partition (2026-11-07 / 08 PT) from the scripts in
`analysis/pilot/`, writing to `/allen/aind/scratch/han.hou/pilots/study30-probe/`. `analysis/freeze_r1.py` froze
those outputs with their sha256.

**Status.** Done; result in [r1](../../analysis/reports/r1-pilot-decoding.md).
