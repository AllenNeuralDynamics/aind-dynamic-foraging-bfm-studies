# gru-adaptation-budget launch (HPC)

- W&B group: `gru-adaptation-budget@20261024-060709-hpc`
- W&B project: https://wandb.ai/AIND-disRNN/gru_cross_species_transfer
- SLURM arrays: `27495243` (one-cell validation, index 15), `27495250` (the other 29), `27495483`
  (rescue of indices 19-29); `aind`, 1 A100 / 8 CPU / 64 GiB per task
- Submitted: 2026-10-24 06:07 PT (validation), 06:12 PT (grid); settled 07:28 PT
- Status: success (30/30) after one rescue: 11 cells failed at the dataset download with
  HTTP 429 and reran on a byte-identical shared build (`hpc_gruk_rescue.json`)
- Supersedes: Beaker `01M5PRT2DQ0RJJ8YV1SV98EW45`, `01M5PRWVDKZSQG18N6AZK1H7GQ`,
  `01M5PRTKMSHQKWY97M3XG38KFB`, stopped before any task started (`hpc_gruk.json`)
- Key numbers: GRU - Q +0.057 (K=1) to +0.029 (K=4) bits/trial Grossman, +0.031 to +0.013 Chen
- Feeds reports: r7
