# q-adaptation-budget launch

- W&B group: `q-adaptation-budget@20261024-035709`
- W&B project: https://wandb.ai/AIND-disRNN/gru_cross_species_transfer
- SLURM array: `27494714` (`aind`, CPU-only, 24 CPU / 64 GiB per task)
- Submitted: 2026-10-24 03:57 PT; settled 04:21 PT
- Status: success (8/8)
- Key numbers: Grossman -0.501 (K=1) to -0.443 (K=4) bits/trial; Chen -0.807 (K=1) to -0.771 (half);
  the half fits agree with q-matched-half to ~1e-6 in normalized likelihood (per-subject
  differences <= 1.4e-4 nats; identical trial keys), not bit-identically
- Feeds reports: r7
