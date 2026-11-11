# Changelog — 30-gru-vs-glmhmm-latents

## 2026-11-11

- Variant `heldout-mice`: offline held-out fine-tune of study 02's GRU (`heldout_offline.yaml`,
  `run_hpc.sbatch`, `launch_hpc.py`), reproduction gate `check_reproduction.py`, W&B upload `upload_wandb.py`.
  PLAN.md records decision (b): the fine-tuned embeddings go to study 30's W&B project.
- PLAN.md: r1 leakage audit, the `heldout-mice` variant (steps, reproduction check, open W&B decision), the
  no-upload decision for r1's intermediates, later follow-ups.

## 2026-11-08

- Study created (issue #136).
- r1 (draft): training-mice pilot. Freeze `freeze_r1.py` of the pilot's scratch outputs (code in
  `analysis/pilot/`), offline producer `r1_pilot_decoding.py`.
