# studies/12-cognitive-literature-leaderboard/Makefile
#
# `all` regenerates every report OFFLINE from committed frozen inputs (CI runs it).
# One target per report (rN); one `freeze-*` target per freeze, never part of `all`.

PY ?= python
.PHONY: all freeze-fits

all:

# The only live read: per-subject tables of the pinned W&B group, from the run directories on
# /allen. Needs WANDB_API_KEY (the NeuralDynamics key); run on a compute node.
freeze-fits:
	$(PY) analysis/freeze_fits.py

# r1 = Literature models on the held-out subjects: the leaderboard
.PHONY: r1
r1:
	$(PY) analysis/r1_leaderboard.py
