# studies/14-glmhmm-vs-gru-heldout/Makefile
#
# `all` regenerates every report OFFLINE from committed frozen inputs (CI runs it).
# One target per report (rN); one `freeze-*` target per freeze, never part of `all`.

PY ?= python
.PHONY: all

all: r1 r2 r3

# r1 = GLM-HMM round 1: held-out score vs K, and what the states are
.PHONY: r1
r1:
	$(PY) analysis/r1_round_1.py

.PHONY: freeze-round1
# Node-only (reads /allen and the snapshot): WRAPPER_ROOT and DISPATCHER_ROOT must point at the
# checkouts the runs used. Writes analysis/frozen_round1.json; never part of `all`.
freeze-round1:
	$(PY) analysis/freeze_round1.py

# r2 = Round 2: per-mouse refits pooled toward the global fit
.PHONY: r2
r2:
	$(PY) analysis/r2_refit_prior.py

.PHONY: freeze-round2
# Node-only (reads /allen): the refit-prior runs' fit.json and per-subject scores. Writes
# analysis/frozen_round2.json; never part of `all`.
freeze-round2:
	$(PY) analysis/freeze_round2.py

# r3 = How each state spreads across subjects
.PHONY: r3
r3:
	$(PY) analysis/r3_state_spread.py

.PHONY: freeze-r3
# Node-only (reads /allen and the snapshot), same WRAPPER_ROOT / DISPATCHER_ROOT as
# freeze-round1. Run as the SLURM array analysis/freeze_r3.sbatch (fit and score one K-grid run
# per task into R3_PARTS), then R3_MERGE=1 writes analysis/frozen_r3.json and
# analysis/frozen_r3_figure.npz (committed) and the archive's npz files (analysis/_cache_r3/).
freeze-r3:
	$(PY) analysis/freeze_r3.py

.PHONY: archive-r3
# Once per freeze: R3_PARTS and analysis/_cache_r3/ to the W&B artifact recorded in
# analysis/archives_r3.json (AIND-disRNN/mice_glmhmm); never part of `all`.
archive-r3:
	$(PY) analysis/archive_r3.py

.PHONY: r3-explorer
# The interactive explorer of every r3 setting, from the W&B archive (downloaded into
# analysis/_cache_r3/ when missing, checked against archives_r3.json). Writes the git-ignored
# analysis/interactive/r3_state_spread.html and r3_data/; never part of `all`.
r3-explorer:
	$(PY) analysis/explorer_r3.py
