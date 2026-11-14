# studies/12-cognitive-literature-leaderboard/Makefile
#
# `all` regenerates every report OFFLINE from committed frozen inputs (CI runs it).
# One target per report (rN); one `freeze-*` target per freeze, never part of `all`.

PY ?= python
.PHONY: all freeze-fits

all: r1 r2 r3

# The only live read: per-subject tables of the pinned W&B group, from the run directories on
# /allen. Needs WANDB_API_KEY (the NeuralDynamics key); run on a compute node.
freeze-fits:
	$(PY) analysis/freeze_fits.py

# 2-level EM rows from MEvolve's score CLI outputs on /allen; run on a compute node.
.PHONY: freeze-two-level-em
freeze-two-level-em:
	$(PY) analysis/freeze_two_level_em.py

# The ctt-credit group's two runs (credit form and CompareToThreshold), from /allen; compute node.
.PHONY: freeze-ctt-credit
freeze-ctt-credit:
	$(PY) analysis/freeze_ctt_credit.py

# r1 = Literature models on the held-out subjects: the leaderboard
.PHONY: r1
r1:
	$(PY) analysis/r1_leaderboard.py

# r2 = CompareToThreshold in stay-credit coordinates vs the original
.PHONY: r2
r2:
	$(PY) analysis/r2_ctt_credit.py

# r3 = Literature models as 2-level EM (session → subject, Laplace-EM; regime 2)
.PHONY: r3
r3:
	$(PY) analysis/r3_2level_em.py
	$(PY) analysis/r3_fallback.py
