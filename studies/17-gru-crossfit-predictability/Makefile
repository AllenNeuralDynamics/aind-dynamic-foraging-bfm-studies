# studies/17-gru-crossfit-predictability/Makefile
#
# `all` regenerates every report OFFLINE from committed frozen inputs (CI runs it).
# One target per report (rN); one `freeze-*` target per freeze, never part of `all`.

PY ?= python
.PHONY: all freeze-folds

all:

# freeze-folds = the fold assignment (reference/folds.json) and the pool-folds sweep, from study 40's
# cohort file and the MEvolve data freeze on /allen (snapshot 20260603). Needs pandas: the
# mevolve-search env, on a compute node.
freeze-folds:
	$(PY) analysis/freeze_folds.py
