# studies/17-gru-crossfit-predictability/Makefile
#
# `all` regenerates every report OFFLINE from committed frozen inputs (CI runs it).
# One target per report (rN); one `freeze-*` target per freeze, never part of `all`.

PY ?= python
.PHONY: all freeze-folds freeze-scaling freeze-roles freeze-scoring

all:

# freeze-folds = the fold assignment (reference/folds.json) and the pool-folds sweep, from study 40's
# cohort file and the MEvolve data freeze on /allen (snapshot 20260603). Needs pandas: the
# mevolve-search env, on a compute node.
freeze-folds:
	$(PY) analysis/freeze_folds.py

# freeze-scaling = the nested training-mouse subsets of the scaling curve
# (reference/scaling_subsets.json) and the scaling-folds sweeps, from reference/folds.json and the
# freeze's curricula. Needs pandas: the mevolve-search env, on a compute node.
freeze-scaling:
	$(PY) analysis/freeze_scaling_subsets.py

# freeze-roles = the fit/score role of every session (reference/roles.csv + roles.json), the table
# the held-out scoring reads. From the MEvolve data freeze on /allen (snapshot 20260603), checked
# against the digest in reference/folds.json. Needs pandas: the mevolve-search env, on a compute node.
freeze-roles:
	$(PY) analysis/freeze_roles.py

# freeze-scoring = per-mouse held-out scores of every scored model (analysis/scores.json), from the
# variant `scoring` outputs on /allen scratch. Needs pandas: the mevolve-search env, on a compute node.
freeze-scoring:
	$(PY) analysis/freeze_scoring.py
