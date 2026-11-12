# studies/41-cognitive-strategy-profiles/Makefile
#
# `all` regenerates every report OFFLINE from committed frozen inputs (CI runs it).
# One target per report (rN); one `freeze-*` target per freeze, never part of `all`.

PY ?= python
.PHONY: all freeze-tiers

all:

# freeze-tiers = the mouse tiers (reference/tiers.json) from study 40's cohort file and the MEvolve data
# freeze on /allen (snapshot 20260603). Needs pandas: the mevolve-search env, on a compute node.
freeze-tiers:
	$(PY) analysis/freeze_tiers.py
