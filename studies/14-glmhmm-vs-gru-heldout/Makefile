# studies/14-glmhmm-vs-gru-heldout/Makefile
#
# `all` regenerates every report OFFLINE from committed frozen inputs (CI runs it).
# One target per report (rN); one `freeze-*` target per freeze, never part of `all`.

PY ?= python
.PHONY: all

all:

# r1 = GLM-HMM round 1: held-out score vs K, and what the states are
.PHONY: r1
r1:
	$(PY) analysis/r1_round_1.py

.PHONY: freeze-round1
# Node-only (reads /allen and the snapshot): WRAPPER_ROOT and DISPATCHER_ROOT must point at the
# checkouts the runs used. Writes analysis/frozen_round1.json; never part of `all`.
freeze-round1:
	$(PY) analysis/freeze_round1.py
