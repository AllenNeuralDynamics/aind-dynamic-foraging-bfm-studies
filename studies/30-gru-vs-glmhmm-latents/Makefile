# studies/30-gru-vs-glmhmm-latents/Makefile
#
# `all` regenerates every report OFFLINE from committed frozen inputs (CI runs it).
# One target per report (rN); one `freeze-*` target per freeze, never part of `all`.

PY ?= python
.PHONY: all

all: r1

# Freeze r1's inputs from the pilot's scratch outputs (needs /allen: run on an HPC compute node)
.PHONY: freeze-r1
freeze-r1:
	$(PY) analysis/freeze_r1.py

# r1 = Pilot: GLM-HMM states decoded from GRU latents on training mice
.PHONY: r1
r1:
	$(PY) analysis/r1_pilot_decoding.py
