# studies/15-behavioral-fingerprints/Makefile
#
# `all` regenerates every report OFFLINE from committed frozen inputs (CI runs it).
# One target per report (rN); one `freeze-*` target per freeze, never part of `all`.
# The freezes need /allen and a compute node; see each script's docstring for the srun line.

PY ?= python
MODELS ?= ../../../aind-dynamic-foraging-models
MEVOLVE ?= /allen/aind/scratch/han.hou/tmp/s15/mevolve-2eef359
MEVOLVE_PY ?= /allen/aind/scratch/han.hou/miniforge3/envs/mevolve/bin/python
.PHONY: all freeze-p0-golden-mevolve freeze-p0-golden-wrapper freeze-p0-task-checks

all: r1

freeze-p0-golden-mevolve:
	PYTHONPATH=$(MODELS)/src:$(MEVOLVE)/src $(MEVOLVE_PY) analysis/freeze_p0_golden_mevolve.py \
	  --models-repo $(MODELS) --mevolve-repo $(MEVOLVE)

freeze-p0-golden-wrapper:
	PYTHONPATH=$(MODELS)/src $(PY) analysis/freeze_p0_golden_wrapper.py --models-repo $(MODELS)

freeze-p0-task-checks:
	$(PY) analysis/freeze_p0_task_checks.py

# r1 = P0: the fingerprint library against the wrapper and MEvolve, and the task checks
.PHONY: r1
r1:
	$(PY) analysis/r1_p0_checks.py
