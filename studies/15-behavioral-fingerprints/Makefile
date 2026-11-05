# studies/15-behavioral-fingerprints/Makefile
#
# `all` regenerates every report OFFLINE from committed frozen inputs (CI runs it).
# One target per report (rN); one `freeze-*` target per freeze, never part of `all`.
# The freezes need /allen and a compute node; see each script's docstring for the srun line.

PY ?= python
MODELS ?= ../../../aind-dynamic-foraging-models
MEVOLVE ?= /allen/aind/scratch/han.hou/tmp/s15/mevolve-30ae4f3
MEVOLVE_PY ?= /allen/aind/scratch/han.hou/miniforge3/envs/mevolve/bin/python
GRU_OUTPUTS ?= /allen/aind/scratch/han.hou/outputs/study15/gru-heldout-replay/hpc/20261104-150009
GRU_PRIOR ?= /allen/aind/scratch/han.hou/outputs/study15/gru-heldout-replay/prior
.PHONY: all freeze-p0-golden-mevolve freeze-p0-golden-wrapper freeze-p0-task-checks \
  freeze-p1-session-groups freeze-p1-replay freeze-p1-glm freeze-p1-gru freeze-p1-gaps freeze-p1-gru-scratch

all: r1 r2

freeze-p0-golden-mevolve:
	PYTHONPATH=$(MODELS)/src:$(MEVOLVE)/src $(MEVOLVE_PY) analysis/freeze_p0_golden_mevolve.py \
	  --models-repo $(MODELS) --mevolve-repo $(MEVOLVE)

freeze-p0-golden-wrapper:
	PYTHONPATH=$(MODELS)/src $(PY) analysis/freeze_p0_golden_wrapper.py --models-repo $(MODELS)

freeze-p0-task-checks:
	$(PY) analysis/freeze_p0_task_checks.py

# P1, in order: groups (database), then replay and GLM (mevolve env), then gaps (the library only)
freeze-p1-session-groups:
	$(PY) analysis/freeze_p1_session_groups.py

freeze-p1-replay:
	PYTHONPATH=$(MODELS)/src:$(MEVOLVE)/src $(MEVOLVE_PY) analysis/freeze_p1_replay.py \
	  --models-repo $(MODELS) --mevolve-repo $(MEVOLVE)

freeze-p1-glm:
	PYTHONPATH=$(MODELS)/src:$(MEVOLVE)/src $(MEVOLVE_PY) analysis/freeze_p1_glm.py \
	  --models-repo $(MODELS) --mevolve-repo $(MEVOLVE)

freeze-p1-gru:
	PYTHONPATH=$(MODELS)/src:$(MEVOLVE)/src $(MEVOLVE_PY) analysis/freeze_p1_gru.py \
	  --models-repo $(MODELS) --outputs $(GRU_OUTPUTS) \
	  --prior-root $(GRU_PRIOR)

freeze-p1-gaps:
	PYTHONPATH=$(MODELS)/src:$(MEVOLVE)/src $(MEVOLVE_PY) analysis/freeze_p1_gaps.py --models-repo $(MODELS)

freeze-p1-gru-scratch:
	PYTHONPATH=$(MODELS)/src:$(MEVOLVE)/src $(MEVOLVE_PY) analysis/freeze_p1_gru_scratch.py --models-repo $(MODELS)

# r1 = P0: the fingerprint library against the wrapper and MEvolve, and the task checks
.PHONY: r1
r1:
	$(PY) analysis/r1_p0_checks.py

# r2 = P1: replay fingerprints of the literature models, the GLM and Win-Stay-Lose-Shift
.PHONY: r2
r2:
	$(PY) analysis/r2_replay_grid.py
