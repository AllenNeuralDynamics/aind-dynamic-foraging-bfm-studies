# studies/13-hb-subject-vs-gru-heldout/Makefile
#
# `all` regenerates every report OFFLINE from committed frozen inputs (CI runs it).
# One target per report (rN); one `freeze-*` target per freeze, never part of `all`.

PY ?= python
.PHONY: all

all: r1 r2 r3

# r1 = HB-Hattori2019 two-level vs GRU and per-subject MLE on held-out subjects
.PHONY: r1
r1:
	$(PY) analysis/r1_hattori_pooled.py

# freeze-hattori = the live read for r1 (W&B GraphQL + HPC artifact files); never part of `all`.
# Run on a compute node with WANDB_API_KEY exported, then commit analysis/hattori_runs.json.
.PHONY: freeze-hattori
freeze-hattori:
	$(PY) analysis/freeze_hattori.py

# r2 = Per-subject held-out likelihood: HB vs GRU vs per-subject MLE
.PHONY: r2
r2:
	$(PY) analysis/r2_per_subject.py

# r3 = What the HB learns: session-to-session variability and parameters vs per-subject MLE
.PHONY: r3
r3:
	$(PY) analysis/r3_posteriors_vs_mle.py

# freeze-posteriors = per-subject posterior summaries for r3, from each run's saved draws on the HPC
# filesystem; never part of `all`. Run on a compute node, then commit analysis/hattori_posteriors.json.
.PHONY: freeze-posteriors
freeze-posteriors:
	JAX_PLATFORMS=cpu $(PY) analysis/freeze_posteriors.py
