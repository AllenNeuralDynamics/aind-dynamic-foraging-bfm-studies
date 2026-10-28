# studies/13-hb-subject-vs-gru-heldout/Makefile
#
# `all` regenerates every report OFFLINE from committed frozen inputs (CI runs it).
# One target per report (rN); one `freeze-*` target per freeze, never part of `all`.

PY ?= python
.PHONY: all

all: r1 r2 r3 r4 r5

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

# freeze-ctt = the live read for r5 (W&B GraphQL + HPC artifact files); never part of `all`.
# Run on a compute node with WANDB_API_KEY exported, then commit analysis/ctt_*runs.json.
.PHONY: freeze-ctt
freeze-ctt:
	$(PY) analysis/freeze_ctt.py

# freeze-posteriors = per-subject posterior summaries for r3, from each run's saved draws on the HPC
# filesystem; never part of `all`. Run on a compute node, then commit analysis/hattori_posteriors.json.
.PHONY: freeze-posteriors
freeze-posteriors:
	JAX_PLATFORMS=cpu $(PY) analysis/freeze_posteriors.py

# freeze-plugin = the HB plug-in control for r1: reloads the eval sessions through the wrapper's loader
# (pinned code at ~/code/wt/hpc13b) and scores one fixed parameter set per subject; never part of `all`.
.PHONY: freeze-plugin
freeze-plugin:
	JAX_PLATFORMS=cpu $(PY) analysis/freeze_plugin.py

# r4 = The HB ladder: from per-subject MLE to the full two-level HB, step by step
.PHONY: r4
r4:
	$(PY) analysis/r4_ladder.py

# r5 = HB-CTT two-level vs HB-Hattori, per-subject MLE and GRU on held-out subjects
.PHONY: r5
r5:
	$(PY) analysis/r5_ctt_pooled.py
