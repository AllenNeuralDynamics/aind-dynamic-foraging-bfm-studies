# studies/ignore-trials-scaling/Makefile
#
# One PHONY target per report; `all` regenerates everything. Per
# docs/posthoc-analysis.md "Makefile convention". Invoke from this directory
# (studies/ignore-trials-scaling/); producers are cwd-independent via
# HERE = Path(__file__).parent.
#
# scaling.py is the single producer for BOTH reports (r1 = LR-engaged, r2 =
# ignore detection): it reads the frozen grid (analysis/frozen_runs.json), writes
# analysis/scaling.json (+ scaling.csv, fig_scaling.png), and regenerates the
# BEGIN/END blocks in both reports via update_reports.py. Offline; CI runs it.
#
# `freeze` is the only target that reads W&B (needs WANDB_API_KEY): it refreshes
# frozen_runs.json from the pinned groups. Never a prerequisite of `all`.

PY ?= python
.PHONY: all r1 r2 freeze

all: r1

# r1 and r2 are produced together by scaling.py (one W&B pull, one figure,
# both report blocks). Listing them separately is for discoverability; both
# run the same command.
r1 r2:
	$(PY) analysis/scaling.py

freeze:
	$(PY) analysis/freeze_runs.py
