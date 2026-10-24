# studies/05-disrnn-scaling-law/Makefile
#
# One PHONY target per report; `all` regenerates everything. Per docs/posthoc-analysis.md
# "Makefile convention". Invoke from this directory; producers are cwd-independent via
# HERE = Path(__file__).parent.
#
# scaling_report.py is the single producer for BOTH reports (r1 = held-out scaling,
# r2 = sparsity + multiplier): it reads the committed grid analysis/grid.csv, writes
# analysis/summary.{json,csv} + the three figures, and regenerates the BEGIN/END blocks.
# So r1 and r2 share one producer run.
#
# WANDB_API_KEY is only needed to RE-PULL the grid from W&B (analysis/pull_grid.py); the
# committed CSV is the default source of truth, so `make` runs offline.

PY ?= python
.PHONY: all r1 r2 r3 r4 pull freeze-generative

all: r1 r3 r4

# r1 and r2 are produced together by scaling_report.py (one figure set, both report blocks).
# Listed separately for discoverability; both run the same command.
r1 r2:
	$(PY) analysis/scaling_report.py

# r3 = subject-capacity. Own producer (separate variant/question from r1/r2), reads the same
# committed grid.csv, so it also runs offline.
r3:
	$(PY) analysis/subject_capacity_report.py

# r4 = generative behavioral match. Reads the frozen rollout scalars
# (analysis/generative_runs.json), so it runs offline like r1/r2/r3.
r4:
	$(PY) analysis/generative_match.py

# Refresh analysis/generative_runs.json from W&B (needs credentials). Not part of `all`.
freeze-generative:
	$(PY) analysis/freeze_generative.py

# Refresh analysis/grid.csv from W&B (needs WANDB_API_KEY). Not part of `all` — the
# committed CSV is the source of truth so reports regenerate offline.
pull:
	$(PY) analysis/pull_grid.py
