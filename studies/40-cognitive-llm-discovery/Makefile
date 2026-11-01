# studies/40-cognitive-llm-discovery/Makefile
#
# `all` regenerates every report OFFLINE from committed frozen inputs (CI runs it).
# One target per report (rN); one `freeze-*` target per freeze, never part of `all`.

PY ?= python
.PHONY: all explainer

all:

# The explainer's reference numbers, read from committed JSON (not part of `all`).
explainer:
	$(PY) analysis/explainers/build_explainer.py
