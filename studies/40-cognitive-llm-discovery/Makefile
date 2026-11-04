# studies/40-cognitive-llm-discovery/Makefile
#
# `all` regenerates every report OFFLINE from committed frozen inputs (CI runs it).
# One target per report (rN); one `freeze-*` target per freeze, never part of `all`.

PY ?= python
.PHONY: all explainer gate3 freeze-engine-smoke

all: gate3

# gate3 = gate 3's numbers for variants/engine-smoke/notes.md (Result), from analysis/engine_smoke/.
gate3:
	$(PY) analysis/gate3_engine_smoke.py

# freeze-engine-smoke = the live read for gate3: the run directory of engine-smoke@20261103-030341
# and the preflight ledger on /allen (read-only), plus the Console figures typed in by hand; never
# part of `all`. Run on a compute node, commit analysis/engine_smoke/, then run `make gate3`.
freeze-engine-smoke:
	$(PY) analysis/freeze_engine_smoke.py

# The explainer's reference numbers, read from committed JSON (not part of `all`).
explainer:
	$(PY) analysis/explainers/build_explainer.py
