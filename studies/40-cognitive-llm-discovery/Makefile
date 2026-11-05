# studies/40-cognitive-llm-discovery/Makefile
#
# `all` regenerates every report OFFLINE from committed frozen inputs (CI runs it).
# One target per report (rN); one `freeze-*` target per freeze, never part of `all`.

PY ?= python
.PHONY: all explainer gate3 freeze-engine-smoke archive-gate4 freeze-gate4 freeze-gate4-novelty

all: gate3 r1

# gate3 = gate 3's numbers for variants/engine-smoke/notes.md (Result), from analysis/engine_smoke/.
gate3:
	$(PY) analysis/gate3_engine_smoke.py

# freeze-engine-smoke = the live read for gate3: the run directory of engine-smoke@20261103-030341
# and the preflight ledger on /allen (read-only), plus the Console figures typed in by hand; never
# part of `all`. Run on a compute node, commit analysis/engine_smoke/, then run `make gate3`.
freeze-engine-smoke:
	$(PY) analysis/freeze_engine_smoke.py

# archive-gate4 = once per run: each gate 4 search's run directory to a W&B artifact in
# NeuralDynamics/dfbfm-mevolve, recorded in analysis/gate4/archives.json; never part of `all`.
archive-gate4:
	$(PY) analysis/archive_gate4_runs.py

# freeze-gate4 = the live read for r1: both gate 4 searches' run directories, the first half's
# gate4.json and simulated subjects on /allen (read-only), and an example session from the committed
# generator; never part of `all`. Run on a compute node with MEvolve on PYTHONPATH, commit
# analysis/gate4/, then run `make freeze-gate4-novelty` and `make r1`.
freeze-gate4:
	$(PY) analysis/freeze_gate4.py

# freeze-gate4-novelty = the post hoc novelty and repeat tables, from the committed runs.json: needs
# torch, transformers and the embedding model (HF_HOME), and MEvolve's shinka_eval on PYTHONPATH;
# never part of `all`.
freeze-gate4-novelty:
	$(PY) analysis/freeze_gate4_novelty.py

# The explainer's reference numbers, read from committed JSON (not part of `all`).
explainer:
	$(PY) analysis/explainers/build_explainer.py

# r1 = Gate 4: program recovery on simulated mice
.PHONY: r1
r1:
	$(PY) analysis/r1_gate4_program_recovery.py
