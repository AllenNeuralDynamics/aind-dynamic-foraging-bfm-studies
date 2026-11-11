# studies/40-cognitive-llm-discovery/Makefile
#
# `all` regenerates every report OFFLINE from committed frozen inputs (CI runs it).
# One target per report (rN); one `freeze-*` target per freeze, never part of `all`.

PY ?= python
.PHONY: all explainer procedure-figure gate3 seed-choice smoke-result freeze-seeded-islands-smoke cap12-result freeze-seeded-islands-cap12 freeze-engine-smoke archive-gate4 freeze-gate4 freeze-gate4-novelty

all: gate3 seed-choice smoke-result cap12-result r1 r2 r3 r4 procedure-figure

# procedure-figure = the study's procedure as a paper method figure (analysis/fig_procedure.svg),
# with cohort sizes from reference/cohorts_seed0.json; independent of the archify pages.
procedure-figure:
	$(PY) analysis/fig_procedure.py

# seed-choice = the seeds of the seeded-islands searches by ADR 0006's rule, from study 12's committed
# 2-level EM fits on the validation mice (analysis/seed_choice.json).
seed-choice:
	$(PY) analysis/seed_choice.py

# smoke-result = variants/seeded-islands-smoke's pre-registered numbers (analysis/seeded_islands_smoke.json),
# from the committed freeze analysis/seeded_islands_smoke/run.json.
smoke-result:
	$(PY) analysis/seeded_islands_smoke.py

# freeze-seeded-islands-smoke = the live read for smoke-result: the run's W&B archive (needs the W&B
# key in ~/.config/mevolve/ and /allen); never part of `all`. Commit analysis/seeded_islands_smoke/.
freeze-seeded-islands-smoke:
	$(PY) analysis/freeze_seeded_islands_smoke.py

# cap12-result = variants/seeded-islands-cap12's pre-registered numbers against seeded-islands-smoke
# (analysis/seeded_islands_cap12.json, fig_seeded_islands_cap12.svg), from both runs' committed freezes.
cap12-result:
	$(PY) analysis/seeded_islands_cap12.py

# freeze-seeded-islands-cap12 = the live read for cap12-result: the run's W&B archive and its Slurm
# steps (needs the W&B key in ~/.config/mevolve/, /allen and sacct); never part of `all`. Commit
# analysis/seeded_islands_cap12/.
freeze-seeded-islands-cap12:
	$(PY) analysis/freeze_seeded_islands_cap12.py

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
# To freeze again after scratch is purged, download each artifact into <dir>/<variant>/<launch_id>,
# the layout the freezes read (RUNS / variant / launch_id / "search"), e.g.
#   wandb artifact get <artifact> --root <dir>/gate4-recovery/20261104-132600
# then point RUNS in freeze_gate4.py at <dir>; the freezes check every file against archives.json.
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

# r2 = The first production search: curve, mechanisms and gains on unseen mice
.PHONY: r2
r2:
	$(PY) analysis/r2_search_curve.py

# r3 = The search's best on the selection and test subjects
.PHONY: r3
r3:
	$(PY) analysis/r3_search_best.py

# r4 = Simplification at three floors: fit against complexity on the test subjects
.PHONY: r4
r4:
	$(PY) analysis/r4_simplification.py
