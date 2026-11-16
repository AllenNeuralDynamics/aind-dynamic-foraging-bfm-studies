# studies/17-gru-crossfit-predictability/Makefile
#
# `all` regenerates every report OFFLINE from committed frozen inputs (CI runs it).
# One target per report (rN); one `freeze-*` target per freeze, never part of `all`.

PY ?= python
.PHONY: all freeze-folds freeze-scaling freeze-roles freeze-scoring freeze-ignore-split freeze-rebuild-check freeze-pool-strata

all: r1 gru-map verdicts

# freeze-folds = the fold assignment (reference/folds.json) and the pool-folds sweep, from study 40's
# cohort file and the MEvolve data freeze on /allen (snapshot 20260603). Needs pandas: the
# mevolve-search env, on a compute node.
freeze-folds:
	$(PY) analysis/freeze_folds.py

# freeze-scaling = the nested training-mouse subsets of the scaling curve
# (reference/scaling_subsets.json) and the scaling-folds sweeps, from reference/folds.json and the
# freeze's curricula. Needs pandas: the mevolve-search env, on a compute node.
freeze-scaling:
	$(PY) analysis/freeze_scaling_subsets.py

# freeze-roles = the fit/score role of every session (reference/roles.csv + roles.json), the table
# the held-out scoring reads. From the MEvolve data freeze on /allen (snapshot 20260603), checked
# against the digest in reference/folds.json. Needs pandas: the mevolve-search env, on a compute node.
freeze-roles:
	$(PY) analysis/freeze_roles.py

# freeze-scoring = per-mouse held-out scores of every scored model (analysis/scores.json), from the
# variant `scoring` outputs on /allen scratch. Needs pandas: the mevolve-search env, on a compute node.
freeze-scoring:
	$(PY) analysis/freeze_scoring.py

# freeze-ignore-split = 3-way vs 2-way held-out L/R log-likelihood sums by trials since the last ignore
# (analysis/ignore_split.json), from the `scoring` outputs on /allen scratch. mevolve-search env, compute node.
freeze-ignore-split:
	$(PY) analysis/freeze_ignore_split.py

# r1 = Cross-validated GRU predictability
.PHONY: r1
r1:
	$(PY) analysis/cv_schematic.py
	$(PY) analysis/cv_headline.py
	$(PY) analysis/cv_ignore_split.py
	$(PY) analysis/cv_2way_vs_s01.py
	$(PY) analysis/cv_fold_spread.py
	$(PY) analysis/cv_crosscheck.py
	$(PY) analysis/cv_lambda.py
	$(PY) analysis/cv_rebuild_check.py
	$(PY) analysis/r1_cv_predictability.py

# gru-map = the GRU held-out map's index (analysis/gru_map_index.json): each model x pass's W&B artifact
# of per-trial predictions, fold and tier per mouse, and the pass-A pooled values the registry entries
# gru_cv_map_3way / gru_cv_map_2way point at (#148). Offline.
.PHONY: gru-map
gru-map:
	$(PY) analysis/gru_map_index.py

# verdicts = the INDEX.md table and README `## Verdict`, generated from the version-2
# reports' frontmatter claims (scripts/report_verdicts.py); edit the claims, never the output
.PHONY: verdicts
verdicts:
	$(PY) ../../scripts/report_verdicts.py .

# freeze-rebuild-check = the held-out model rebuild check (wrapper #130) on pool4-3way-full pass A
# (analysis/rebuild_check.json), from /allen scratch. mevolve-search env (pyarrow), compute node.
freeze-rebuild-check:
	$(PY) analysis/freeze_rebuild_check.py

# freeze-pool-strata = each pool mouse's stratum (tier x curriculum x session-count third) as
# freeze_folds.py dealt the pool folds (reference/pool_strata.json). mevolve-search env, compute node.
freeze-pool-strata:
	$(PY) analysis/freeze_pool_strata.py
