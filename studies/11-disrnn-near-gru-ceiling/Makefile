# studies/11-disrnn-near-gru-ceiling/Makefile
#
# One PHONY target per report; `all` regenerates everything. Per docs/posthoc-analysis.md
# "Makefile convention". Invoke from this directory; producers are cwd-independent via
# HERE = Path(__file__).parent.
#
# r1 regenerates OFFLINE from the committed analysis/grid.csv + analysis/reference_points.csv;
# `pull` re-pulls both from W&B (needs WANDB_API_KEY).

PY ?= python
.PHONY: all r1 pull

all: r1

# r1 = staged gap-closure attribution across the 3-stage ablation.
r1:
	$(PY) analysis/attribution_report.py

# Refresh analysis/grid.csv + analysis/reference_points.csv from W&B (needs WANDB_API_KEY).
pull:
	$(PY) analysis/pull_grid.py

# freeze-provenance = what the headline results' launches trained on and ran in, recovered from
# W&B run configs and Beaker jobs (#28). Needs W&B credentials and BEAKER_TOKEN: compute node only.
.PHONY: freeze-provenance
freeze-provenance:
	$(PY) ../util/freeze_run_provenance.py --study . --project AIND-disRNN/disrnn_near_gru_ceiling \
	  --group stage1-no-penalty@20260909-021133 \
	  --group stage2-linear-update-net@20260909-021223 \
	  --group stage3-wide-latent@20260909-140640
