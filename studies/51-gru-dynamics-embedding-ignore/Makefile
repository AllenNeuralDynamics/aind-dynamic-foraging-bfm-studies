PYTHON ?= python
WRAPPER ?= ../../../aind-dynamic-foraging-bfm-wrapper/code
MODEL ?= C3-s0
RUN = PYTHONPATH=$(WRAPPER) $(PYTHON)

# Every report. Until the producers that read analysis/_cache/ move behind freeze-*
# targets, this needs the cached A1 tables and does not run offline.
.PHONY: all
all: r1 r2 r3 r4 r5 r6 r7

.PHONY: freeze-sources fetch extract split-alignment b1 b2 b3 b4 b5 b6 b7 b8 r1 c1 c2 c4 r2 d r3 i1 i2 i3 i3c i4 r6 e2 r4 f r5 g-seeds g r7

# Reads W&B over GraphQL; needs WANDB_API_KEY (or ~/.netrc). Rewrites source_runs.json.
freeze-sources:
	$(PYTHON) analysis/freeze_source_runs.py

# Stage A (weights + data + JAX; GPU box, or CPU with JAX_PLATFORMS=cpu). MODEL=<key> from source_runs.json.
# extract peaks at ~31 GB RAM per model and refuses to start below 40 GB available:
# run one model at a time, never with -j or alongside another heavy job.
fetch:
	$(RUN) analysis/fetch_models.py $(MODEL)

extract: fetch
	$(RUN) analysis/extract_trial_states.py $(MODEL)

# Shared eval sessions of Study 50's C-s0 and C3-s0 (reads both cached A1 tables; ~6 GB RAM).
split-alignment:
	$(PYTHON) analysis/split_alignment.py

# Phase B producers: read the cached A1 table (run `make extract` first on a new machine).
b1:
	$(PYTHON) analysis/b1_example_sessions.py
b2:
	$(PYTHON) analysis/b2_unit_heatmap.py
b3:
	$(PYTHON) analysis/b3_pca.py
	$(PYTHON) analysis/b3_state_space.py
# B4–B6 compare with Study 50's C-s0 table on the shared sessions (needs split-alignment).
b4: split-alignment
	$(PYTHON) analysis/b4_update_maps.py
b5: split-alignment
	$(PYTHON) analysis/b5_outcome_triggered.py
b6: split-alignment
	$(PYTHON) analysis/b6_tracking.py
# ~15 min on CPU (31 ridge fits x 5 folds on 400k steps).
b7: b3
	$(PYTHON) analysis/b7_decoders.py
b8:
	$(PYTHON) analysis/b8_engagement.py

# Offline formatter over the committed B JSONs.
r1: split-alignment b1 b2 b3 b4 b5 b6 b7 b8
	$(PYTHON) analysis/report_r1.py

# Phase C. C2/C4/C1b read the cached tables plus Study 50's C-s0 table and RL replays
# (studies/50-gru-dynamics-embedding `make c-rollouts`). C1a is stage A: loads the data
# bundle and re-runs the GRU (~15 GB RAM; GPU, or JAX_PLATFORMS=cpu).
c2: split-alignment
	$(PYTHON) analysis/c2_compare.py
c4: split-alignment
	$(PYTHON) analysis/c4_after_ignore.py
c1: split-alignment
	$(PYTHON) analysis/c1_kernels.py
	$(RUN) analysis/c1_impulse.py
	$(PYTHON) analysis/c1_figure.py
r2: c1 c2 c4
	$(PYTHON) analysis/report_r2.py

# Phase D (stage A): loads the data bundle (~17 GB RAM); float64, GPU or JAX_PLATFORMS=cpu.
d: b3
	$(RUN) analysis/d_dynamics.py
r3: d
	$(PYTHON) analysis/report_r3.py

# Phase I (tables only; I1 holds ~6 GB of states for C3 and C at once).
i1: split-alignment
	$(PYTHON) analysis/i1_cluster_axes.py
i4: split-alignment
	$(PYTHON) analysis/i4_distilled_model.py
# I2–I3 need d's cached contexts; I3 finds fixed points on CPU (~2 min).
i2: d
	$(PYTHON) analysis/i2_tdr.py
i3: i2
	JAX_PLATFORMS=cpu $(RUN) analysis/i3_tdr_dynamics.py
# Causal nudge along the value axis (table + params; ~2 min on CPU).
i3c: i3
	JAX_PLATFORMS=cpu $(RUN) analysis/i3c_value_causal.py
r6: i1 i2 i3 i3c i4
	$(PYTHON) analysis/report_r6.py

# Phase E (stage A: data bundle, ~15 GB RAM; needs d's cached contexts).
e2: d
	$(RUN) analysis/e2_ablations.py
r4: e2
	$(PYTHON) analysis/report_r4.py

# Phase F (tables + params; needs d's cached contexts; CPU is fine).
f: d
	JAX_PLATFORMS=cpu $(RUN) analysis/f_engagement_embedding.py
r5: f
	$(PYTHON) analysis/report_r5.py

# Phase G1: seeds 1–2. Extract one at a time (~42 GB peak on GPU), then the table metrics
# (~11 min) and the per-seed dynamics (stage A, ~17 GB each, sequential).
g-seeds:
	$(MAKE) extract MODEL=C3-s1
	$(MAKE) extract MODEL=C3-s2
g: split-alignment
	$(PYTHON) analysis/g_seeds.py
	for k in C3-s0 C3-s1 C3-s2; do $(RUN) analysis/g_dynamics_seed.py $$k; done
	$(PYTHON) analysis/g_figure.py
r7: g
	$(PYTHON) analysis/report_r7.py
