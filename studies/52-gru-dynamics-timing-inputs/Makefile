PYTHON ?= python
WRAPPER ?= ../../../aind-dynamic-foraging-bfm-wrapper/code
MODEL ?= R-s0
RUN = PYTHONPATH=$(WRAPPER) $(PYTHON)

# Every report. Until the producers that read analysis/_cache/ move behind freeze-*
# targets, this needs the cached A1 tables and does not run offline.
.PHONY: all
all: r1 r2 r3 r4 r5

.PHONY: freeze-sources fetch extract a2 b0 b3 r1 i-rt r2 followups r3 b0h f-rt d-rt r4 b0-l b3-l i-l fl f-l r5

# Reads W&B over GraphQL; needs WANDB_API_KEY (or ~/.netrc). Rewrites source_runs.json.
freeze-sources:
	$(PYTHON) analysis/freeze_source_runs.py

# Stage A (weights + data + JAX; GPU box, or CPU with JAX_PLATFORMS=cpu). MODEL=<key> from source_runs.json.
# extract peaks at ~31 GB RAM per model (Study 51) and refuses to start below 40 GB available:
# run one model at a time, never with -j or alongside another heavy job.
fetch:
	$(RUN) analysis/fetch_models.py $(MODEL)

extract: fetch
	$(RUN) analysis/extract_trial_states.py $(MODEL)

# Same-trial alignment with Study 50's C-s0 (reads both cached A1 tables).
a2:
	$(PYTHON) analysis/a2_align_with_c.py $(MODEL)

# Tiers 1–2 on R-s0 vs C-s0 (read the cached A1 tables; B0 also reads R-s1). No `make` in the
# container: run these lines directly from the study folder.
b0:
	$(PYTHON) analysis/b0_where_rt_helps.py
b3:
	$(PYTHON) analysis/b3_state_rt.py
r1: a2 b0 b3

# Tier 3 (stage A: JAX + the model; GPU, or JAX_PLATFORMS=cpu).
i-rt:
	$(RUN) analysis/i_rt_mechanism.py R-s0
	$(RUN) analysis/i_rt_mechanism.py R-s1
r2: i-rt

# Tier-3 follow-ups (stage A per seed, then offline figures). R-s1 must be extracted first.
followups:
	$(RUN) analysis/i_rt_followups.py R-s0
	$(RUN) analysis/i_rt_followups.py R-s1
	$(PYTHON) analysis/fig_i_rt_followups.py
r3: followups

# r4: behaviour (offline), embeddings (offline, reads model params), attractors (stage A, CPU x64).
b0h:
	$(PYTHON) analysis/b0h_outcome_by_rt.py
f-rt:
	$(PYTHON) analysis/f_embedding_rt.py
d-rt:
	JAX_PLATFORMS=cpu $(RUN) analysis/d_rt_fixed_points.py R-s0
	JAX_PLATFORMS=cpu $(RUN) analysis/d_rt_fixed_points.py R-s1
	$(PYTHON) analysis/fig_d_rt_fixed_points.py
r4: b0h f-rt d-rt

# r5: licks model (extract L-s0 and L-s1 first: make extract MODEL=L-s0, MODEL=L-s1).
b0-l:
	$(PYTHON) analysis/b0_licks.py
b3-l:
	$(PYTHON) analysis/b3_state_licks.py
i-l:
	$(RUN) analysis/i_licks_mechanism.py L-s0
	$(RUN) analysis/i_licks_mechanism.py L-s1
fl:
	$(RUN) analysis/i_licks_followups.py L-s0
	$(RUN) analysis/i_licks_followups.py L-s1
	$(PYTHON) analysis/fig_i_licks_followups.py
f-l:
	$(PYTHON) analysis/f_embedding_licks.py
r5: b0-l b3-l i-l fl f-l
