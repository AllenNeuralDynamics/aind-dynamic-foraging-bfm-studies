# studies/52-gru-dynamics-timing-inputs/Makefile
#
# `all` regenerates what runs OFFLINE from committed JSON (CI runs it). The reports are written
# by hand from the committed JSON their frontmatter lists, so most rN have nothing to run;
# r4's summary (fig_d_rt_fixed_points.py) reads only committed JSON.
#
# Everything that reads analysis/_cache/ (this study's or Study 50's A1 tables, contexts,
# checkpoints), the GPU or the wrapper is a freeze: `make freeze-rN` reruns the producers
# behind report rN and rewrites their committed JSON; `make freeze` reruns them all. Never
# part of `all`. From the raw sources: `make extract MODEL=<key>` for each model first.
# Freezes need the wrapper checkout (WRAPPER) and the env pinned in environment.lock.

PYTHON ?= python
WRAPPER ?= ../../../aind-dynamic-foraging-bfm-wrapper/code
MODEL ?= R-s0
RUN = PYTHONPATH=$(WRAPPER) $(PYTHON)

all: r1 r2 r3 r4 r5

.PHONY: all freeze-sources fetch extract a2 b0 b3 freeze-r1 r1 i-rt freeze-r2 r2 followups freeze-r3 r3 b0h f-rt d-rt freeze-r4 r4 a2-l b0-l b3-l i-l fl f-l freeze-r5 r5 freeze

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
	$(PYTHON) analysis/a2_align_with_c.py R-s0

# Tiers 1–2 on R-s0 vs C-s0 (read the cached A1 tables; B0 also reads R-s1).
b0:
	$(PYTHON) analysis/b0_where_rt_helps.py
b3:
	$(PYTHON) analysis/b3_state_rt.py
freeze-r1: a2 b0 b3

r1: ;

# Tier 3 (stage A: JAX + the model; GPU, or JAX_PLATFORMS=cpu).
i-rt:
	$(RUN) analysis/i_rt_mechanism.py R-s0
	$(RUN) analysis/i_rt_mechanism.py R-s1
freeze-r2: i-rt

r2: ;

# Tier-3 follow-ups (stage A per seed, then offline figures). R-s1 must be extracted first.
followups:
	$(RUN) analysis/i_rt_followups.py R-s0
	$(RUN) analysis/i_rt_followups.py R-s1
	$(PYTHON) analysis/fig_i_rt_followups.py
freeze-r3: followups

r3: ;

# r4: behaviour (offline), embeddings (offline, reads model params), attractors (stage A, CPU x64).
b0h:
	$(PYTHON) analysis/b0h_outcome_by_rt.py
f-rt:
	$(PYTHON) analysis/f_embedding_rt.py
d-rt:
	JAX_PLATFORMS=cpu $(RUN) analysis/d_rt_fixed_points.py R-s0
	JAX_PLATFORMS=cpu $(RUN) analysis/d_rt_fixed_points.py R-s1
freeze-r4: b0h f-rt d-rt

# Offline: the summary and figure from the committed per-seed fixed points.
r4:
	$(PYTHON) analysis/fig_d_rt_fixed_points.py

# r5: licks model (extract L-s0 and L-s1 first: make extract MODEL=L-s0, MODEL=L-s1). Its
# alignment with C (a2-l) is part of the freeze.
a2-l:
	$(PYTHON) analysis/a2_align_with_c.py L-s0
b0-l: a2-l
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
freeze-r5: b0-l b3-l i-l fl f-l

r5: ;

freeze: freeze-sources freeze-r1 freeze-r2 freeze-r3 freeze-r4 freeze-r5
