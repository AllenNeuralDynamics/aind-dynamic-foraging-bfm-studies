# studies/50-gru-dynamics-embedding/Makefile
#
# `all` regenerates every report OFFLINE from committed JSON (CI runs it): each rN runs only
# its renderer, plus g1 for r7, which reads the committed per-seed JSON.
#
# Everything that reads analysis/_cache/ (the A1 trial tables, PCA, downloaded checkpoints
# and RL outputs), the GPU, W&B or the wrapper is a freeze: `make freeze-rN` reruns the
# producers behind report rN and rewrites their committed JSON; `make freeze` reruns them all.
# Never part of `all`. To reproduce rN from the raw sources: `make extract freeze-rN rN`.
# Freezes need the wrapper checkout (WRAPPER) and the env pinned in environment.lock.

PYTHON ?= python
WRAPPER ?= ../../../aind-dynamic-foraging-bfm-wrapper/code
MODEL ?= C-s0
RUN = PYTHONPATH=$(WRAPPER) $(PYTHON)

.PHONY: all freeze-sources fetch extract b1 b2 b3 b3c-history b3h b3a-10 b4 b5 b6 b7 freeze-r1 r1 c-rollouts c1 c2 c2c-pc-pairs c4 c4e c3 freeze-r2 r2 e1 f3 f1 f4 f2 f5 f4b f4c f7 f8 f9 f6 f10 f11 f12 f13 freeze-r4-r5 r4 r5 d d1b d1c d1d d1f d1e freeze-r3 r3 i1 i2 i3b i2b i3 i4 i5 i6 i7 i8 i9 i9b i9c i9d i9e i10 i11 i11d freeze-rl-medians freeze-r6 r6 e2 e2b h3 h4 h1 h2 freeze-r8 r8 g-seeds g1 g3 freeze-r7 r7 freeze
all: r1 r2 r3 r4 r5 r6 r7 r8

# Reads W&B over GraphQL; needs WANDB_API_KEY (or ~/.netrc). Rewrites source_runs.json.
freeze-sources:
	$(PYTHON) analysis/freeze_source_runs.py

# Stage A (weights + data + JAX; runs on a GPU box). MODEL=<key> from source_runs.json.
fetch:
	$(RUN) analysis/fetch_models.py $(MODEL)

extract: fetch
	$(RUN) analysis/extract_trial_states.py $(MODEL)

# Phase B producers: read the cached A1 table (run `make extract` first on a new machine).
b1:
	$(RUN) analysis/b1_example_sessions.py
b2:
	$(RUN) analysis/b2_unit_heatmap.py
b3:
	$(RUN) analysis/b3_pca.py
	$(RUN) analysis/b3_state_space.py
b3c-history: b3
	$(PYTHON) analysis/b3c_history.py
	$(RUN) analysis/b3g_viewer.py
b3h: b3
	$(PYTHON) analysis/b3h_pc_overview.py
b3a-10: b3
	$(PYTHON) analysis/b3a_pca_variance_10.py
b4:
	$(RUN) analysis/b4_update_maps.py
b5:
	$(RUN) analysis/b5_outcome_triggered.py
b6:
	$(RUN) analysis/b6_tracking.py
b7: b3
	$(RUN) analysis/b7_decoders.py

freeze-r1: b1 b2 b3 b3c-history b4 b5 b6 b7

# Offline formatter over the committed B JSONs.
r1:
	$(PYTHON) analysis/report_r1.py

# Phase C. Stage A: RL replays (CPU) and the GRU counterfactual (GPU, or JAX_PLATFORMS=cpu).
c-rollouts:
	$(RUN) analysis/c2_rl_rollouts.py
c1: c-rollouts
	$(RUN) analysis/c1_impulse.py
	$(RUN) analysis/c1_kernels.py
	$(RUN) analysis/c1_figure.py
c2: c-rollouts b3
	$(RUN) analysis/c2_compare.py
c2c-pc-pairs: c2
	$(PYTHON) analysis/c2c_pc_pairs.py
c4: c-rollouts
	$(PYTHON) analysis/c4_behaviour.py
c4e: c-rollouts
	$(PYTHON) analysis/c4e_signatures.py
c3: c-rollouts b3
	$(RUN) analysis/c3_decoders.py

freeze-r2: c1 c2 c2c-pc-pairs c3 c4 c4e

r2:
	$(PYTHON) analysis/report_r2.py

# Phase E/F, parts that need no step function (cached tables only).
e1: c-rollouts b3
	$(RUN) analysis/e1_subspaces.py
f3: c1
	$(RUN) analysis/f3_embedding_features.py
f1: f3 b3
	$(RUN) analysis/f1_embedding_extremes.py
f4: c1
	$(RUN) analysis/f4_early_late.py

f2: d f3 i5
	$(RUN) analysis/f2_mouse_dynamics.py   # GPU; reuses _cache/f2_fixed_points_*.parquet if present
f5: f2 f3 i5 c-rollouts
	$(PYTHON) analysis/f5_embedding_atlas.py   # fetches the snapshot session table once (public S3)
f4b: f4 d
	$(RUN) analysis/f4b_context_drift.py   # GPU
f4c: d c1
	$(RUN) analysis/f4c_context_trajectories.py
f7: c1
	$(PYTHON) analysis/f7_feature_library.py
f8: f7 d
	$(RUN) analysis/f8_embedding_traversal.py   # GPU
f9: f7 c1
	$(RUN) analysis/f9_strategy_cards.py
f6: f5
	$(PYTHON) analysis/f6_embedding_types.py
f10: f7 f5
	$(PYTHON) analysis/f10_embedding_behaviour_maps.py
f11: f10 extract
	$(PYTHON) analysis/f11_prototype_examples.py
f12: f10 extract
	$(PYTHON) analysis/f12_context_learning_examples.py
f13: f11 extract
	$(PYTHON) analysis/f13_prototype_state_space.py
freeze-r4-r5: e1 e2 e2b f1 f3 f4 f4b f4c f2 f5 f6 f7 f8 f9 f10 f11 f12 f13

# One renderer writes both reports.
r4:
	$(PYTHON) analysis/report_r4_r5.py
r5: r4

# Phase D (stage A; ~40 min on CPU with JAX_PLATFORMS=cpu). Needs F1's groups.
d: f1 b3
	$(RUN) analysis/d_dynamics.py
d1b: d
	$(RUN) analysis/d1b_convergence.py
d1c: d
	$(RUN) analysis/d1c_flow_two_scales.py
d1d: d1c
	$(RUN) analysis/d1d_flow_scaled.py
d1f: d1d b3h
	$(RUN) analysis/d1f_fixed_points_flow.py
d1e: d1d f2 f9
	$(RUN) analysis/d1e_flow_own_context.py   # GPU
freeze-r3: d

r3:
	$(PYTHON) analysis/report_r3.py

# Phase I — integration axis (needs D's cached contexts and C's Hattori rollout).
i1: d c-rollouts
	$(RUN) analysis/i1_cluster_axes.py
i2: d
	$(RUN) analysis/i2_tdr.py
i3b: i3 i4
	$(PYTHON) analysis/i3b_stay_switch.py
i2b: i2 i4
	$(PYTHON) analysis/i2b_value_axis_in_pcs.py
i3: i2
	$(RUN) analysis/i3_tdr_dynamics.py
i4: i2 c-rollouts e2
	$(PYTHON) analysis/i4_distilled_model.py
i5: i4 c1 f3
	$(PYTHON) analysis/i5_mouse_dynamics.py
i6: i4
	$(PYTHON) analysis/i6_rl_transitions.py
i7: i4
	$(PYTHON) analysis/i7_second_variable.py
i8: i4 i7 c-rollouts
	$(RUN) analysis/i8_v_tests.py
i9: i4 c-rollouts
	$(PYTHON) analysis/i9_v_kernel.py
i9b: i9 f7
	$(PYTHON) analysis/i9b_v_kernel_per_mouse.py
i9c: i9 f7
	$(PYTHON) analysis/i9c_v_kernel_variants.py
i9d: i9
	$(PYTHON) analysis/i9d_v_kernel_long.py
i9e: i9d f7
	$(PYTHON) analysis/i9e_trace_models.py   # GPU
i10: i2 d
	$(RUN) analysis/i10_gates.py
i11: i2 i3 b3h
	$(PYTHON) analysis/i11_value_across_lobes.py
i11d: i11 i3b
	$(PYTHON) analysis/i11d_carry_summary.py
# r6 quotes the RL baselines' median learning rates from the downloaded fits.
freeze-rl-medians: c-rollouts
	$(PYTHON) analysis/freeze_rl_fit_medians.py

freeze-r6: i1 i2 i3 i4 i5 i6 i7 i8 i9 i9b i9c i9d i9e i10 i11 freeze-rl-medians

r6:
	$(PYTHON) analysis/report_r6.py

# Phase E2 (stage A, CPU ok): ablations; e2b supersedes e2's single-axis reading.
e2: i2
	$(RUN) analysis/e2_ablations.py
e2b: i2
	$(RUN) analysis/e2b_conditional_ablations.py

# Phase H — the GRU's strategy vs Q-learning (r8). h4 uses the GPU if present.
h3: c-rollouts b3
	$(PYTHON) analysis/h3_q_phase_portraits.py
h4: d c-rollouts
	$(RUN) analysis/h4_scenarios.py
h1: c-rollouts
	$(PYTHON) analysis/h1_model_ladder.py   # GPU
h2: d c-rollouts
	$(RUN) analysis/h2_learning_rate.py   # GPU
freeze-r8: h1 h2 h3 h4

r8:
	$(PYTHON) analysis/report_r8.py

# Phase G — robustness (r7). G1/G3 seed checks: fetch + extract C-s1, C-s2, rerun the core
# producers per seed into analysis/seeds/<key>/ (S12_MODEL_KEY; GPU), then compare.
g-seeds:
	for k in C-s1 C-s2; do $(MAKE) extract MODEL=$$k && analysis/g_run_seed.sh $$k; done
g1:
	$(PYTHON) analysis/g1_seed_robustness.py
g3:
	$(PYTHON) analysis/g3_cross_seed_alignment.py
freeze-r7: g-seeds g3

# g1 reads the committed per-seed JSON under analysis/seeds/ and runs offline.
r7: g1
	$(PYTHON) analysis/report_r7.py

freeze: freeze-sources freeze-r1 freeze-r2 freeze-r3 freeze-r4-r5 freeze-r6 freeze-r7 freeze-r8
