# Changelog

## 2026-10-28

- Migrated from the dispatcher's `studies/pk-02-gru-dynamics-embedding-ignore/` (never committed there, except its first three commits, replayed here from dispatcher branch
  `archive/study/13-gru-dynamics-embedding-ignore`, where it was `studies/13-…`).
  Study pk-02 is now 51; older notes and W&B text may say pk-02 or 13.
- Layout for this repository: `source_runs.json` moved to `analysis/`, variant `base` added,
  non-`rN` summaries moved to the study root, report frontmatter on schema v1.
- PNG figures are not committed (gitignored) until they are redrawn as SVGs.

## 2026-10-23

- Created the study as the 3-way (L/R/ignore) counterpart of Study 50. Surveyed every
  `ignore_policy=include` run in the `AIND-disRNN` entity: the only 3-way GRUs are Study 02's
  `nxd-3way` grid (`mice_ignore_scaling`); the other hits are early-2026 disRNN test runs.
- Pinned C3 = `nxd-3way` D=614 / H128, 3 seeds (`make freeze-sources`), chosen as the
  config twin of Study 50's C (see README for the field-by-field check).
- Wrapper `gru_trial_states` generalised to 3-action heads (wrapper commit `8682fe7` on
  branch `study/13-gru-dynamics-embedding-ignore`, also applied to the shared checkout).
  Regression: re-extracting Study 50's C-s0 reproduces its stored table (same rows,
  columns, dtypes; floats within GPU-vs-CPU error, likelihood within 5e-7).
- A1 extraction for C3 seed 0 on CPU: 4,761,751 eval trials, all three logged likelihoods
  reproduced (diffs ≤ 4e-7), readout error 2e-6. The first attempt was OOM-killed while a
  second 26 GB job ran; the rerun alone peaked at 31 GB, so `extract_trial_states.py` now
  refuses to start below 40 GB available. The JSON's `wrapper_checkout.path` is the
  worktree the run used; its `head` (`8682fe7`) is the commit to check out.
- Traced the C vs C3 eval-split difference (`analysis/split_alignment.py` → `.json`): the
  every-other-session split runs after all-ignore sessions are dropped, which happens only
  under `ignore_policy=exclude`, so 5 mice have shifted eval sessions (57 C-only, 56
  C3-only). The 8,846 shared sessions hold identical engaged trials. Wrapper fix (shared
  checkout, uncommitted): all-ignore sessions keep their `session_id` in the trial table.
- Phase B, first part (r1 draft): `_pk02.py` (two readouts, ignore runs, Study 50 twin
  loading); B1 example sessions, C3 above C on Study 50's sessions; B3a PCA over all
  timesteps; B3b–c readout geometry and the state by previous input (a fifth, post-ignore
  cluster); B8 engagement calibration (bins, bout-exit hazard, onset/end, trial position,
  reward history).

## 2026-10-24

- Phase B completed (r1 draft): B2 units by readout; B4 update maps and B5 outcome-
  triggered averages for both readouts, C3 and C side by side on the shared sessions; B6
  tracking (C3, C) plus ignore rate vs reward-probability sum (no consistent relation;
  confounded by era); B7 decoders with ignore history and engagement variables. B7's bar
  chart is aligned row-for-row with its PC heatmap (Study 50's `b7_decoders.py` has the
  same layout and probably the same offset; not changed here).
- Phase C (r2 draft), framed as C3 vs C and what the ignore input does: C2 likelihood of
  C3, C and the RL fits on the shared engaged trials, split by ignores just before; C4
  stay-after-gap behaviour; C1b Su2022 kernels per gap group; C1a counterfactual edits
  (reward flip, engaged → ignore, ignore → unrewarded choice). Wrapper (shared checkout,
  uncommitted): `gru_counterfactual` accepts 3-action runs (`logits`, `readouts`,
  `input_edit_effects`; `choice_logit_diff` / `reward_flip_effects` unchanged) with tests.
  Counterfactuals run at full float32 matmul precision: on GPU, TF32 made edited and
  baseline runs differ by ~5e-4 with no edit.
- `_pk02.engaged_comparison_table`: the shared engaged trials with every model's P(right)
  and the gap (checked against a brute-force count).
- Phase D (r3 draft): `d_dynamics.py` with the wrapper's `gru_step` unchanged (the ignore
  input is just obs (2, 0); the engage readout is computed from the readout weights in the
  study). Fixed points of five constant inputs (vs C's four from Study 50), flow fields,
  relaxation under repeated ignores and recovery. Float64 at full matmul precision.
  Note: three `tests/test_gru_step.py` tests (another session's) fail on GPU by 3–6e-4
  (TF32) and pass on CPU — not a regression.
- Phases I, E, F (r6, r4, r5 drafts). I1: side / engage encoding axes per last-input cluster,
  C3 vs C. I4: two-variable (side, engage) distilled model from C3's readouts, simulated from
  the animals' inputs; variants with the ignore rule removed and C's own distillation. E2:
  mean-replacement ablations of the side / engage directions with random controls
  (own session contexts from `d_dynamics.py`'s cache). F: per-mouse engagement features,
  decoded from C3's and C's embeddings, and C3's per-mouse ignore dynamics.
- Phase G1 (r7 draft): A1 extraction of C3-s1 and C3-s2 on GPU (peak ~42 GB each, run one
  at a time); `g_seeds.py` (table metrics per seed) and `g_dynamics_seed.py` (per-seed
  attractors and relaxation); `_pk02.engaged_comparison_table(key)`. All headline results
  hold across seeds. Corrected r5's embedding R² (0.68 → 0.47–0.68 across seeds).

## 2026-10-26

- I2–I3 (r6): Study 50's TDR on C3 with "previous step ignored" and engagement as
  regressors, and value-coordinate transitions per input × source incl. ignore, plus the
  side readout ≈ a·v + b per source cluster. The value coordinate is carried through ignores
  (R² 0.95–0.99). Corrected the "ignore shrinks the side preference toward neutral" wording:
  true of the readout; on the value coordinate an ignore carries v like any input and removes
  the readout's last-choice offset.
- I3c (r6): causal nudge along the value axis, re-run on observed inputs, with engage and
  random-orthogonal controls. One-step causal carry 0.61–0.69 in every input × source cell
  (ignore 0.65–0.69); I3's regression slopes overstate the spread. Decay slower than the
  I3-slope chain after ~4 steps.
