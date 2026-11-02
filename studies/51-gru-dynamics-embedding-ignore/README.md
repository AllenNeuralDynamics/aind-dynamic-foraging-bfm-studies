---
schema_version: 1
id: 51-gru-dynamics-embedding-ignore
question: "What computation does the best 3-way (left / right / ignore) GRU implement — how its hidden-state dynamics represent engagement alongside the choice, how the subject embedding reshapes them across mice — and which of Study 50's findings on the 2-way GRU survive when the network must also predict disengagement?"
status: active
wandb:
  entity: AIND-disRNN
  project: mice_ignore_scaling
motivated_by:
- study: 02-gru-scaling-law-ignore
  note: "the 3-way GRU also predicts disengagement; how does it represent it?"
- study: 50-gru-dynamics-embedding
  note: "which of the 2-way GRU's dynamics findings survive a 3-way head?"
tags:
  model: [gru]
  question: [interpretation]
---
# Study 51 — GRU dynamics and subject embedding with an ignore action

**Question.** What computation does the best 3-way (left / right / ignore) GRU implement —
how its hidden-state dynamics represent engagement alongside the choice, how the subject
embedding reshapes them across mice — and which of Study 50's findings on the 2-way GRU
survive when the network must also predict disengagement?

This is an **analysis-only study** and the 3-way counterpart of
[Study 50](../50-gru-dynamics-embedding/). It trains nothing; its inputs are trained GRUs
from [Study 02](../02-gru-scaling-law-ignore/), pinned in [`source_runs.json`](analysis/source_runs.json).
Its one variant, [`variants/base/`](variants/base/notes.md), launches nothing; the W&B
project in the frontmatter is the source study's.

## Source models

Study 02's `nxd-3way` D=614 / H128 cell: choice/reward inputs only, 4-d subject embedding,
scalar session conditioning, 3 seeds, `best_eval` checkpoint (step 90k).

| key | role | W&B group | hidden | held-out L/R \| engaged (3 seeds) | ignore PR-AUC | source wrapper commit |
|---|---|---|---|---|---|---|
| `C3` | **primary** | [`nxd-3way@20260707-075401`](https://wandb.ai/AIND-disRNN/mice_ignore_scaling) | 128 | 0.7301–0.7304 | 0.635–0.637 | `4530d04` |

**Why C3 is a twin of Study 50's C.** Field by field, the run config differs from C-s0 only
in `data.ignore_policy` (`include`) and the wrapper commit (`4530d04`, 2026-07-05, vs C's
`9595dd3`, 2026-09-05). Snapshot, splits (`eval_every_n=2`, `heldout_every_n=5`), seeds,
architecture, schedule, lr and batch match, and both stop at step 90k. `4530d04` is an
ancestor of `9595dd3` and already contains the length-bucketed batching and snapshot pin
that separate C from Study 01's older H128 cells. Between the two commits the GRU trainer
changed only a log line; the data-loader changes add features these runs do not use, plus
`dac2f72`, which passes the minibatch RNG through merged datasets — it alters minibatch
order (seed-level noise), not which sessions or mice are in each split. *Inferred from the
diffs, not tested by training C on `4530d04`.*

**Metric caveat (from Study 02).** The 3-way likelihood has a 1/3 chance level and a
different trial support, so it is not comparable to 2-way numbers. Compare with Study 50
on L/R likelihood over engaged trials; report engagement separately.

## Pipeline

As in Study 50:

```
A. extract (needs W&B weights + JAX; CPU works: JAX_PLATFORMS=cpu)
   source_runs.json → fetch_models.py → analysis/_cache/models/<key>/run
                    → extract_trial_states.py → analysis/_cache/trial_states_<key>.parquet (gitignored)
                                              → analysis/trial_states_<key>.json (committed)
B. report (local, offline: committed files only)
```

The extractor is the wrapper's `post_training_analysis.gru_trial_states`, generalised to the
3-way head on the wrapper branch `study/13-gru-dynamics-embedding-ignore`: ignored trials
stay in the table as `action == 2` rows at their GRU timestep. Conventions for the third
action (two readouts, timesteps vs engaged trials, the 5-symbol input alphabet) are in
[`analysis/PLAN.md`](analysis/PLAN.md).

## Status

| step | state |
|---|---|
| C3 source runs frozen | done 2026-10-23 (`make freeze-sources`); L/R likelihoods match Study 02's `scaling.json` |
| A1 trial table, C3 seed 0 | done 2026-10-23 on CPU — 4.76M eval trials (8.9% ignored), 8,906 sessions, 614 mice; reproduces all three logged likelihoods to <5e-7 ([`analysis/trial_states_C3-s0.json`](analysis/trial_states_C3-s0.json)). Peak 31 GB RAM: run extractions one at a time |
| Phase B (r1) | draft 2026-10-24 — B1–B8 complete ([`analysis/reports/r1-first-look.md`](analysis/reports/r1-first-look.md)) |
| Phase C (r2) | draft 2026-10-24 — C1 (kernels by gap; counterfactual edits incl. engaged ↔ ignore), C2, C4 ([`analysis/reports/r2-vs-animals-and-2way.md`](analysis/reports/r2-vs-animals-and-2way.md)); wrapper `gru_counterfactual` now takes 3-way runs |
| Phase D (r3) | draft 2026-10-24 — fixed points of all five constant inputs, flow fields, relaxation under repeated ignores and recovery ([`analysis/reports/r3-dynamics.md`](analysis/reports/r3-dynamics.md)) |
| Phase I (r6) | draft 2026-10-24 — I1 cluster axes, I4 two-variable distilled model ([`analysis/reports/r6-integration-axis.md`](analysis/reports/r6-integration-axis.md)) |
| Phase E (r4) | draft 2026-10-24 — E2 ablations of the side / engage directions ([`analysis/reports/r4-subspaces.md`](analysis/reports/r4-subspaces.md)) |
| Phase F (r5) | draft 2026-10-24 — engagement in the embedding, per-mouse ignore dynamics ([`analysis/reports/r5-embedding.md`](analysis/reports/r5-embedding.md)) |
| Phase G1 (r7) | draft 2026-10-24 — C3 seeds 1–2 extracted (likelihoods reproduced within 6e-8); every headline result holds across seeds; r5's embedding R² corrected to a range ([`analysis/reports/r7-robustness.md`](analysis/reports/r7-robustness.md)) |
| Open | C3 free-running rollout vs Study 02 r3; per-mouse distilled models; conditional-mean ablations (E2b); C seeds 1–2 if a C-side spread is needed |

## Resuming (state at 2026-10-23 16:07 PT, before a container restart)

**Where the work lives.** Both repos are host mounts, so everything below survives a
restart. Nothing is pushed (the container has no `ssh`).

- Dispatcher checkout, branch `study/12-gru-dynamics-embedding`: this folder is
  **untracked** (like Study 50). Branch `study/13-gru-dynamics-embedding-ignore` (`10dbfce`)
  holds an identical committed copy — a backup, not the working copy.
- Wrapper checkout, branch `study/12-gru-dynamics-embedding`: the 3-way extractor lives in
  the **untracked** `code/post_training_analysis/gru_trial_states.py` and
  `code/tests/test_gru_trial_states.py`. Wrapper branch `study/13-gru-dynamics-embedding-ignore`
  (`8682fe7`) holds the same content; keep it — `trial_states_C3-s0.json` records it as the
  code that produced the table.
- `analysis/_cache/` (gitignored): `models/C3-s0/` (downloaded run), `trial_states_C3-s0.parquet`
  (3.0 GB A1 table), `extract_C3-s0.log`, `mem_C3-s0.log` (10 s memory samples).

**After a restart, check:**

```bash
cd /workspace/aind-dynamic-foraging-bfm-dispatcher/studies/51-gru-dynamics-embedding-ignore
ls -la analysis/_cache/trial_states_C3-s0.parquet          # 3,051,917,969 bytes
python -c "import pandas as pd; t = pd.read_parquet('analysis/_cache/trial_states_C3-s0.parquet', columns=['action']); print(len(t), t.action.value_counts().to_dict())"
# expect 4761751 {1: 2189963, 0: 2148764, 2: 423024}
pip install pytest   # optional; container-local, lost on restart
cd ../../../aind-dynamic-foraging-bfm-wrapper/code
JAX_PLATFORMS=cpu PYTHONPATH=$PWD python -m unittest tests.test_gru_trial_states   # 11 tests OK
```

**If the cache is gone**, rebuild it (W&B credentials are in the mounted `~/.netrc`; ~35 min
on CPU, 31 GB peak RAM — run alone):

```bash
JAX_PLATFORMS=cpu make extract MODEL=C3-s0
```

**Environment notes.**
- No GPU is visible to JAX here; set `JAX_PLATFORMS=cpu` for anything that imports JAX.
- Run the wrapper tests with `python -m unittest` from `code/`; pytest's rootdir handling
  breaks on the `code/` directory name.
- `.claude/settings.local.json` (gitignored, both repos) sets `worktree.bgIsolation: none`,
  so background Claude sessions edit these checkouts directly instead of a worktree.

**Next steps, in order.**
1. ~~Trace the C vs C3 eval-split difference~~ — done, see below.
2. ~~Phase B on C3-s0~~ — done (r1 draft, B1–B8).
3. A2: 3-way `gru_step` (ignore input `(2, 0)`) before phase D; `gru_counterfactual` still
   rejects `ignore_policy=include` and is needed for C1.

**C vs C3 eval split (resolved 2026-10-23, [`analysis/split_alignment.json`](analysis/split_alignment.json)).**
The eval split takes every other session per mouse, counted after
`_align_raw_df_with_valid_sessions`, which drops sessions where every trial was ignored —
but only under `ignore_policy=exclude`. Five mice (751769, 762206, 766559, 778099, 778236)
have such a session, so their later sessions swap between train and eval: 57 sessions
are eval only for C and 56 only for C3. C3's eval split also holds 4 all-ignore sessions
(1–14 trials). The other **8,846 sessions are shared and hold identical engaged trials**
(4,314,340 trials; same trial numbers, choices and rewards). Any same-trial comparison of C
and C3 uses those shared sessions. The cached `trial_states_C3-s0.parquet` predates a
wrapper fix and has an empty `session_id` on the 4 all-ignore sessions; fill it from
`source_ses_idx`.
