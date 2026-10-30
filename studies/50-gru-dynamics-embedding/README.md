---
schema_version: 1
id: 50-gru-dynamics-embedding
question: "What computation do the best-trained GRUs implement — the structure of their hidden-state dynamics, and how the subject embedding reshapes those dynamics across mice — and does it change between hidden size 128 and 256?"
status: active
wandb:
  entity: AIND-disRNN
  project: mice_data_scaling
motivated_by:
- study: 01-gru-scaling-law
  note: "the best D=614 GRUs predict mice well; what computation do they implement?"
tags:
  model: [gru]
  question: [interpretation]
---
# Study 50 — GRU dynamics and subject embedding

**Question.** What computation do the best-trained GRUs implement — the structure of
their hidden-state dynamics, and how the subject embedding reshapes those dynamics across
mice — and does it change between hidden size 128 and 256?

This is an **analysis-only study**: it trains nothing. Its inputs are trained GRUs from
Study 01, pinned in [`source_runs.json`](analysis/source_runs.json). Its one variant, [`variants/base/`](variants/base/notes.md), launches nothing; the W&B
project in the frontmatter is the source study's. If an extraction step is submitted to
Beaker, it gets a launch record under that variant then.

## Source models

Study 01's largest-cohort cells: D=614 mice, choice/reward inputs only (no timing
features), 4-d subject embedding, scalar session conditioning, 3 seeds each, `best_eval`
checkpoint. Run ids and immutable artifact digests are in `source_runs.json`.

| key | role | W&B group | hidden | held-out LL (3 seeds) | source wrapper commit |
|---|---|---|---|---|---|
| `C` | **primary** | [`e4-e8-d614-source@20260906-195409`](https://wandb.ai/AIND-disRNN/mice_data_scaling) (E4 runs only) | 128 | 0.7289–0.7290 | `9595dd3` |
| `B` | next: capacity comparison | [`nxd-grid@20260623-102649`](https://wandb.ai/AIND-disRNN/mice_data_scaling) | 256 | 0.7290–0.7291 | `4f29680` |

C was chosen over Study 01's original H128 cells (`v2-sc-active@20260622-144622`, 0.7282,
wrapper `65c3350`) because the same config on newer code scores 0.7289 — i.e. most of the
apparent H128→H256 gain in Study 01 is a code-version effect, and C is the best-scoring,
newest plain H128. C's digests match Study 09's freeze of the same runs. B and C were
trained from different wrapper commits; read any B-vs-C difference with that in mind.

## Pipeline

```
A. extract (needs W&B weights + JAX; Beaker or a compute node)
   source_runs.json → wrapper post_training_analysis.wandb_model_dir → model_dir
                    → wrapper run_analysis <subcommand> → analysis/_cache/ (gitignored)
                                                       → analysis/*.json|csv (committed)
B. report (local, offline: committed files only)
   analysis/*.json|csv → fig_*.png, reports/r<n>-*.md
```

Reusable analysis code (hidden-state extraction, fixed points, embedding geometry) lives in
the wrapper's `code/post_training_analysis/`; this folder holds only the thin
study-specific callers, frozen numbers, figures and reports.

**Resuming in a new session:** [`HANDOFF.md`](HANDOFF.md).

**Figure guide:** [`analysis/FIGURES.md`](analysis/FIGURES.md) — how every committed figure is made and what it shows.

## Status

| step | state |
|---|---|
| C, B source runs frozen | done 2026-10-15 (`make freeze-sources`) |
| A1 trial table, C seed 0 | done 2026-10-16 — 4.34M eval trials; likelihood reproduces the logged 0.73385 |
| Phase B (r1) | draft 2026-10-16 — [`analysis/reports/r1-first-look.md`](analysis/reports/r1-first-look.md) |
| Phase C (r2) | draft 2026-10-16 — [`analysis/reports/r2-vs-animals-and-rl.md`](analysis/reports/r2-vs-animals-and-rl.md); RL baselines pinned in `source_runs.json` |
| E, F — table-only parts (r4, r5) | draft 2026-10-19 — E1 without slow modes; behavioural halves of F1, F3, F4 |
| A2 + D (r3) | draft 2026-10-19 — free-context step (wrapper `gru_step`), fixed points / timescales / flow for 5 contexts, on CPU |
| Independent review | 2026-10-19 — fixes applied to C1, C2, F3, F4 and report claims (see CHANGELOG) |
| Phase I — integration axis (r6) | draft 2026-10-20 — per-cluster encoding axes, TDR, value-coordinate dynamics, distilled 1-variable model, per-mouse dynamics, RL transitions |
| E2 ablations (r4) | draft 2026-10-20 — global-mean (E2) and conditional-mean (E2b) ablations with random controls |
| B3c deeper history, C2c in all PC pairs | done 2026-10-23 |
| C4 behaviour vs animals (r2) | draft 2026-10-23 — logit-change maps, update / history calibration, disagreement, two signatures (C4a–e) |
| I7, I8 (r6) | draft 2026-10-23 — second carried coordinate; causal tests of v (closure, perturbation, flip kernels vs RL) |
| F2 (r5) | done 2026-10-24 on GPU — attractors / timescales / v retention per mouse context |
| H3, H4 (r8) | draft 2026-10-24 — Q-coordinate phase portraits; canonical scenarios |
| H1 ladder, H2 learning rate (r8); F5 atlas, F6 types (r5) | draft 2026-10-24 |
| F4b context drift (r5); four-reviewer correctness pass | done 2026-10-26 (fixes in CHANGELOG) |
| G1 + G3 seeds (r7) | draft 2026-10-27 — [`analysis/reports/r7-robustness.md`](analysis/reports/r7-robustness.md); C seeds 1–2 extracted and 35 producers rerun under `analysis/seeds/` |
| H1 follow-ups, embedding traversal, loose ends, G2 (model B) | next — see [`HANDOFF.md`](HANDOFF.md) |
