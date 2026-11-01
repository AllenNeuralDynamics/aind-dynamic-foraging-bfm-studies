---
schema_version: 1
id: 52-gru-dynamics-timing-inputs
question: "How does giving the GRU the previous trial's *response* — its reaction time (RT) and lick counts — change the computation it learns? Which of Study 50's findings on the choice/reward-only GRU (model C) survive, where does the extra input enter the dynamics, and what does it buy behaviourally?"
status: active
wandb:
  entity: AIND-disRNN
  project: mice_rt_lick_scaling
motivated_by:
- study: 07-gru-timing-inputs
  note: "previous-trial RT and lick inputs improve prediction; how do they change the computation?"
- study: 50-gru-dynamics-embedding
  note: "model C is the no-added-inputs baseline"
tags:
  model: [gru]
  question: [interpretation, more-inputs]
---
# Study 52 — GRU dynamics with previous-trial response inputs

**Question.** How does giving the GRU the previous trial's *response* — its reaction time
(RT) and lick counts — change the computation it learns? Which of Study 50's findings on
the choice/reward-only GRU (model C) survive, where does the extra input enter the
dynamics, and what does it buy behaviourally?

This is an **analysis-only study**, the response-input counterpart of
[Study 50](../50-gru-dynamics-embedding/). It trains nothing; its inputs are trained GRUs
from [Study 07](../07-gru-timing-inputs/), pinned in [`source_runs.json`](analysis/source_runs.json).
Its one variant, [`variants/base/`](variants/base/notes.md), launches nothing; the W&B
project in the frontmatter is the source study's.

## Source models

Study 07's D=614 / H128 cells (W&B project `mice_rt_lick_scaling`): 4-d subject embedding,
scalar session conditioning, 3 seeds, `best_eval` checkpoint (step 90k). All are pinned to
snapshot `20260603` and train on **exactly model C's 614 mice** (checked by
`freeze_source_runs.py` against C-s0's `resolved_subject_ids`), so **model C is the
no-added-inputs baseline**. Study 07's own timing-OFF arm (`h128-dscan-off-bigD`) is a
re-run of C's recipe: within-subject eval 0.733853 vs C-s0's 0.733851.

| key | inputs after the subject index | W&B group | held-out (3 seeds) | within-subject eval | status |
|---|---|---|---|---|---|
| `C` (Study 50) | prev choice, prev reward | `e4-e8-d614-source@20260906-195409` (`mice_data_scaling`) | 0.72887–0.72897 | 0.73377–0.73385 | baseline |
| **`R`** | + prev log RT | `h128-dscan-rt-only@20260823-093058` | 0.72973–0.72979 | 0.73472–0.73478 | **frozen, first** |
| `L` | + prev n_lick_left, n_lick_right | `h128-dscan-licks-only@20260823-093110` | 0.73652–0.73673 | 0.7421–0.7422 | **frozen, analysed (r5)** |
| `T` | + RT + both lick counts | `h128-dscan-onprem@20260822-012657` | 0.73709, 0.73715, s2 missing | 0.7427–0.7428 | planned |
| `S` | as `T`, permuted within session | `h128-dscan-shuffled@20260822-153424` | 0.72944–0.72969 | 0.7342–0.7345 | planned (control) |

**Why start with R.** It adds a single continuous channel that is nearly orthogonal to the
existing inputs (R² ≈ 0 on prev choice + reward, Study 07). That makes it the simplest case
for building the continuous-input analysis tools before moving to the two lick channels,
which carry most of the gain. The effect is small: +0.0008 held-out and +0.0009
within-subject over C. Expect per-trial differences to be concentrated, not uniform.

## Input encoding (by construction, from the wrapper)

- RT = go cue → first lick, clipped to [1 ms, 10 s], log-transformed, then standardized
  with fixed global constants (mean −1.822, sd 0.599 in log seconds; never per-mouse).
- Row `t` receives trial `t−1`'s encoded RT. "Previous" means the previous **engaged**
  trial, since `ignore_policy=exclude` drops ignored trials before the shift.
- On a session's first row, every input channel, RT included, is the padding value −1
  (≈ 0.09 s in RT units).
- **Out-of-range RTs.** 3,045 eval trials have a raw RT below 1 ms (1,883 negative, 26
  exactly 0, 1,136 in (0, 1 ms)). The encoder clips them to 1 ms, so the model receives them at about −8.5 population sd (about
  −15 within-mouse sd). Rank-based analyses (deciles) are unaffected. Regressions on RT
  winsorize at ±4 sd (B3), because these few trials would otherwise dominate the fits.
- An RT that cannot be matched is encoded as 0, the population mean. **This applies to 24% of
  eval trials**, and they come in whole sessions: 1,659 sessions and 108 mice have no RT at
  all (62% of Coupled Baiting trials, 25% of Uncoupled Without Baiting, all Random Walk).
  In those sessions the RT input is a constant 0, a potential session-type flag. A2 shows
  that R gains nothing from it, so RT analyses use **RT-present sessions** (7,076 sessions,
  506 mice, 3.26M trials) unless stated otherwise.
- The output and loss are unchanged from C (2-way choice on engaged trials). Nothing asks
  the model to predict RT.

## Pipeline

As in Study 50:

```
A. extract (needs W&B weights + JAX; GPU, or CPU with JAX_PLATFORMS=cpu)
   source_runs.json → fetch_models.py → analysis/_cache/models/<key>/run
                    → extract_trial_states.py → analysis/_cache/trial_states_<key>.parquet (gitignored)
                                              → analysis/trial_states_<key>.json (committed)
B. report (local, offline: committed files only)
```

The extractor is the wrapper's `post_training_analysis.gru_trial_states`. It now passes the
response columns (`reaction_time`, `log_reaction_time`, `n_lick_*`) through when a run has
them. The study script adds `prev_log_rt_input`, the RT channel the GRU actually received
on each row.

## Status

| step | state |
|---|---|
| R source runs frozen | done 2026-10-23 (`make freeze-sources`); held-out matches Study 07's `timing_scaling.json`, cohort = C's |
| A1 trial table, R seed 0 | done 2026-10-24 on GPU — 4,339,402 eval trials, 8,903 sessions, 614 mice; logged eval likelihood 0.734719 reproduced to 4e-9, readout error 1.3e-3 ([`analysis/trial_states_R-s0.json`](analysis/trial_states_R-s0.json)) |
| A2 alignment with C-s0 | done — identical trials; R − C = +0.00087 overall, **all of it in sessions that have RT** (+0.00123; RT-absent sessions −0.00015) ([`analysis/a2_alignment_R-s0.json`](analysis/a2_alignment_R-s0.json)) |
| A1 trial table, R seed 1 | done 2026-10-24 (likelihood diff 2e-8); used as the seed-noise baseline |
| Tiers 1–2 (B0, B3) on R-s0 | done 2026-10-24 — [`analysis/reports/r1-rt-model-descriptive.md`](analysis/reports/r1-rt-model-descriptive.md) |
| Tier 3 (RT update rule, impulse, clamps) | done 2026-10-24 — [`analysis/reports/r2-rt-mechanism.md`](analysis/reports/r2-rt-mechanism.md) |
| Code review + corrections | done 2026-10-24 — independent audit found no data bugs; report claims corrected in place (each report has a "Corrections" section); B0g, B3 nuisance/winsorizing, clamp-mean fixes |
| Seed repeat (R-s1) | done — B0c, B0g, I-RT1/C1-RT/I-RT2 replicate (except the fast-end reversal after wins) |
| Follow-ups F1–F3 | done — [`analysis/reports/r3-rt-gain-and-outcomes.md`](analysis/reports/r3-rt-gain-and-outcomes.md) |
| Behaviour, embedding, attractors (B0h, F-RT, D-RT) | done 2026-10-24 — [`analysis/reports/r4-behaviour-embedding-attractors.md`](analysis/reports/r4-behaviour-embedding-attractors.md) |
| RT model (R) | **analyses complete** (r1–r4); one-page summary for discussion: [`analysis/reports/rt-model-summary.md`](RT_MODEL_SUMMARY.md). Licks model next |
| Licks model (L) | done 2026-10-26 — L-s0/L-s1 extracted (likelihood diffs 4e-8, 5e-8); B0-L, B3-L, I-L, FL1–3, F-L; report [`analysis/reports/r5-licks-model.md`](analysis/reports/r5-licks-model.md); independently reviewed and corrected |
| Next | combined model (T: RT + licks) — does RT add anything once licks are present? |

```bash
cd /workspace/aind-dynamic-foraging-bfm-dispatcher/studies/52-gru-dynamics-timing-inputs
make freeze-sources                 # re-verify the pins
make extract MODEL=R-s0             # ~31 GB RAM peak; run alone
```
