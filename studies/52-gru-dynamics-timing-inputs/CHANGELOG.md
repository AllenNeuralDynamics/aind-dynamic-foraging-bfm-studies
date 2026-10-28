# Changelog

## 2026-10-28

- Migrated from the dispatcher's `studies/pk-03-gru-dynamics-timing-inputs/` (never committed there).
  Study pk-03 is now 52; older notes and W&B text may say pk-03 or 14.
- Layout for this repository: `source_runs.json` moved to `analysis/`, variant `base` added,
  non-`rN` summaries moved to the study root, report frontmatter on schema v1.
- PNG figures are not committed (gitignored) until they are redrawn as SVGs.

## 2026-10-23

- Created the study as the response-input counterpart of Study 50. Surveyed W&B: the only
  GRUs with RT / lick inputs are Study 07's (`mice_rt_lick_scaling`). Its D=614 / H128
  cells (OFF, RT-only, licks-only, both, shuffled; 3 seeds each) all train on the same 614
  mice and snapshot as Study 50's model C, so C is the baseline.
- Pinned R = RT-only arm `h128-dscan-rt-only@20260823-093058`, 3 seeds
  (`make freeze-sources`). Held-out likelihoods match Study 07's `timing_scaling.json`;
  cohort identical to C-s0. Source wrapper commit `78b8d11`.
- Wrapper `gru_trial_states` (shared checkout, uncommitted): passes the response columns
  through when a run has them (`RESPONSE_COLUMNS`), with 2 new unit tests (14 tests OK).
  Timing-off runs are unaffected because the columns are absent.

## 2026-10-24

- A1 extraction for R seed 0 on GPU (`make extract MODEL=R-s0`): 4,339,402 eval trials,
  8,903 sessions, 614 mice; logged eval likelihood reproduced (diff 4e-9), readout error
  1.3e-3. The table adds `prev_log_rt_input`, the RT channel the GRU received.
- A2 (`analysis/a2_align_with_c.py`): R-s0's eval split is identical to C-s0's (same
  sessions, trials, actions and rewards). R − C = +0.00087 pooled likelihood.
- Found that RT is missing for 24% of eval trials, as whole sessions (1,659 sessions,
  108 mice; mostly Coupled Baiting), where the RT input is a constant 0. R gains nothing
  there (−0.00015) and +0.00123 in RT-present sessions, so the gain is from RT itself, not
  a session flag. RT analyses default to RT-present sessions.
- A1 for R seed 1 (seed-noise baseline): logged likelihood reproduced (diff 2e-8).
- Tier 1 (`b0_where_rt_helps.py`, figs B0a–f) and tier 2 (`b3_state_rt.py`, figs B3a–d);
  write-up in `analysis/reports/r1-rt-model-descriptive.md`. Headlines: mice switch more after
  slow RTs (also 2–3 trials back); C flattens that curve and R matches it. R stores RT on a
  large, readout-orthogonal axis shared across input clusters, and keeps a multi-trial RT
  memory.
- Fixes during the run: raw RTs of exactly 0 broke log RT (now clipped to the encoder's
  [1 ms, 10 s]); `spearmanr(nan_policy="omit")` on 3M rows took ~85 min (now drop NaNs first,
  and B0 runs in ~1 min); B0b redesigned around a seed-noise baseline.
- Tier 3 (`i_rt_mechanism.py`, figs I_RT1, C1_RT, I_RT2; report `r2-rt-mechanism.md`). The
  replay reproduces the logged likelihood (4e-7), and the `xs` RT column equals
  `prev_log_rt_input` exactly. Slow RT pushes toward switching; very fast RT after a reward
  also does (U-shape). The effect is a state-dependent gain (larger after losses and when
  confident). A 1 sd RT is about a quarter of a reward flip and fades within about 3
  trials. Clamping RT to any constant puts R below C, and shuffling it is worst.
- **Correction to r1:** "additive push, similar after wins and losses" was wrong; r1 is
  amended in place, pointing to r2.
- **Independent code review** of the tier 1–3 code and reports. No bug changed a figure's
  data: alignment, signs, subsets and bootstraps verified against the data. Claims
  corrected, each listed in the report's "Corrections" section:
  - r1: older-RT effects are reproduced by C conditionally (new B0g); seed noise is about
    half, not most, of the R−C disagreement; PC3 is RT mixed with reward; B3d is
    observational (C shows 0.19–0.31 logit/sd without RT) and outlier-sensitive;
    low/mid → mid confidence.
  - r2: the fast-end reversal is after wins, seed-dependent, and does not explain B0a's U;
    the minimum location differs by outcome; the clamp interpretation is ambiguous.
- Code changes from the review:
  - B3: nuisance history extended to lag 6; RT z winsorized at ±4; PC–variable
    correlations and R−C logged.
  - B0: new B0g conditional lag fit (mice, R, R-s1, C); R-s1 added to the B0c errors.
  - I-RT: parameterized by seed; confidence measured at RT input 0; clamp means over real
    RT inputs only; inputs without an RT left at 0.
- Seed repeat: `i_rt_mechanism.py R-s1` (replay diff 2e-7) replicates slopes, impulses and
  clamps.
- Follow-ups F1–F3 (`i_rt_followups.py`, `fig_i_rt_followups.py`, report r3):
  - RT's effect scales with commitment to staying (push vs shrink not separable: 98–99% of
    states favour staying);
  - outcomes after very fast responses count about 45% less;
  - the previous RT only weakly gates learning.
- Fixed during the run: a literal newline broke `b0_where_rt_helps.py` once (caught on
  rerun); figure label overlaps in I-RT1, C1-RT, I-RT2, F1, F2/F3.
- r4:
  - B0h: in the mice, the outcome's effect on switching is about 24% smaller after their
    fastest responses (0.64 vs ~0.84 log-odds); both R seeds reproduce it and C does not.
  - F-RT: C's embedding carries a little mouse-level RT (rank R² 0.085) and R's carries none;
    R's embedding is C's up to a linear map (R² 0.87; R seed-to-seed 0.93).
  - D-RT: one stable "stay" attractor per input at every RT, but RT moves it inconsistently
    across sides and seeds — a negative result. RT acts through the single-step update.

## 2026-10-26

- **Second independent review** of everything added after the first (B0g, B0h, B3 changes,
  I-RT changes, F1–F3, F-RT, D-RT) and of r1–r4.
  - One real bug: `i_rt_mechanism.py` reused `key` as a loop variable, so the JSON `key`
    field read "reward flipped". Fixed; reran both seeds (numbers unchanged).
  - Analysis fixes:
    - F3 was confounded by the current RT (ρ = 0.58 with the previous RT); it is now
      measured with the current RT held at 0, and the effect vanishes;
    - B0g plots lag-1 slopes after losses and after wins (the main effect was the
      after-loss slope only), with 100 mouse resamples (was 20).
  - Report corrections (each report's "Corrections" section):
    - R − C is no longer called a decomposition;
    - B0h's C does drop at the fast end (by half as much as the mice);
    - F-RT wording, alignment direction, p floor;
    - D-RT example ranges and convergence caveat;
    - several numbers (autocorrelation, floor counts, F2 reference, clamp denominator).
  - Seed-1 figure titles now name the seed.
- **Licks model (L):** pinned 3 seeds (cohort = C's, held-out matches Study 07); extracted L-s0
  and L-s1. The extractor now records raw lick counts and the lick inputs the GRU received;
  the first patch attempt did not apply and L-s0 was re-extracted with the corrected script.
  `load_paired_trials` takes a model key (RT-model outputs reproduced unchanged), and
  `i_rt_mechanism.build` takes `n_obs`.
- r5 analyses (B0-L, B3-L, I-L, FL1–3, F-L). Headlines:
  - L gains +0.0084 over C, with 99% of mice gaining;
  - licking the other port predicts switching (mice 0.13 → 0.56 after a loss), which C
    misses;
  - in L, other-side licks act mainly as a reset of commitment;
  - L reads ~40% of the outcome from the licks;
  - the representation is per port;
  - mouse-level other-port licking stays in the embedding.
  Independent review pending.
- **Third independent review** (licks code and r5). No alignment, sign or side-assignment
  bug was found.
  - Fixed the chosen-side decile binning: tied counts piled into single bins and caused the
    decile-5 bump; ties are now broken at random, and the bump is gone.
  - F-L: added the ≥ 500-trials filter (all 614 mice qualify) and readable labels.
  - The FL figure now states its IQR bands.
  - r5 claims corrected in its "Corrections" section: the port representation is suggestive
    only, the chosen-vigour wording, FL3 fit quality, "explains" → "consistent with", the
    I-L1 slice reference, and seed coverage.
