# Selection and test procedure for the search's programs, fixed before any selection or test score

**Status:** proposed (2026-11-08); fixes how ADR 0001 and ADR 0007 are carried out for
`seeded-islands@20261106-114959`. Written before any program was scored on the selection or test
subjects (issue #134).

The search ended by its stopping rule on 2026-11-08 (2,562 proposals). Han asked to score the search's
best on the test subjects first, then simplify at the three floors and score those (2026-11-08). That
is ADR 0007 done in two stages; this record fixes every choice those stages make, so no selection or
test score can shape it.

## Candidates and reference

- **Candidates**: the five programs ranked first by the search's validation score, distinct by code,
  from the final database. Ranks 6 to 20 are the order of replacements if a candidate fails the
  fingerprint gate (ADR 0007). The list, the database's sha256 and the rule are in
  [`reference/selection_candidates_seeded-islands-2.json`](../reference/selection_candidates_seeded-islands-2.json).
- **Reference** (ADR 0001, ADR 0005): Miller-RHG, the literature model with the best 2-level EM score
  on the validation subjects, read from study 12's `two_level_em_fits.csv` (study 12 r3).
- **Not used**: the monitor scores of the archive (ADR 0003 amendment); they never enter the choice.

## Evaluator

MEvolve `146153c01f6391122414525022e9126f200a1586` (the merge of MEvolve PR 82: per-mouse 2-level
fallback beyond 0.005 nats per trial, MEvolve #81 and its ADR 0004 amendment), 2-level Laplace-EM, seed 0, EM run to convergence (`max_em=100`, MEvolve ADR
0004), 2,048 draws, fit on each subject's even sessions and scored on its odd ones. Candidates, the
reference and later the floor programs are all scored this way.

## Selection score

Each candidate's and the reference's pooled held-out likelihood per trial on the 100 selection
subjects (`reference/cohorts_seed0.json`).

## Fingerprint gate (ADR 0001)

Computed with study 15's library (`aind-dynamic-foraging-models` 392c504, fingerprint spec v1), on
the selection subjects' score sessions, for each candidate and the reference alike:

- **Replay score**: study 15's replay fingerprint score (its ADR 0009): the geometric mean, over the
  four rate statistics `losses`, `pattern`, `hazard` and `post_switch`, of the floor ratio (the gap
  between the model's predicted and the subjects' observed switch rates, over the binomial floor,
  each balanced over subjects), on the subjects' `all` session set (`replay_fingerprints`,
  `aggregate_subjects`, `fingerprint_score`).
- **Rollout score**: study 15's rollout fingerprint score: the geometric mean of the floor ratios
  (gap over the split-half floor) of the same four rate statistics and the `run_length` distribution.
  Ten rollouts per score session (study 15's `N_ROLLOUTS`) in session-matched gym tasks (study 15 ADR 0006; its
  `p2_task.build_task`), with `aind-behavior-gym` 0.4.5, `perseverative_limit=4`,
  `max_block_tally=3`. Each rollout draws its parameters from the 2-level fit,
  θ = μ + exp(log σ)·ε in fitting space (study 15 ADR 0005).
- **Sessions excluded from rollouts** (they still count for likelihood and replay): sessions with no
  recorded block parameters, uncoupled sessions with a single reward probability, and coupled sessions
  whose pool has no unequal pair (study 15's gym traps). Every model uses the same session set.
- **Paired seeds**: the task stream is seeded without the model (`SeedSequence([0, crc32(session_id),
  k])`), so candidates and the reference play the same tasks; the agent stream adds the model.
- **Rule**: a candidate passes if its replay score and its rollout score are both no larger than
  the reference's (lower is closer to the subjects). The search's best is the passing candidate with the highest selection score; a failed
  candidate is replaced by the next rank.

## Test scoring

The chosen program is scored once on the 149 test subjects, and once on the GRU's own 149 (ADR 0003),
with the evaluator above. The headline is pooled over the 146 shared subjects, beside HB-CTT (study 13
r5) and GRU v2 H128 (study 01), both re-pooled from their committed per-subject values: **gap closed**
= (program − HB-CTT) / (GRU − HB-CTT). The same applies to each floor program later.

## Order

1. This record merged.
2. Selection: likelihood, replay and rollout for the candidates and the reference.
3. The simplification design (a variant's `run.json` and notes) committed **before** the search's
   best is scored on the test subjects, so the test score cannot shape it.
4. The search's best scored on the test subjects.
5. Simplification at 90%, 75% and 50%; selection and gate for each; one test score each.

## Gate statistics: more of them, decided by Han

Study 15 r3 found that the rollout `hazard` statistic, ADR 0001's only rollout statistic, is matched
well by every good model and barely separates them; in rollout, `losses` and `post_switch` are where
models differ. Han chose to use more fingerprints, including those that differ (2026-11-08). The floor
ratios put statistics of different scale, rates and a distribution, on one footing, so each mode's
score uses all of its statistics. This amends ADR 0001 before any selection subject was rolled out
or replayed.

## Amendment (2026-11-08, before any selection subject was scored): ten rollouts per session, and the code

The record said one rollout per score session, a misreading of study 15's note (one parameter draw per
rollout). Study 15 plays each session ten times (`N_ROLLOUTS = 10`), which keeps the rollout tallies far
less noisy at little cost; ADR 0008 now does the same. The procedure runs as
`analysis/freeze_selection_tasks.py` (session groups and gym tasks for a cohort; on the 149 test subjects
it reproduces study 15's committed groups, specs and exclusions exactly) and `analysis/freeze_selection.py`
(one program on one cohort: score, replay and rollout fingerprint scores), piloted on the generation
subjects before the selection subjects. A fallen-back mouse (MEvolve ADR 0004) is replayed and rolled out
with its per-subject fit, as it is scored.

## Amendment (2026-11-08, before any candidate's selection score was seen): the mouse-average curves

Han added the error of the mouse-average curves to the gate (2026-11-08): study 15's Figure 5 view, where
each point is one bin of a statistic, the subjects' mean against the model's, and the GRU is uniquely
close (study 15 r2 and r3, Figure 6). The floor ratios balance each subject's gap against its own noise;
this term asks whether the model reproduces the average mouse's curve, so it adds a second, different
view of the same statistics.

- **Curves**: per statistic, each subject's value per bin where its own tally keeps the bin (study 15's
  `kept_bins`), averaged over the cohort's subjects, for the subjects and for the model (replay: the
  predicted rate on the same trials; rollout: the rate in the subject's pooled rollouts). This is study
  15's `_per_bin`, applied to one cohort.
- **Curve score**: the RMSE over the bins where both means exist, each bin once, per statistic; then the
  geometric mean over the mode's statistics (the same four in replay, five in rollout), so statistics of
  different scale weigh alike and the comparison with the reference is a ratio.
- **Rule**: a candidate passes if its replay and rollout fingerprint scores **and** its replay and
  rollout curve scores are all no larger than the reference's. Ties to the selection score as before.
- **Timing**: when this was decided, only the reference had finished on the selection subjects; the five
  candidates' jobs were cancelled before writing any output and rerun with the subjects' tallies kept.
  The reference was rerun the same way, so all six use one code path.

## Amendment (2026-11-08, before any simplification run): the floors and the floor programs

ADR 0007's "best of five" is the highest selection score for the search's best. A floor program is
scored by its Halstead effort above a floor, so its five candidates and its choice are fixed here:

- **Floors** (each run's check): from Miller-RHG's and the chosen search-best's validation scores,
  both rescored with the search's evaluator settings (`max_em` 20, 2,048 draws, seed 0) and the
  per-subject fallback (`analysis/freeze_floor_anchors.py`), so the floor and the run's check share one
  scale. floor = low + share × (high − low), share 0.9, 0.75, 0.5 (`analysis/simplify_floors.py`,
  `analysis/simplify/floors.json`). ADR 0005 named study 12 r3 for the low end; that score is at
  `max_em` 100 and without the fallback, so it is not on the run's scale.
- **Seeds**: the five selection candidates whose rescored validation score reaches the floor, the
  chosen search-best on island 0 and the rest by search rank (a seed below the floor stops the run).
- **Runs**: one per floor (`variants/simplify-90`, `-75`, `-50`): Claude Opus 5.5 at effort medium,
  at most $40 each, the search's program limits and evaluator, stopping after 300 proposals without a
  correct program of lower effort (MEvolve ADR 0006).
- **Candidates**: the five correct programs of lowest Halstead effort, distinct by code, from the run's
  final database.
- **Choice**: each candidate is scored on the selection subjects like the search's candidates (this
  ADR's evaluator, `max_em` 100, the fingerprint gate as amended). The floor program is the candidate
  of lowest effort that passes the gate and whose selection score reaches the selection floor: the
  same share of the gain between Miller-RHG's and the chosen search-best's selection scores. If none
  qualifies, the next five by effort are scored.
- **Test**: the chosen floor program is scored once on the test subjects, as the search's best.

## Considered options

- **Choosing the search's best by the monitor scores**: forbidden by the ADR 0003 amendment.
- **Testing before the simplification design is fixed**: the test score could shape the
  simplification; step 3 prevents it.
- **Replay fingerprints only**: cheaper, but ADR 0001 makes both a gate; study 15's library makes the
  rollouts cheap enough.
- **Raw gaps averaged over statistics** (ADR 0001's form): mixes scales, and cannot include the
  `run_length` distribution; the floor ratios can.
