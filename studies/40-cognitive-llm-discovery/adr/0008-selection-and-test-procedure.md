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

MEvolve at the commit that merges #81 (per-mouse 2-level fallback beyond 0.005 nats per trial; MEvolve
ADR 0004 amendment), 2-level Laplace-EM, seed 0, EM run to convergence (`max_em=100`, MEvolve ADR
0004), 2,048 draws, fit on each subject's even sessions and scored on its odd ones. Candidates, the
reference and later the floor programs are all scored this way. The commit's full SHA is added to this
record before the first selection score.

## Selection score

Each candidate's and the reference's pooled held-out likelihood per trial on the 100 selection
subjects (`reference/cohorts_seed0.json`).

## Fingerprint gate (ADR 0001)

Computed with study 15's library (`aind-dynamic-foraging-models` 392c504, fingerprint spec v1), on
the selection subjects' score sessions, for each candidate and the reference alike:

- **Replay gap**: the mean, over the replay statistics `losses`, `pattern` and `hazard` (ADR 0001's
  three), of the library's `replay_gap` per subject, averaged over subjects with equal weight
  (`aggregate_subjects`).
- **Rollout gap**: the library's `rollout_gap` for `hazard` (switching by last reward and run length,
  ADR 0001's rollout statistic), averaged over subjects with equal weight. One rollout per score
  session in session-matched gym tasks (study 15 ADR 0006; its `p2_task.build_task`), with
  `aind-behavior-gym` 0.4.5, `perseverative_limit=4`, `max_block_tally=3`. Each rollout draws its
  parameters from the 2-level fit, θ = μ + exp(log σ)·ε in fitting space (study 15 ADR 0005).
- **Sessions excluded from rollouts** (they still count for likelihood and replay): sessions with no
  recorded block parameters, uncoupled sessions with a single reward probability, and coupled sessions
  whose pool has no unequal pair (study 15's gym traps). Every model uses the same session set.
- **Paired seeds**: the task stream is seeded without the model (`SeedSequence([0, crc32(session_id),
  k])`), so candidates and the reference play the same tasks; the agent stream adds the model.
- **Rule**: a candidate passes if its replay gap and its rollout gap are both no larger than the
  reference's. The search's best is the passing candidate with the highest selection score; a failed
  candidate is replaced by the next rank.

## Test scoring

The chosen program is scored once on the 149 test subjects, and once on the GRU's own 149 (ADR 0003),
with the evaluator above. The headline is pooled over the 146 shared subjects, beside HB-CTT (study 13
r5) and GRU v2 H128 (study 01), both re-pooled from their committed per-subject values: **gap closed**
= (program − HB-CTT) / (GRU − HB-CTT). The same applies to each floor program later.

## Order

1. This record merged; the MEvolve SHA filled in.
2. Selection: likelihood, replay and rollout for the candidates and the reference.
3. The simplification design (a variant's `run.json` and notes) committed **before** the search's
   best is scored on the test subjects, so the test score cannot shape it.
4. The search's best scored on the test subjects.
5. Simplification at 90%, 75% and 50%; selection and gate for each; one test score each.

## Open question for Han (decide before step 2)

Study 15 r3 found that the rollout `hazard` statistic is matched well by every good model, so it will
barely separate candidates; in rollout, `losses` and `post_switch` are where models differ (2-level fits
often lose there). ADR 0001 fixes `hazard`, and this record keeps it unless Han changes it now:

- **Default (ADR 0001 as written)**: rollout gap = `hazard`.
- **Alternative**: rollout gap = the mean over `losses`, `hazard` and `post_switch`, an amendment to
  ADR 0001 made before any rollout is run.

## Considered options

- **Choosing the search's best by the monitor scores**: forbidden by the ADR 0003 amendment.
- **Testing before the simplification design is fixed**: the test score could shape the
  simplification; step 3 prevents it.
- **Replay fingerprints only**: cheaper, but ADR 0001 makes both a gate; study 15's library makes the
  rollouts cheap enough.
