# Changelog — 40-cognitive-llm-discovery

## 2026-11-08 (selection and test)

- ADR 0008 (#134): the selection and test procedure for `seeded-islands-2`, fixed before any selection or
  test score: five candidates by validation score (`reference/selection_candidates_seeded-islands-2.json`),
  Miller-RHG as reference, EM to convergence with MEvolve's per-mouse fallback (#81), the fingerprint gate
  with study 15's library, the test scoring and its order. One open question for Han: the rollout gate's
  statistic.

## 2026-11-06

- `seeded-islands` before its first proposal: the proposer is Claude Opus 5.5 (effort medium), and each
  evaluation is its own Slurm job of 24 CPUs under a small driver job (`eval_launch: "job"`, MEvolve #70;
  at most 48 evaluations and 24 proposals at a time), so the search uses CPUs as they free up instead of
  waiting for one block. `seeded-islands-1` (Sonnet, one block of 32 slots) never started and was cancelled;
  the rehearsal (`REHEARSAL=1`) now makes 6 real Claude calls. Notes: deviation, pre-registration, Run.
- `seeded-islands` pre-registered and ready to launch: `run.json` (12 parameters and 6 latent variables,
  $450, stopping rule, snapshots every 10 minutes) and `search.sbatch` (slots of 24 CPUs, their number
  chosen at launch; TERM 120 s before the time limit; `RESUME_OF` for a resume; `FAKE=1` for a $0
  rehearsal). ADR 0005 amended to 12 parameters (6 latent variables unchanged), after
  `seeded-islands-cap12`. The stopping rule's standard error is 0.03 (×100), the low end of gate 2's
  paired standard errors. PLAN's compute line now has the measured costs.

## 2026-11-05

- r1 is live: gate 4 has passed and the report is complete.
- Screenshots of Shinka's portal (program tree, score trace, the best program's ancestors) in r1, for
  both gate 4 searches, and in the `seeded-islands-smoke` notes, from each run's W&B archive
  (`analysis/screenshots/`, with its README). Screenshots of a tool are PNG; the SVG rule covers figures
  drawn by code (`scripts/check_figures.py` skips `screenshots/`).
- `seeded-islands-cap12` result (pass): `analysis/freeze_seeded_islands_cap12.py` (`make
  freeze-seeded-islands-cap12`) freezes the run's W&B archive and Slurm steps;
  `analysis/seeded_islands_cap12.py` (`make cap12-result`, in `all`) writes the pre-registered numbers
  against the smoke run and `fig_seeded_islands_cap12.svg`. 37 of 89 scored proposals used the extra
  room, nearly all of it parameters; the best score is within noise of the smoke run's; 16 slots ran
  3.3 times faster at about the same CPU cost per proposal.
- Variant `seeded-islands-cap12` (issue #118): the `seeded-islands-smoke` setup with program limits of 12
  parameters and 8 latent variables instead of ADR 0005's 8 and 6, a deliberate test of the limits that the
  smoke run hit; the first search with evaluations on many nodes, as 16 slots of 24 CPUs chosen from the
  free-CPU landscape.
- Variant `seeded-islands-smoke` (issue #115): PLAN step 5's 100-proposal search on real mice, in the
  `seeded-islands` configuration ($20 list), to measure node time and dollars per proposal on real mice
  and run the production configuration end to end. `analysis/seed_choice.py` (`make seed-choice`, in
  `all`) applies ADR 0006's seed rule to study 12 r3's 2-level EM fits on the validation mice:
  Miller-RHG, Lopez-DoubleTrace, Lebedeva-PR, then Eckstein-BI.

## 2026-11-04

- r1 (draft): gate 4, program recovery on simulated mice: both halves, the seeded and no-seed searches,
  repeats and the novelty filter, and compute for the main search. Both runs are archived as W&B artifacts;
  the freeze keeps one row per program. The PLAN's report table is renumbered (recovery is r1).
- Variant `gate4-noseed` (issue #110): gate 4's control arm, the `gate4-recovery` search with no seed
  programs, on the same simulated mice.
- `gate4-recovery`: the pre-check passes for all three hand-set settings; setting a is used (target
  72.928; the reference without the slow trace 0.429 below, the seeds 3.5 to 6.6 below).
- `gate4-recovery`: a hand-set design, as simple as the engine smoke test (Han). The generating program
  has two reward timescales plus a choice kernel (`generator/two_timescale_ck.py`), with hand-set values
  (`make_truths.py`), not fits to mice; three simplified seeds (`seeds/`), the library off, and a
  reference without the slow trace that the target must beat by 0.4. The satiety design is kept as
  superseded: the mice showed almost no satiety (0.015 above Hattori2019).
- `gate4-recovery`: Han chose a generating program in neither the seeds nor the library, Hattori2019
  plus satiety (`generator/hattori_satiety.py`), after the Hattori2019 design's pre-check stopped on the
  seed Lebedeva-PR; its parameters come from its own 2-level fit of the template mice
  (AllenNeuralDynamics/aind-dynamic-foraging-mevolve#51). The Hattori2019 attempt is kept as superseded.
- Variant `gate4-recovery` (issue #105): gate 4's second half, a 100-proposal search on mice simulated from
  Hattori2019 with Hattori2019 left out of the prompt, pre-registered with a pre-check of every program
  Claude is shown; code in AllenNeuralDynamics/aind-dynamic-foraging-mevolve#48.
- Cohorts (issue #103): the seed-0 draft is adopted as drawn and committed as `reference/cohorts_seed0.json`
  (sha256 `cbe2cd02...c7cb`); ADR 0003's amendment records its checks against the test subjects.
- Gate 3's numbers are regenerable (issue #97, review of #98): `make freeze-engine-smoke` freezes the
  engine-smoke ledgers, Shinka's program table and the hand-typed Console figures into
  `analysis/engine_smoke/`, and `make gate3` (in `all`) writes `analysis/gate3_engine_smoke.json`, which
  the notes' Result section now cites; the ledger's cost reads $0.616 (was $0.615), and the Console's
  Caching page is dated 11:00 UTC, not 19:48.

## 2026-11-03

- Variant `engine-smoke` (issue #97): gate 3's engine smoke, two 10-proposal ShinkaEvolve runs
  (Claude Sonnet 5.5 at effort medium and high) on 8 + 8 smoke subjects drawn after the draft cohorts,
  with the five-island seed layout of ADR 0006; code in AllenNeuralDynamics/aind-dynamic-foraging-mevolve#32.
- Gate 3 passes (`engine-smoke@20261103-030341`, results in `variants/engine-smoke/notes.md`): 22 Claude
  calls with no errors; the ledger's token counts match the Console and its cost is 4% above the
  Console's list price; caching works; seeds sit on their islands. Effort medium is kept for the search:
  high cost 38% more per proposal and improved fewer parents. PLAN.md notes that its budget is at list
  price, about twice the bill after the organisation's discount.
- W&B (issue #99): new runs log to `NeuralDynamics/dfbfm-mevolve`, shared with MEvolve; the README
  frontmatter and the engine-smoke run configs name it. `engine-smoke@20261103-030341` stays in
  `aind-dynamic-foraging-model`.

## 2026-11-02

- "Session level" retired (issue #87): fits are named by their levels and method (per session, per
  subject, 3-level HB, 2-level HB, 2-level EM; table in MEvolve's `CONTEXT.md`). ADR 0002 renamed to
  `0002-fit-programs-with-the-2-level-model.md`. PLAN.md cites study 12 r3 (`r3_2level_em.json`) for
  the literature models as 2-level EM, and ADR 0006 states the seed-order rule once.

## 2026-10-31

- Study created (issue #73): README design and motivation, `seeded-ctt` variant, PLAN.md (draft,
  before the design grill), CONTEXT.md, ADR 0001 (one score, then simplify) and ADR 0002 (fit programs
  at the session level). Adds the `cognitive/mevolve` model tag.
