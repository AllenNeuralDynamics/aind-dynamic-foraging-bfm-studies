# Variant `gate4-recovery`

**What.** Gate 4's second half (PLAN): can a short search find a known program from data that the
program generated? Mice are simulated from Hattori2019, the prompt's library leaves Hattori2019 out,
and the search must reach Hattori2019's own score. It uses the engine of the search to come: the
same seeds and islands, effort medium, and one repair attempt per edit.

| setting | value |
|---|---|
| target (generating program) | Hattori2019, which passed gate 4's first half (AllenNeuralDynamics/aind-dynamic-foraging-mevolve#36) |
| simulated mice | 24 generation mice, those of the first half, templated on generation[0:24]; 24 validation mice, templated on validation[0:24]; 24 held-out mice, templated on validation[24:48]. Each keeps the real mouse's sessions (count, lengths, stages, order), with each session's parameters drawn around study 12's 2-level fit of that mouse, playing the coupled block task with baiting. The selection cohort is not touched. |
| files | generation: `gate4/run_85de84f/simulated_Hattori2019.json.gz` (sha256 `3c85ab19…f281`); validation and held-out: `gate4/second_half_cee79da/simulated_Hattori2019_{validation,heldout}.json.gz`, each with a manifest of digests. The validation file's sha256 is `fdbb4376…4930`. All are under `/allen/aind/scratch/han.hou/mevolve/`, drawn from study 12's fits at studies 9aae8e6, identical to those the first half read at c2bc537. |
| library in the prompt | the 19 literature programs except Hattori2019, and any the pre-check leaves out |
| seeds | the template on island 0; Miller-RHG, Lopez-DoubleTrace, Lebedeva-PR and Eckstein-BI on islands 1 to 4 (ADR 0006) |
| search | 100 proposals by Claude Sonnet 5.5 at effort medium, one repair attempt per edit, a $8 cap |
| evaluator | 2-level EM, at most 20 EM iterations, 2,048 draws: the search's settings for every score here |
| code | MEvolve at the launch record's commit (AllenNeuralDynamics/aind-dynamic-foraging-mevolve#48), ShinkaEvolve pinned at 8adc053a |

**Why.** If the search cannot reach a program it is searching toward, a failure of the real search
would say nothing about mouse behaviour. Gate 4 checks this before the $450 search.

**Pre-registered (2026-11-04, before any score on these mice).**

1. **Target:** Hattori2019's own score on the simulated validation mice, with the search's settings.
2. **Pre-check, before any Claude call** (`precheck.sbatch`, `mevolve.search.precheck`):
   - Score the template and all 19 library programs on the same mice.
   - Leave out of the prompt the target, and any program within 0.4 of the target (twice the tolerance).
   - Stop and ask Han if the target's own 2-level fit fails, if a seed or more than 3 other programs would be left out, or if the template is within 0.4.
3. **Pass:** the best validation score of the search is at least the target − 0.2 on the ×100
   scale, the PLAN's 0.002 in pooled likelihood. The test is one-sided, because on finite data a
   program may beat the one that generated it. On 24 mice the paired standard error between two
   programs is about 0.045 to 0.08, so 0.2 is about 3 to 4 standard errors.
4. **Winner's curse:** the winner is chosen on the validation mice, so its score there is biased
   upward. The winner and the target are each scored once on the 24 held-out mice (`heldout.py`),
   and both numbers are reported. The pass is decided on validation, as written in step 3.

**What a pass shows.** The engine reaches a known standard model that it was not shown. Claude knows
Hattori-style models from training, so a pass does not show that the search invents new mechanisms.
That needs a generating program outside the seeds and the library, a later variant (Han asked about
it on 2026-11-04).

**Steps.**
1. `precheck.sbatch` writes `precheck.json`.
2. Its exclusions go into `run.json`'s `library_exclude` in a new commit.
3. `search.sbatch` runs the search and writes the launch record.

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group `gate4-recovery@<launch_id>` in `NeuralDynamics/dfbfm-mevolve` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Status.** Not launched.
