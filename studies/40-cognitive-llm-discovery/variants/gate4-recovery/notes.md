# Variant `gate4-recovery`

**What.** Gate 4's second half (PLAN): can a short search find a program from data that the program
generated? The generating program is **new: in neither the seeds nor the library** (Han,
2026-11-04). It is Hattori2019 plus satiety, `generator/hattori_satiety.py`: Q-learning with separate
rates after reward and no reward and forgetting of the unchosen side, whose softmax inverse
temperature falls as rewards accumulate within a session, as β·exp(−satiety_rate × rewards so far).
No literature model changes within a session. The search uses the engine of the search to come: the
same seeds and islands, effort medium, and one repair attempt per edit.

| setting | value |
|---|---|
| target (generating program) | `generator/hattori_satiety.py`: 6 parameters, 3 latent variables (`q_left`, `q_right`, `rewards_so_far`) |
| generating parameters | its own 2-level EM fit (at most 100 iterations) of each real template mouse: generation[0:24], validation[0:24], validation[24:48] (`mevolve.gate4 --fit-truths`). The selection cohort is not touched. |
| simulated mice | 24 generation, 24 validation and 24 held-out mice, one per template mouse, each with that mouse's sessions (count, lengths, stages, order), each session's parameters drawn around the mouse's fit, playing the coupled block task with baiting (`--simulate-only`) |
| library in the prompt | all 19 literature programs, except any the pre-check leaves out |
| seeds | the template on island 0; Miller-RHG, Lopez-DoubleTrace, Lebedeva-PR and Eckstein-BI on islands 1 to 4 (ADR 0006) |
| search | 100 proposals by Claude Sonnet 5.5 at effort medium, one repair attempt per edit, a $8 cap |
| evaluator | 2-level EM, at most 20 EM iterations, 2,048 draws: the search's settings for every score here |
| code | MEvolve at the launch record's commit (AllenNeuralDynamics/aind-dynamic-foraging-mevolve#51), ShinkaEvolve pinned at 8adc053a |

**Why.** If the search cannot reach a program it is searching toward, a failure of the real search
would say nothing about mouse behaviour. Gate 4 checks this before the $450 search, here with a
mechanism the search is not shown.

**Pre-registered (2026-11-04, before any fit or score of the new program).**

1. **Target:** the generating program's own score on the simulated validation mice, with the
   search's settings.
2. **Pre-check, before any Claude call** (`precheck.sbatch`, `mevolve.search.precheck`):
   - Score the template, all 19 library programs and the target on the same mice.
   - Leave out of the prompt any program within 0.4 of the target (twice the tolerance).
   - Stop and ask Han if the target's own 2-level fit fails, if a seed or more than 3 programs would be left out, or if the template is within 0.4.
3. **If satiety is too weak to matter:** if the pre-check stops because library programs come within
   0.4 of the target, and the fitted satiety is the reason, the mice are simulated once more. Every
   fitted parameter is kept except `satiety_rate`, which is set, for every mouse, to the 75th
   percentile of the fitted typical values over all 72 mice; its fitted spread is kept. The pre-check
   is then run once more. If it still stops, or if the real mice show no satiety at all (most fits at
   the lower bound), Han decides.
4. **Pass:** the best validation score of the search is at least the target − 0.2 on the ×100
   scale, the PLAN's 0.002 in pooled likelihood. The test is one-sided, because on finite data a
   program may beat the one that generated it. On 24 mice the paired standard error between two
   programs is about 0.045 to 0.08, so 0.2 is about 3 to 4 standard errors.
5. **Winner's curse:** the winner is chosen on the validation mice, so its score there is biased
   upward. The winner and the target are each scored once on the 24 held-out mice (`heldout.py`),
   and both numbers are reported. The pass is decided on validation, as written in step 4.

**What a pass shows.** The search can find a within-session mechanism that none of the programs it
is shown has. **A failure is ambiguous:** the search may be weak, or a counter-based drift may simply
be hard to invent in 100 proposals. Han chose this knowingly.

**Steps.**
1. `truths.sbatch` fits the generating program to the template mice and simulates the three sets.
2. `precheck.sbatch` writes `precheck.json`; its exclusions go into `run.json` in a new commit.
3. `search.sbatch` runs the search and writes the launch record.

**Superseded first design: Hattori2019 as the target (2026-11-04).** The first design used Hattori2019
(simulated mice in `gate4/run_85de84f/` and `gate4/second_half_cee79da/`). Its pre-check: SLURM 27609017, from studies 6e92764 and MEvolve cee79da, wrote
`runs/gate4-recovery/precheck-6e92764/precheck.json`. The target, Hattori2019, scores 66.528, and its
2-level fit did not fail. Margins below the target:

| program | margin | | program | margin |
|---|---|---|---|---|
| Bari2019 | 0.301 | | Eckstein-BI (seed) | 1.266 |
| Lebedeva-PR (seed) | 0.394 | | Beron-RFLR | 1.367 |
| Miller-RHG (seed) | 0.453 | | Chen-RLCK | 1.378 |
| Lopez-DoubleTrace (seed) | 0.502 | | Eckstein-RL | 1.402 |
| CompareToThreshold | 0.651 | | FeedbackDependent-RL+Bias+CK1 | 2.261 |
| Grossman-MetaLearning | 0.832 | | Alsio-DualRateSticky-RL | 3.077 |
| Zid-HistoryKernel2-ForagingRL | 1.037 | | template | 3.575 |

The other five programs are 4.7 to 13.6 below. The rule leaves out Bari2019 and Lebedeva-PR, and
Lebedeva-PR is a seed, so the pre-registered stop fired and the choice went to Han. Two seeds sit at
the rule's edge: Lebedeva-PR is inside by 0.006, and Miller-RHG is outside by 0.053, both within
noise of the limit. The pre-check was run again from MEvolve 1c57149, which adds the stage fix
(AllenNeuralDynamics/aind-dynamic-foraging-mevolve#49), so the search's seed scores come from the same
code (SLURM 27609980, studies 1f1371a). All 20 scores were identical (largest difference 0.0); its
result is `precheck_hattori2019.json` here. Han then chose a generating program in neither the
seeds nor the library, the current design above.

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group `gate4-recovery@<launch_id>` in `NeuralDynamics/dfbfm-mevolve` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Status.** Not launched.
