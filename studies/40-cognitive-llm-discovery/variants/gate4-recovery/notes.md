# Variant `gate4-recovery`

**What.** Gate 4's second half (PLAN): can a short search find a program from data that the program
generated? As simple as the engine smoke test (Han, 2026-11-04): the generating program is
**hand-set, not fitted to mice**, the seeds are **simplified programs** that each lack part of it,
and the literature library is **off**. The generating program, `generator/two_timescale_ck.py`, has
reward history at two timescales plus a choice kernel: a fast and a slow reward trace for each side
(learned on the chosen side), a choice kernel, and a side bias. The search uses the engine of the
search to come: islands, effort medium, and one repair attempt per edit.

| setting | value |
|---|---|
| target (generating program) | `generator/two_timescale_ck.py`: 7 parameters, 5 latent variables |
| generating parameters | hand-set (`make_truths.py`): every simulated mouse has the same typical values, and each session's parameters are drawn around them with a spread of 0.3 on the probit scale. Three settings, a to c, from mild to strong mechanisms. |
| simulated mice | 24 generation, 24 validation and 24 held-out mice. Each borrows only its session layout (count, lengths, order) from a real template mouse: generation[0:24], validation[0:24], validation[24:48]. They play the coupled block task with baiting (`mevolve.gate4 --simulate-only`). The selection cohort is not touched. |
| library in the prompt | off (`library_in_prompt: false`): Lopez-DoubleTrace already has two timescales |
| seeds | the template (side bias + stickiness) on island 0; `seeds/q_learning.py` (one timescale), `seeds/choice_kernel.py` (no reward learning) and `seeds/two_timescale.py` (no kernel) on islands 1 to 3 |
| reference, scored but never shown | `generator/one_timescale_ck.py`: the generating program without its slow trace |
| search | 100 proposals by Claude Sonnet 5.5 at effort medium, one repair attempt per edit, a $8 cap |
| evaluator | 2-level EM, at most 20 EM iterations, 2,048 draws: the search's settings for every score here |
| code | MEvolve at the launch record's commit (AllenNeuralDynamics/aind-dynamic-foraging-mevolve#51), ShinkaEvolve pinned at 8adc053a |

**Why.** If the search cannot reach a program it is searching toward, a failure of the real search
would say nothing about mouse behaviour. Gate 4 checks this before the $450 search. A hand-set
generator lets each mechanism matter by a set margin, which fitted ones did not (the superseded
designs below).

**Pre-registered (2026-11-04, before any simulation of this program).**

1. **Target:** the generating program's own score on the simulated validation mice, with the
   search's settings.
2. **Pre-check, before any Claude call** (`precheck.sbatch`, `mevolve.search.precheck`): score the
   template, the three seeds, the reference and the target on the same mice. Stop if the target's own
   2-level fit fails, or if the template, a seed or the reference comes within 0.4 of the target
   (twice the tolerance): then the search could pass without the slow trace or without the kernel.
3. **Calibration:** simulate and pre-check setting a; if it stops, b; then c. The first setting that
   passes is used, and its directory is fixed in `search.sbatch`. If c stops too, Han decides.
4. **Pass:** the best validation score of the search is at least the target − 0.2 on the ×100
   scale, the PLAN's 0.002 in pooled likelihood. The test is one-sided, because on finite data a
   program may beat the one that generated it.
5. **Winner's curse:** the winner is chosen on the validation mice, so its score there is biased
   upward. The winner and the target are each scored once on the 24 held-out mice (`heldout.py`),
   and both numbers are reported. The pass is decided on validation, as written in step 4.
6. **Mechanism recovery, reported but not part of the pass** (added 2026-11-04, before any pre-check
   result): whether the winner has a slow reward trace and a choice kernel (its code and
   `LATENT_NAMES`), and which island it came from.

**What a pass shows.** The search can combine and extend simplified programs into a mechanism none
of them has, on data where that mechanism clearly matters. The target is the two-timescale seed plus
the choice-kernel seed, so a pass shows recombination under this evaluator. It does not show that
the search finds subtle mechanisms in real mice. **If it fails,** first check whether proposals
combined mechanisms across islands at all (migration every 10 proposals at rate 0.1 is slow) before
concluding that Claude could not.

**Steps.**
1. `truths.sbatch` (with `SETTING`) writes the hand-set truths and simulates the three sets.
2. `precheck.sbatch` (with `SIM_DIR`, step 1's directory) writes `precheck.json` there.
3. `search.sbatch` runs the search and the held-out check; the launch writes its record.

**Superseded second design: Hattori2019 plus satiety (2026-11-04).** The target was Hattori2019 whose
softmax inverse temperature falls as rewards accumulate, β·exp(−satiety_rate × rewards so far)
(`generator/hattori_satiety.py`), fitted by 2-level EM to each real template mouse. The real mice show
almost no satiety: on the 24 real generation mice it beat Hattori2019 by only 0.015 (66.378 against
66.363; median satiety_rate 0.0008 per reward), and by 0.030 on its own simulated mice. A pass would
not have needed satiety, so Han stopped this design before its pre-check and chose the hand-set one
above. Its fits and simulated mice are in `gate4/satiety_8e38b20/`.

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
seeds nor the library: the satiety design, then the hand-set one.

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group `gate4-recovery@<launch_id>` in `NeuralDynamics/dfbfm-mevolve` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Status.** Not launched.
