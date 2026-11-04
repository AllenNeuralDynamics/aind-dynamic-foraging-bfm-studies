# Rollouts run in the gym, built from each session's recorded task parameters

Each rollout plays a task built with `aind_behavior_gym` from its real session's recorded values:
the session's `task` (the paradigm that ran, not `curriculum_name`, which is the programme the
mouse is enrolled in), `block_min`, `block_max` and `block_beta` per trial, baiting, and the trial
count. Uncoupled sessions keep the gym's perseverance rule with `perseverative_limit=4` and
`max_block_tally=3`, the values the real task fixes in code (`dynamic-foraging-task`
`MyFunctions.py`); the gym's default tally is 4. The simulator gets no new rules (decided with Han,
2026-11-03).

The reward-probability pairs of a coupled session are not recorded, only their sum
(`base_reward_probability_sum`), and a short session shows only some of them. The pool is the set
of pairs seen in the subject's sessions with the same task and stage, scaled to the session's sum;
P0 checks this against the curriculum definitions. Random-walk parameters are not recorded either,
so those sessions use the gym's defaults. A warm-up session (ADR 0004) rolls out with the single
pair [1, 0] scaled to its recorded base sum, which is 1 for the older Bpod warm-ups that record
only 0 and 1, so a missing parameter never turns a warm-up into a foraging task.

## Considered options

- **Copy each session's schedule.** Then only the agent differs from the real session. But a real
  coupled block can be held past its drawn length by the animal's own behaviour, so a copied
  schedule is wrong for any agent that behaves differently.
- **Gym defaults per curriculum family**, as the wrapper does today (`generative_analysis.py`
  `_build_curriculum_matched_task`). Every stage of a curriculum collapses into one generic task.
- **Add the minimum-reward hold.** It is recorded per trial (`min_reward_each_block`), but the
  other hold is not, so adding one is a partial fix. The measurement below decides whether either
  matters.

## Consequences

- Two holds of the real coupled task are not simulated: `BlockMinReward`, and `AdvancedBlockAuto`,
  which keeps a block until the animal's choices follow the better side (not recorded; the
  curricula turn it on in the early coupled stages). The manual next-block button is not recorded
  either. P0 (r1) measured how often a real coupled block had more engaged trials than its
  `block_max`, so was certainly held: 12.5% of measurable blocks, 17% in baited `STAGE_2`, 28%
  in the warm-up, none in the final stages; uncoupled sessions run the real rule already.
- So the headline rollout fingerprints use uncoupled and final-stage sessions. Early coupled
  stages are shown beside them, marked as limited by the simulator (decided with Han,
  2026-11-03).
- P2 runs a sensitivity check: the two holds as an optional gym rule, set as the curricula set
  them for those stages, and 2 or 3 models rolled out with the holds on and off. If the
  fingerprints move less than the noise floor, this ADR stands; if they move more, the holds are
  turned on where they can be inferred, by a new ADR.
