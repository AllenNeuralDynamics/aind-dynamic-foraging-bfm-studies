# Variant `gate4-noseed`

**What.** The control arm of gate 4's second half (issue #110; Han, 2026-11-04): the same search as
[`gate4-recovery`](../gate4-recovery/notes.md), with **no seed programs**. Claude starts from the
template alone (side bias and stickiness). Everything else is the same: the same simulated mice
(setting a, `gate4/ttck_a_1100041/` on scratch), target, prompt with the library off, evaluator, 100
proposals, $8 cap, effort medium, one repair attempt per edit, and no embeddings. With no seeds,
MEvolve makes one island instead of four, so the arms also differ in island structure.

| setting | value |
|---|---|
| target (generating program) | `generator/two_timescale_ck.py`, byte-identical to `gate4-recovery`'s: 7 parameters, 5 latent variables, hand-set setting a |
| simulated mice | `gate4-recovery`'s: 24 generation, 24 validation and 24 held-out mice |
| seeds | none: the template on the one island |
| library in the prompt | off |
| search | 100 proposals by Claude Sonnet 5.5 at effort medium, one repair attempt per edit, a $8 cap |
| evaluator | 2-level EM, at most 20 EM iterations, 2,048 draws |
| code | MEvolve 59adafd (AllenNeuralDynamics/aind-dynamic-foraging-mevolve#51), ShinkaEvolve pinned at 8adc053a |

**Why.** `gate4-recovery` passed at proposal 3, from the island seeded with the two-timescale program,
which lacked only the choice kernel; its other islands reached only one-timescale Q-learning plus a
kernel by proposal 4. This arm asks whether Claude invents the slow reward trace from the template
alone, and what the seeds were worth.

**Pre-registered (2026-11-04, before launch; `gate4-recovery`'s result was known).**

1. **Pass:** as in `gate4-recovery`: the best validation score is at least the target − 0.2,
   72.928 − 0.2 = 72.728 on the ×100 scale.
2. **Compared with `gate4-recovery`:** best validation score against proposals and dollars; the
   proposal that first reaches the pass mark, in each arm.
3. **Mechanism recovery**, reported: whether the best program has a slow reward trace and a choice
   kernel (its code and `LATENT_NAMES`).
4. **Winner's curse:** the winner and the target, each scored once on the 24 held-out mice
   (`heldout.py`).

**What it can show.** One run per arm, so a difference in proposals to the pass mark is one sample,
not an estimate; a failure here with a pass there says the seeds mattered for this mechanism in 100
proposals.

**Launches.** Each launch writes its record to `launch_record/` and mints one W&B group `gate4-noseed@<launch_id>` in `NeuralDynamics/dfbfm-mevolve` (docs/adr/0001). Interventions (rescue, resubmit, extend, ...) join that group.

**Status.** Not launched.
