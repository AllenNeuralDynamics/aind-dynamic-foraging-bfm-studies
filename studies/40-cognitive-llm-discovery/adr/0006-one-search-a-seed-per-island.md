# One search, a different seed on each island

The pilot runs one search, not one per starting point: each island starts from a different seed
program (a near-empty template, and literature models from the seed library chosen by a rule fixed
in advance), and the whole library is shown to the LLM as inspirations. The study asks for the best
interpretable model, not for a comparison of starting points, and CogFunSearch already showed that
seeding from strong models from the literature helps. Islands let each seed's lineage develop before
migration lets mechanisms from different models combine (decided with Han, 2026-11-01).

**Seeds.** The seed library is the literature models study 12 ranks, each rewritten as a program and
checked against its original; nothing else enters the prompt pool. Study 12 fits 20 models and ranks
19: Win-Stay-Lose-Shift is fitted but not ranked, because its preset fixes the threshold and, with no
lapse term, scores near zero, so it is neither a seed nor an inspiration (updated 2026-11-01, when
study 12 added the free LossCounting model). There
are five islands: a near-empty template; the three literature models with the best scores on the
validation subjects (from study 12, which fits them on every cohort subject: its session-level scores
if they exist when the pilot starts, otherwise its per-subject ones); and
Eckstein's Bayesian-inference model as the library's model with discrete states, or the fourth-ranked
model if Eckstein's is already in the top three. That spreads the $450 at roughly $90 per island
(decided with Han, 2026-11-01).

ShinkaEvolve starts every island from one program, but its program database accepts scored programs
placed on a chosen island, so MEvolve's adapter places the seeds before the search starts (to be
confirmed at the engine smoke gate).

## Considered options

- **CogFunSearch's published programs as extra inspirations**: they come from other species and tasks,
  carry the problems the paper lists, and have no reference to check them against.
- **A GLM-HMM-style seed with discrete states**: deferred to a variant with three choices (left, right,
  ignore), where engagement states matter most.
- **One island per literature model**: too little budget per island for any lineage to develop.
- **Separate searches per starting point** (`base` from a template, `seeded-ctt` from
  CompareToThreshold, $225 each): comparable search curves, but half the budget each and no crossing
  between the models' mechanisms.

## Consequences

- There is no clean "from scratch" result: after the first migration the template's island has seen
  the literature models. A from-scratch search, with no library in the prompt, can be a later variant.
- Each program's genealogy still records which seeds it descends from.
