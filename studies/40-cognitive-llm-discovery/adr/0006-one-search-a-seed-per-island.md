# One search, a different seed on each island

The pilot runs one search, not one per starting point: each island starts from a different seed
program (a near-empty template, and published models from the seed library chosen by a rule fixed
in advance), and the whole library is shown to the LLM as inspirations. The study asks for the best
interpretable model, not for a comparison of starting points, and CogFunSearch already showed that
seeding from strong published models helps. Islands let each seed's lineage develop before
migration lets mechanisms from different models combine (decided with Han, 2026-11-01).

ShinkaEvolve starts every island from one program, but its program database accepts scored programs
placed on a chosen island, so MEvolve's adapter places the seeds before the search starts (to be
confirmed at the engine smoke gate).

## Considered options

- **Separate searches per starting point** (`base` from a template, `seeded-ctt` from
  CompareToThreshold, $225 each): comparable search curves, but half the budget each and no crossing
  between the models' mechanisms.

## Consequences

- There is no clean "from scratch" result: after the first migration the template's island has seen
  the published models. A from-scratch search, with no library in the prompt, can be a later variant.
- Each program's genealogy still records which seeds it descends from.
