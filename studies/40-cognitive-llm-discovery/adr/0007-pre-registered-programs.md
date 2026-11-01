# Four pre-registered programs, chosen on the selection subjects

Four programs go to the test subjects, chosen by a rule fixed before any search runs: the best program
from the search, and the best program at each of the three simplification floors (90%, 75%, 50%). For
each of the four, the five top candidates on the validation subjects are rescored on the selection
subjects, and the best of them goes to the test subjects, where it is scored once. A candidate whose
replay or rollout fingerprints are worse than the best published model's is skipped for the next one
(ADR 0001).
Picking among thousands of programs on the validation subjects inflates the winner's score, so the
choice is made on subjects the search never saw, and the test subjects only measure (decided with Han,
2026-11-01).

## Consequences

- The headline reports four numbers, one per program, each with its fit-versus-complexity position.
- Nothing scored on the test subjects feeds back into the search, the simplification or the choice.
