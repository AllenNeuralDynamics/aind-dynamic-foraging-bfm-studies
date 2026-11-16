# Study 17 — cross-validated GRU predictability

The scientific vocabulary of study 17, which gives every mouse a held-out GRU prediction. The
repository-wide vocabulary is the root `CONTEXT.md`; study 40's cohort terms are in
`studies/40-cognitive-llm-discovery/CONTEXT.md`.

## Language

### Mice and folds

**Fold**:
A set of mice held out from one GRU training. The 763 mice form 5 folds; every mouse is in exactly
one.
_Avoid_: split, held-out set

**Test fold**:
The 149 test subjects of the data freeze (`pool == heldout`), the same mice as study 40's test
subjects.
_Avoid_: test mice, GRU held-out list

**Pool fold**:
One of the 4 parts of the 614 training-pool mice (`pool == train`).

**Fold model**:
The 3-way GRU trained on every mouse outside one fold, test-fold mice included.

### Sessions and passes

**Fine-tune sessions**:
The sessions of a held-out mouse on which only its subject embedding is fitted; the freeze's
`fit` sessions.
_Avoid_: context sessions, train sessions

**Scored sessions**:
The sessions of a held-out mouse that a fold model predicts after fine-tuning; the freeze's
`score` sessions.
_Avoid_: eval sessions, test sessions

**Pass**:
One fine-tune-and-score of a held-out mouse. Pass A fine-tunes on the fit sessions and scores the
score sessions; pass B swaps them, so every session is scored once.

### Results

**Predictability map**:
The held-out per-trial predictions, with per-session and per-mouse scores, for all 763 mice.
_Avoid_: ceiling map, GRU map
