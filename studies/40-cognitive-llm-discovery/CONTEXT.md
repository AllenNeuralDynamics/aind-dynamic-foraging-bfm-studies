# Study 40 — LLM-driven discovery of cognitive programs

The vocabulary only this study needs. Hierarchy and evaluation terms (subject, session, held-out
subject, context session, condition) are the
[dispatcher glossary](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/blob/main/CONTEXT.md)'s;
program and search terms (program, parameter, latent variable, state, the fitting regimes (per subject,
2-level HB, 2-level EM, ...), 2-level model, session marginal, rollout, fingerprint, proposal, score) are
[MEvolve's](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-mevolve/blob/main/CONTEXT.md).
Both keep their meaning here.

## Language

### Cohorts

**Generation subjects**:
Training-pool subjects whose diagnostics go into the feedback the LLM sees.
_Avoid_: training mice

**Validation subjects**:
Training-pool subjects, disjoint from the generation subjects, whose pooled likelihood is a
program's score; the LLM sees only that one number per program.

**Selection subjects**:
Training-pool subjects, disjoint from the other cohorts, used once at the end to choose the
pre-registered programs.
_Avoid_: second validation set, holdout set

**Monitor subjects**:
Training-pool subjects, disjoint from the other cohorts, on which programs already found are
rescored to see whether their validation gains carry to subjects the search never used. Nothing
scored on them feeds back into the search or the choice of the pre-registered programs.
_Avoid_: holdout set, selection subjects

**Test subjects**:
The 149 held-out subjects of the per-subject MLE and HB baselines, scored once per variant at the
end.
_Avoid_: test mice, evaluation set

**Shared test subjects**:
The 146 test subjects that are also on the GRU's held-out list; the headline is pooled over them
for every model.

### Models

**Literature model**:
A cognitive model from the literature, as implemented in `aind-dynamic-foraging-models` and fitted by
study 12, which defines the term; the ones study 12 ranks form the seed library.
_Avoid_: published model, author model, hand-crafted model, baseline model

### Results

**Gap closed**:
On the shared test subjects, the program's likelihood minus HB-CTT's, divided by the GRU's minus
HB-CTT's.
_Avoid_: improvement

**Stage group**:
A session's training stage collapsed to final (final or graduated), earlier, or unknown; every
score is split by stage group and task variant.

**Both-regime test**:
Scoring an elite with and without one of its mechanisms, once with one parameter set per subject and
once as 2-level EM.

**Flagged mechanism**:
A program edit whose gain appears with one parameter set per subject but vanishes as 2-level EM,
a sign that it absorbs drift between sessions rather than describing computation within a session.

**Fingerprint gate**:
The rule that skips a pre-registration candidate whose replay gap or rollout gap on the selection
subjects exceeds that of the best literature model.

**Fit floor**:
The validation score a simplified program must keep, as a fraction of the distance from the best
literature model to the best discovered program, both scored as 2-level EM.

**Pre-registered programs**:
The programs chosen on the selection subjects, by a rule fixed in advance, before the test subjects
are scored.
