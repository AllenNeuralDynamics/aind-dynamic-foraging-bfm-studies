# Study 13 — two-level HB on held-out subjects

The vocabulary only this study needs. Hierarchy and evaluation terms (subject, session-level
parameter, subject-level hyperparameter, held-out subject, held-out session, context session,
condition, partial pooling) are the
[dispatcher glossary](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/blob/main/CONTEXT.md)'s
and keep that meaning here.

## Language

### The model

**Two-level HB**:
A hierarchical Bayesian forager with subject-level hyperparameters over session-level
parameters and no population level, fitted to one subject at a time. It still partially pools
across that subject's sessions.
_Avoid_: subject-only HB, no-pooling HB, Sue's model

**Fixed hyperprior**:
The prior on the subject-level hyperparameters, chosen before seeing data and identical for
every subject. It stands where the population level would be.
_Avoid_: population prior, empirical prior

**Condition** (as used here):
The dispatcher's term, with the fixed hyperprior in place of the population posterior: infer a
held-out subject's subject-level hyperparameters from its context sessions alone.

### Scoring

**Session-marginal likelihood**:
The probability of a whole held-out session, averaged over draws of its session-level
parameters from the conditioned subject. The headline score.
_Avoid_: predictive likelihood, marginal likelihood (that is model evidence)

**Trial-marginal likelihood**:
The product over trials of each trial's choice probability averaged over those same draws;
study 08's score, kept for continuity.
_Avoid_: matched likelihood (08's metric name, which names the split, not the method)

**Trial-weighted pooled likelihood**:
The geometric-mean choice probability over every scored trial of every held-out subject:
`exp(total log likelihood / total trials)`. How either score above is reported, matching the
GRU and per-subject MLE baselines.
_Avoid_: mean likelihood (a per-subject average is a different number)
