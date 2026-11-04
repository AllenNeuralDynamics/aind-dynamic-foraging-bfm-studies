# 15-behavioral-fingerprints

Vocabulary used only in this study. Repo-wide terms are in the root
[CONTEXT.md](../../CONTEXT.md); program, rollout, replay fingerprint, rollout fingerprint and the
fitting regimes (per subject, 2-level EM, ...) are
[MEvolve's](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-mevolve/blob/main/CONTEXT.md)
and keep their meaning here.

## Language

### Statistics

**Fingerprint statistic**:
A model-free choice statistic defined by a key, computed from the history before a trial, and an
outcome on that trial, such as p(switch) by the last three trials, or the count of runs of each
length.
_Avoid_: summary statistic, aux stat, behavioural metric

**Bin**:
One value of a fingerprint statistic's key, such as the history `S+ O− S−` (same side rewarded,
other side unrewarded, same side unrewarded, most recent first).
_Avoid_: pattern row, condition

**Hazard**:
p(switch) by last reward and current run length: the chance that a run ends on the next trial.
_Avoid_: switch-triggered curve (that name is the wrapper's post-switch statistic)

**Fingerprint spec**:
The versioned list of fingerprint statistics, keys and caps (`fingerprints-v1`) that every output
names.

**Tally**:
For one bin, subject and session partition: the trial count, the summed predicted outcome and the
summed observed outcome.

### Sessions

**Warm-up session**:
A session in which, on every trial, one side's reward probability is 0 (or the probabilities are
only 0 and 1): one side always pays, so there is nothing to forage.
_Avoid_: training session, shaping session

### Rollouts

**Parameter-matched rollout**:
A rollout whose task is built from its real session's recorded task parameters, with new blocks
drawn by the task's own rules.
_Avoid_: curriculum-matched rollout (the wrapper's, with default parameters), session-matched
rollout (a copied schedule)

### Comparison

**Gap**:
The trial-weighted mean of |model − subject| over a statistic's bins for one subject, then averaged
over subjects so each counts once.
_Avoid_: error, distance, match

**Noise floor**:
The gap a perfect model would still have, because the subject has finite trials; the binomial
floor in replay, the split-half floor in rollout.
_Avoid_: ceiling, chance level

**Binomial floor**:
The replay noise floor: per bin, a perfect model's expected miss from coin-flip noise on the
subject's own trials.

**Split-half floor**:
The rollout noise floor, a test–retest reference: the gap between two halves of a subject's own
sessions, scaled to the full data, so how far the subject is from itself.

**Coverage**:
The share of a subject's kept-bin weight that a rollout visits; bins it never visits are left
out of its gap.

**Floor ratio**:
A gap divided by its noise floor; 1 means as good as the subject's data allows.

**Fingerprint score**:
One number per model and mode: the geometric mean of its floor ratios over that mode's
statistics; 1 means as good as the data allow on every statistic. Replay and rollout each have
their own.
_Avoid_: behavioural score, combined score
