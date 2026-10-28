# aind-dynamic-foraging-bfm-studies

The record of every scientific question asked of dynamic-foraging-bfm: what was launched, what
it produced, and what was concluded. This glossary fixes the provenance vocabulary
([Language](#language)), the names of the numbers reports print ([Metrics](#metrics)), and
plain explanations of the methods reports use ([Methods](#methods)). Model and evaluation terms
(held-out subject, held-out session, context session, few-shot, condition, partial pooling,
...) are defined in the
[dispatcher glossary](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/blob/main/CONTEXT.md).
A study may keep its own `CONTEXT.md` for the vocabulary only that study uses, and its own
`adr/` for the decisions that shaped it; this file and `docs/adr/` hold what is repo-wide.

## Language

### Studies and their structure

**Study**:
One scientific question, with everything launched, produced and concluded to answer it. Its
accession number is permanent and never reused.
_Avoid_: experiment, project

**Variant**:
A self-contained configuration of a study's experiment, with its own sweep and experiment
manifests. A study compares its variants.
_Avoid_: condition (the dispatcher glossary uses that for Bayesian inference), arm, setting

**Study status**:
Where a study stands: `planned`, `active`, `complete`, or `superseded` (another study now
answers its question).

### Launches

**Launch**:
One initial submission of a variant's work to a compute platform. Every launch mints exactly
one W&B group.
_Avoid_: run (a run is one W&B run inside a launch), job, sweep

**W&B group**:
The identity of a launch's runs in W&B, named `<variant>@<launch_id>`. One launch, one group.
_Avoid_: sweep (a W&B sweep is a scheduling handle, not the group)

**Intervention**:
A later action on an existing launch that keeps its W&B group: a resubmit, rescue, probe,
tier change, backfill, or extend. It reuses the launch's manifests, differing only in declared
deviations; any other change to scientific parameters is a new launch.
_Avoid_: retry, relaunch

**Launch record**:
The immutable file recording one launch or intervention: what was submitted, from which
commits, on which data snapshot and in which image, and why.

### Results

**Report**:
A written answer to one sub-question of a study, regenerated from committed inputs. Its
numbers live in result blocks.
_Avoid_: writeup, notebook, analysis

**Report status**:
`draft`, `live`, `superseded` (another report replaces it entirely), or `retracted` (its
conclusion is wrong and nothing replaces it).

**Correction**:
A later report's amendment to part of an earlier report's conclusion, declared by the
correcting report. The corrected report stays `live` and is never edited to record it.
_Avoid_: update, erratum

**Result block**:
The delimited region of a report holding the numbers its producer regenerates.

**Upstream**:
Another study's report or launch whose results a report uses as input or comparison. Study
dependencies are the union of their reports' upstreams; a mere link is a citation.
_Avoid_: depends_on, prerequisite

**Citation**:
A link in a report or README to another study or report. It implies no use and is not a
relationship.

**Motivation**:
The earlier result a study exists to follow up, with one line on what that result showed. It
records why the work was started, not what it uses (that is upstream).
_Avoid_: decision, prompted-by, parent

**Tag**:
A value from a fixed vocabulary that says which model a study or report trains or fits, and what
kind of question it asks. A study's tags say what it asks; a report's say what it does. A model
whose numbers are only read from another study is not a tag.
_Avoid_: label, category, keyword

**Lineage**:
The chain from a result block back through its producer, frozen inputs, W&B groups and
launches to the data it came from.
_Avoid_: trace, thread, provenance chain

**Producer**:
The committed script that computes an analysis artifact from frozen inputs.
_Avoid_: notebook, analysis script

**Analysis artifact**:
A committed file a producer writes (JSON, CSV, or figure). The producer's JSON artifact
records the provenance of every file it wrote.
_Avoid_: output, result

**Freeze**:
A producer that reads a mutable external source (W&B, a Beaker dataset, a URL) and writes a
committed frozen input, recording which runs or objects, at which step or version, it read.
Every other producer reads only committed files.
_Avoid_: pull, snapshot, export

**Frozen input**:
A committed file written by a freeze, from which reports are regenerated offline.
_Avoid_: cache (a cache is uncommitted and never an input)

**Input digest**:
The SHA-256 of a file a producer read, recorded in the artifact so a number traces to exact
bytes rather than to a commit.

**Provenance baseline**:
The committed list of known provenance findings in existing files. It may only shrink.

## Metrics

Every number a report prints carries its full name from this section, with its aggregation, in
tables, claims and JSON keys. The words after the comma say how values were combined:

- **trial-pooled**: all scored trials go into one sum before the mean, so a subject with more
  trials counts more;
- **per-subject mean** or **per-subject median**: one value per subject, then the mean or the
  median over subjects;
- **subject-balanced**: one mean log value per subject, then the mean over subjects, then back
  to a probability.

If a value also averages seeds, add `seed mean` and n: "held-out likelihood, trial-pooled, seed
mean (n = 3 seeds)". JSON keys use the same words in snake_case
(`heldout_likelihood_trial_pooled`, `heldout_likelihood_per_subject_median`). Link an entry by
its id, which stays fixed when the wording changes:
`[held-out likelihood, trial-pooled](../../../../CONTEXT.md#metric-heldout-likelihood-trial-pooled)`.
`_Avoid_` lists the names earlier reports used for the same quantity, so an old report can be
read against this list. `_Code_` names one producer that computes the metric. A metric that only
one study uses belongs in that study's `CONTEXT.md`; the ones under
[Used by one study](#used-by-one-study) stay here until that study defines them.

### Likelihood of held-out choices

<a id="metric-normalized-likelihood"></a>**Normalized likelihood**:
The geometric mean, over a set of trials, of the probability the model gave to the choice the
animal made: `exp(Σ log p / number of trials)`. It is a probability per trial from 0 to 1;
chance is 0.5 for a left/right choice and 1/3 for left/right/ignore; higher is better. Each
likelihood entry below is a normalized likelihood and names its trials and its aggregation.
_Avoid_: LL, log-likelihood, NLL, nats (it is a probability, not a log)
_Code_: `rnn_utils.normalized_likelihood` in aind-disentangled-rnns, at the wrapper's pin

<a id="metric-heldout-likelihood-trial-pooled"></a>**Held-out likelihood, trial-pooled**:
The normalized likelihood over every scored trial of every held-out subject:
`exp(Σ log p over all trials / total trials)`. A GRU or disRNN run logs it in W&B under keys
such as `heldout/eval_likelihood` and `heldout/final/eval_likelihood`; where each run fits one
subject (study 13), the study's producer pools the runs. GRU and disRNN curves usually give its
seed mean.
_Avoid_: normalized likelihood, held-out LL, pooled LL, pooled likelihood, pooled held-out
likelihood, trial-weighted pooled likelihood, held-out likelihood (trial-weighted), pooled LL
(trial-wtd), cell-level likelihood, held-out choice likelihood per trial
_Code_: `studies/13-hb-subject-vs-gru-heldout/analysis/r1_hattori_pooled.py`

<a id="metric-heldout-likelihood-per-subject"></a>**Held-out likelihood, per subject**:
The normalized likelihood over the scored trials of one held-out subject: one value per subject
and run. It is the input to the per-subject summaries and to the paired differences below.
_Avoid_: per-mouse likelihood, per-mouse LL
_Code_: wrapper `code/evaluation/baseline_rl_evaluation.py`
(`_optional_normalized_likelihood_from_log_stats`)

<a id="metric-heldout-likelihood-per-subject-mean"></a>**Held-out likelihood, per-subject mean**:
The arithmetic mean of the per-subject held-out likelihood over the held-out subjects; every
subject counts the same. In this data it differs from the trial-pooled value by more than many
differences between models, so never compare numbers across the two.
_Avoid_: per-mouse mean LL, mean held-out likelihood, mean likelihood, GRU mean, held-out
likelihood (with no aggregation)
_Code_: `studies/01-gru-scaling-law/analysis/rl_baseline.py` (`per_subject_mean_likelihood`)

<a id="metric-heldout-likelihood-per-subject-median"></a>**Held-out likelihood, per-subject median**:
The median of the per-subject held-out likelihood over the held-out subjects.
_Avoid_: per-subject medians, held-out likelihood (with no aggregation)
_Code_: `studies/13-hb-subject-vs-gru-heldout/analysis/r2_per_subject.py` (`per_arm_median`)

<a id="metric-heldout-likelihood-subject-balanced"></a>**Held-out likelihood, subject-balanced**:
`exp(mean over subjects of each subject's mean log p per trial)`: the geometric mean of the
per-subject values, so every subject counts the same. It equals the trial-pooled value only when
every subject has the same number of trials, and it is never above the per-subject mean.
_Avoid_: normalized likelihood, R1-scale likelihood, "Bari2019 likelihood" or "GRU614
likelihood" with no aggregation
_Code_: `studies/09-gru-cross-species-transfer/analysis/freeze_generalization_drivers.py`

### Differences between models

<a id="metric-heldout-likelihood-difference-trial-pooled"></a>**Held-out likelihood difference, trial-pooled**:
Model A's trial-pooled held-out likelihood minus model B's, in probability units. Write it
"A − B"; positive favours A. The same difference of subject-balanced values is the held-out
likelihood difference, subject-balanced (study 09 r3, r6).
_Avoid_: gap, Δ, step, residual gap, remaining gap, trial-pooled Δ, "GRU minus HB" with no
aggregation, nats
_Code_: `studies/13-hb-subject-vs-gru-heldout/analysis/r4_ladder.py`

<a id="metric-heldout-likelihood-difference-per-subject"></a>**Held-out likelihood difference, per subject**:
For each subject that both models scored, A's per-subject held-out likelihood minus B's. Report
its per-subject median or per-subject mean, with the
[share of subjects where A is higher](#metric-share-of-subjects-higher); positive favours A. If
a model has several seeds, say whether the seeds were averaged as likelihoods or as
log-likelihoods before the difference: the two orders give different numbers.
_Avoid_: median Δ, mean Δ, subject median Δ, subject mean Δ, median Δ likelihood, HB − arm,
step (per subject), E8 minus E4
_Code_: `studies/13-hb-subject-vs-gru-heldout/analysis/r2_per_subject.py` (`hb_minus`)

<a id="metric-share-of-subjects-higher"></a>**Share of subjects where A is higher**:
The fraction of paired subjects whose per-subject difference A − B is above zero; a tie counts
as not higher. Give the count beside it, for example 145/146.
_Avoid_: % mice improved, subjects GRU better, subjects improved, subjects with a gain, %↑,
frac_positive
_Code_: `studies/13-hb-subject-vs-gru-heldout/analysis/r2_per_subject.py` (`frac_hb_higher`)

<a id="metric-share-of-gain"></a>**Share of gain reached**:
For one metric along one axis, (value at X − value at the start) ÷ (value at the end − value at
the start): for example, the share of the D = 10 → 614 gain reached at D = 100, or the share of
the disRNN-to-GRU gap that a stage closes. It can be negative or above 1.
_Avoid_: frac of D-gain, % of baseline gap closed, gap closed
_Code_: `studies/11-disrnn-near-gru-ceiling/analysis/attribution_report.py`

### Bits per trial

<a id="metric-heldout-bits-per-trial"></a>**Held-out bits per trial, subject-balanced**:
The base-2 log of the subject-balanced held-out likelihood: the mean over subjects of each
subject's mean `log2 p` per trial. It is at most 0; chance is −1 for a left/right choice; higher
is better.
_Avoid_: bits/trial, test log2 likelihood
_Code_: `studies/09-gru-cross-species-transfer/analysis/report_adaptation_budget.py`

<a id="metric-heldout-bits-per-trial-difference"></a>**Held-out bits-per-trial difference, subject-balanced**:
A's held-out bits per trial minus B's: the `log2` of the ratio of their subject-balanced
held-out likelihoods. Positive favours A; +0.01 bits per trial means that A gives the observed
choices about 0.7% more probability per trial.
_Avoid_: bits/trial, GRU614−Bari2019 bits/trial, GRU advantage, GRU − Q
_Code_: `studies/09-gru-cross-species-transfer/analysis/freeze_generalization_drivers.py`

<a id="metric-heldout-bits-above-chance"></a>**Held-out bits above chance, subject-balanced**:
Held-out bits per trial plus 1: 0 at chance and 1 for a perfect left/right prediction.
_Avoid_: predictability, bits above chance (with no aggregation)
_Code_: `studies/09-gru-cross-species-transfer/analysis/freeze_generalization_drivers.py`

### Likelihood on other data

<a id="metric-heldout-session-likelihood-training-subjects"></a>**Held-out-session likelihood, training subjects**:
The trial-pooled normalized likelihood on the held-out sessions of the subjects the model trained
on. W&B logs it as `checkpoint/eval_likelihood`. It is not the likelihood of the training
sessions (`checkpoint/train_likelihood`). Study 04's "held-out likelihood" is this quantity, on
synthetic subjects.
_Avoid_: in-sample likelihood, within-subject likelihood, train LL, eval-LL, eval likelihood
_Code_: `studies/06-disrnn-operating-point-at-scale/analysis/model_selection.py`

<a id="metric-generalization-gap"></a>**Generalization gap**:
The held-out-session likelihood of the training subjects minus the trial-pooled held-out
likelihood of the held-out subjects, for one run. Positive means the model predicts the
subjects it trained on better than new subjects. Study 07's JSON `gap` is a
different quantity: training sessions minus held-out sessions of the training subjects.
_Avoid_: gap (in − held-out), train − held-out, residual (study 06 r2; opposite sign)
_Code_: `studies/11-disrnn-near-gru-ceiling/analysis/attribution_report.py`

<a id="metric-relative-likelihood"></a>**Relative likelihood**:
On synthetic data, a model's trial-pooled normalized likelihood divided by that of the agent
that generated the data, on the same sessions. 1 means the model predicts as well as the
generator; it can exceed 1 by chance.
_Avoid_: rel-LL, likelihood relative to ground truth
_Code_: wrapper `likelihood_relative_to_groundtruth`, read by
`studies/04-gru-vs-disrnn-embedding-recovery/analysis/recovery_report.py`

<a id="metric-lr-likelihood-engaged"></a>**Left/right likelihood, engaged trials**:
For a model that predicts left, right or ignore: the normalized likelihood of the side chosen,
over engaged trials only, with the model's probabilities rescaled over left and right. Chance is
0.5, so it compares with a left/right model's held-out likelihood. Name the aggregation as for
held-out likelihood.
_Avoid_: L/R-engaged likelihood, 3-way choice likelihood, conditional L/R likelihood
_Code_: wrapper `code/model_trainers/gru_trainer.py` (`eval_likelihood_LR_engaged`)

<a id="metric-ignore-pr-auc"></a>**Ignore-detection PR-AUC**:
The area under the precision–recall curve for finding ignored trials, ranked by the model's
probability of ignore. From 0 to 1; a model with no skill scores the ignore base rate (the share
of ignored trials), not 0.5.
_Avoid_: PR-AUC, average precision
_Code_: wrapper `code/model_trainers/gru_trainer.py` (`engage_ignore_pr_auc`)

<a id="metric-ignore-recall"></a>**Ignore recall**:
The fraction of ignored trials that the model calls ignored, when it calls a trial ignored at a
probability of ignore of 0.5 or more.
_Avoid_: recall
_Code_: wrapper `code/model_trainers/gru_trainer.py` (`engage_ignore_recall`)

### Generative match

<a id="metric-generative-match-correlation"></a>**Generative-match correlation, subject-mean**:
The Pearson correlation, across task conditions, between the subject-averaged behaviour curve
of the animals and that of the model run as a generator; it is not a mean of per-subject
correlations. Reports use a switch-triggered curve (behaviour after a switch, by reward and run
length) and a history-dependent curve (switch probability after each of 32 recent
choice-and-reward histories). From −1 to 1; higher is better.
_Avoid_: corr, subject-mean correlation, switch corr, history corr, r (both)
_Code_: wrapper `code/post_training_analysis/generative_analysis.py`, read by
`studies/01-gru-scaling-law/analysis/generative_match.py`

<a id="metric-generative-match-rmse-subject-balanced"></a>**Generative-match RMSE, subject-balanced**:
For each subject, the mean over conditions of the difference between model and animal
probability; then the root mean square of those subject means. Probability units, 0 or more;
lower is better. Errors of opposite sign in different conditions cancel inside a subject, so it
measures each subject's average bias, not the shape of the curve.
_Avoid_: RMSE, switch RMSE, history RMSE, sqrt(subject-balanced MSE)
_Code_: wrapper `code/post_training_analysis/generative_analysis.py`
(`subject_balanced_error_summary`), read by
`studies/01-gru-scaling-law/analysis/freeze_generative.py`

<a id="metric-generative-match-rmse-across-patterns"></a>**Generative-match RMSE, across patterns**:
The root mean square difference between the subject-averaged animal and model curves over the 32
history patterns. Unlike the subject-balanced RMSE, it sees the shape of the curve.
_Avoid_: RMSE, RMSE across the 32 pattern rows
_Code_: `studies/01-gru-scaling-law/analysis/pswitch_history_patterns.py`

### Model internals and decoding

<a id="metric-bottleneck-openness"></a>**Bottleneck openness**:
For one family of disRNN bottleneck channels, the sum over its channels of `clip(1 − σ, 0, 1)`,
where σ is the channel's learned noise scale. 0 means every channel is closed and passes no
information; the number of channels means every channel is fully open. Lower is sparser. It has
no unit (it is not in nats).
_Avoid_: total openness, Σ(1−σ), interaction openness, open channels
_Code_: `studies/03-disrnn-beta-scan/analysis/backfill_sparsity_metrics.py`

<a id="metric-decoding-r2"></a>**Decoding R², cross-validated**:
How well a linear decoder predicts a known quantity (a generating parameter, a mixture weight,
session position) from a model's embedding, scored on held-out folds: at most 1, 0 is no better
than the mean, and it can be negative. Reports differ in decoder (ridge or least squares), fold
unit (subjects or sessions) and target (the value or its rank), so name all three; rank R² is
not Spearman ρ². Fitted forager parameters compared directly with the true ones, with no
decoder, give parameter-recovery R² instead.
_Avoid_: recovery, recovery R², Ridge R², mix-R², position-R², session-frac R²
_Code_: `studies/01-gru-scaling-law/analysis/embedding_param_decode.py`,
`studies/04-gru-vs-disrnn-embedding-recovery/analysis/recovery_scoring.py`

<a id="metric-decoding-accuracy"></a>**Decoding accuracy, cross-validated**:
The fraction of held-out-fold subjects (or sessions) whose class, such as the generating forager
family, a logistic-regression decoder predicts correctly from the embedding. Chance is 1 ÷ the
number of classes. Model-selection accuracy is different: the share of subjects whose
best-fitting forager family is the true one, with no decoder.
_Avoid_: family-decoding accuracy, preset accuracy
_Code_: wrapper `code/analysis/stage4b_recovery.py`

### Used by one study

<a id="metric-shrinkage-slope"></a>**Shrinkage slope** (study 08):
The [Theil–Sen slope](#method-theil-sen) of the HB's estimate of a subject's parameter on the
per-subject MLE of the same parameter. 1 means the HB keeps the differences between subjects;
below 1 means it pulls them toward the group.
_Avoid_: slope (1 = no shrinkage)
_Code_: `studies/08-hb-vs-gru-heldout/analysis/shrinkage.py`

<a id="metric-between-session-width"></a>**Between-session width** (study 13):
For one subject and parameter, the distance between its 10th- and 90th-percentile session on the
parameter's own scale, computed for each posterior draw and summarized by the posterior median.
It measures how far a parameter moves from session to session.
_Avoid_: session spread, 10–90% session width
_Code_: `studies/13-hb-subject-vs-gru-heldout/analysis/freeze_posteriors.py`

<a id="metric-embedding-distance"></a>**Embedding distance** (study 09):
The [Mahalanobis distance](#method-mahalanobis-distance) of a subject's embedding from the centre
of the source subjects' embeddings. Reports give it for a cohort's mean embedding (centroid
distance), as the median over a cohort's subjects (median subject distance), and as the share of
a cohort's subjects beyond the source subjects' own 95th percentile. It has no unit and does not
compare across embedding sizes.
_Avoid_: embedding displacement, mean embedding distance (say which)
_Code_: `studies/09-gru-cross-species-transfer/analysis/freeze_generalization_drivers.py`

<a id="metric-task-design-distance"></a>**Task-design distance** (study 09):
How far a cohort's task is from the AIND task. Four kinds appear: task-structure distance (the
share of four categorical design features that differ from the nearest AIND task), full-design
distance (the same over seven features), empirical schedule distance (between z-scored
reward-schedule statistics) and LLM task distance (from a language model's pairwise similarity
judgements). Each is 0 for an identical task.
_Avoid_: task distance (say which)
_Code_: `studies/09-gru-cross-species-transfer/analysis/freeze_task_design_features.py`

<a id="metric-effective-rank"></a>**Effective rank** (study 01):
The number of dimensions an embedding really uses: `exp(entropy of the shares of variance along
its principal axes)`. From 1 to the embedding size.
_Code_: `studies/01-gru-scaling-law/analysis/freeze_embedding_dimension.py`

## Methods

Plain explanations of the methods reports use: what each is, why this project uses it, and
what to watch for. A report explains a method in one sentence at first use and links its entry,
for example `[bootstrap CI](../../../../CONTEXT.md#method-bootstrap-ci)`. Hierarchy and
evaluation terms (partial pooling, condition, context session, zero-shot, few-shot) are in the
[dispatcher glossary](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/blob/main/CONTEXT.md).

### Uncertainty

Convention: give a 95% CI (a credible interval for a Bayesian fit) or the SEM, always named,
with n and what n counts, for example "± SEM, n = 3 seeds" or "95% CI, subject bootstrap,
n = 149 subjects". An SD describes spread, not uncertainty.

<a id="method-sd"></a>**Standard deviation (SD)**:
The spread of single values around their mean, with n − 1 in the denominator. Use it to show how
much seeds or subjects vary, not how precise a mean is. Watch: some older producers divide by n,
which for two or three seeds gives a clearly smaller value; "σ" also names other things here
(bottleneck noise, between-session spread), so do not use it for SD.

<a id="method-sem"></a>**Standard error of the mean (SEM)**:
SD ÷ √n: how far the mean would move if the experiment were repeated with new units of the same
kind. Watch: with n = 2 or 3 seeds the SEM is itself very uncertain, and it covers only the
variation of the unit that n counts.

<a id="method-ci"></a>**Confidence interval (CI)**:
A range made so that, over repeated experiments, 95% of such ranges contain the true value. Name
how it was made (bootstrap, t distribution) and over which unit.

<a id="method-credible-interval"></a>**Credible interval**:
For a Bayesian fit, the central range that holds the stated share of the posterior, for example
the 5–95% range. It describes uncertainty given the model and its prior, not variation across
repeated experiments.

<a id="method-monte-carlo-error"></a>**Monte Carlo error**:
The standard error of an estimate that averages random draws (posterior draws, simulated
sessions, shuffles). For independent draws it falls as 1 ÷ √(number of draws). Watch: it covers only the randomness of
the draws, not variation between subjects or seeds, so it is usually much smaller than either.

### Models and fitting

<a id="method-mle"></a>**Per-subject maximum-likelihood fit (MLE)**:
Fit a forager to one subject's context sessions by finding the single parameter set that makes
its choices most probable. It is the classical baseline in studies 01, 08, 09 and 13. Watch: it
gives no uncertainty, and with few trials a parameter can run to a bound, for example a very
large inverse temperature.

<a id="method-hb"></a>**Hierarchical Bayesian model (HB)**:
A forager whose session-level parameters come from a subject-level distribution (two levels,
study 13), which can itself come from a population distribution (three levels, study 08), all
fitted together, so that sessions with little data borrow strength from the others
([partial pooling](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/blob/main/CONTEXT.md)). Studies use it to ask whether a classical model whose parameters vary
between sessions can match the GRU. Watch: results depend on the prior and on whether the
sampler converged ([convergence gates](#method-convergence-gates)).

<a id="method-plug-in"></a>**Plug-in prediction**:
Predict with one parameter set, such as the posterior median, instead of averaging the
predictions of all posterior draws. Study 13 uses it to show how much the HB gains from averaging
over draws.

<a id="method-session-marginal"></a>**Session-marginal and trial-marginal scoring**:
Two ways to score an HB on a held-out session whose own parameters are unknown, defined in
[study 13's glossary](studies/13-hb-subject-vs-gru-heldout/CONTEXT.md). Session-marginal
(study 13) averages the likelihood of the whole session over parameter draws, which predicts
each trial from the session's earlier trials as a particle filter does; trial-marginal (study
08) averages each trial's choice probability over the draws, so it cannot learn within a
session. Watch: the two can rank models differently.

<a id="method-heldout-subject-scoring"></a>**Held-out-subject scoring**:
How a model is scored on a subject it never trained on. A GRU or disRNN keeps its trained
weights, fits only a new subject embedding on the subject's context sessions, and is scored on
the subject's other sessions; an HB is conditioned on the context sessions instead. Watch:
studies split context and scored sessions differently (alternate sessions, first and second
half, the first K sessions), so state the split.

<a id="method-disrnn-bottleneck"></a>**disRNN information bottleneck**:
In a disRNN, each latent variable and each input to its update and choice networks passes
through a channel that adds learned noise. A penalty weighted by β (and by a multiplier on the
update network's latent inputs) rewards noise, so the channels the model does not need close and
the model becomes sparse and easier to read. Watch: a larger β gives a sparser model and usually
a lower held-out likelihood; compare [openness](#metric-bottleneck-openness) only at the same β
and multiplier.

<a id="method-non-centred"></a>**Non-centred parameterization**:
Write a session's parameters as the subject mean plus the subject spread times a standard-normal
draw, instead of drawing them from the subject's distribution directly. It helps the sampler
when the spread is small or weakly set by the data (a "funnel"). Study 13's fits use it; a
centred refit had many more divergences.

### Bayesian sampling and its checks

<a id="method-mcmc"></a>**MCMC sampling**:
Draw a long sequence of parameter values whose distribution approaches the posterior. Several
chains start apart; each discards an initial warmup and keeps the rest as draws. Studies 08 and
13 fit their HBs this way. Watch: the draws describe the posterior only if the chains converged.

<a id="method-convergence-gates"></a>**Convergence gates (R-hat, bulk ESS, divergences)**:
Checks that MCMC draws can be trusted: R-hat compares the spread within and between chains
(gate: below 1.01), bulk ESS counts how many independent draws the correlated chains are worth
(gate: at least 400), and a divergence is a step where the sampler failed in a sharply curved
region (gate: none). Watch: passing does not prove convergence, and the subjects that fail are
rarely a random subset, so a converged-only result describes a different population.

<a id="method-importance-sampling"></a>**Importance sampling**:
Estimate an average under one distribution from draws taken from another (the proposal), and
weight each draw by the ratio of the two densities. Study 13 uses it to rescore sessions whose
plain draw-average rests on a few draws, with a proposal centred on the session's best-fitting
parameters and mixed with the original distribution so that no weight grows too large.

<a id="method-pareto-k-hat"></a>**Pareto k̂**:
A check on a weighted average: it fits a Pareto tail to the largest weights and reports the
tail's shape, k̂. Below 0.7 the estimate is reliable; above it a few draws dominate, and the
estimate is noisy and can be biased low. Study 13 gates every scored session on it. PSIS-LOO
uses the same check, but no study here does leave-one-out cross-validation.

### Tests and intervals

<a id="method-bootstrap-ci"></a>**Bootstrap confidence interval**:
Resample the units (subjects, cohorts, or subjects with all their trials) with replacement many
times, recompute the statistic each time, and take the 2.5th and 97.5th percentiles as the 95%
CI. Watch: name the resampled unit; the interval covers only variation across that unit (a
subject bootstrap ignores seed-to-seed variation), and with few units, such as 7–12 cohorts, it
is rough.

<a id="method-wilcoxon"></a>**Wilcoxon signed-rank test**:
A paired test of whether per-subject differences between two models are centred on zero. It uses
only the signs and ranks of the differences, so a few extreme subjects cannot drive it. Watch: with
about 150 subjects a tiny difference gives a very small p, so report the size of the difference
beside p; the test also assumes that subjects are independent.

<a id="method-paired-t-test"></a>**Paired t-test**:
A paired test of whether the mean difference is zero, assuming the differences are roughly
normal. Name the paired unit (subjects, or seed-and-D cells).

<a id="method-non-inferiority"></a>**Non-inferiority test**:
A test that a model is not worse than a reference by more than a margin fixed before the data are
seen. It passes only if the whole 95% CI of model − reference lies above −margin. Watch: "no
significant difference" is not non-inferiority. Study 10 uses it.

<a id="method-permutation-test"></a>**Permutation test**:
Shuffle the labels (or one variable) many times, recompute the statistic each time, and take p as
the share of shuffles at least as extreme as the real data. Watch: count the real data as one of
the shuffles, so the smallest possible p is 1 ÷ (shuffles + 1); 200 shuffles cannot show
p < 0.001.

<a id="method-bh-fdr"></a>**Benjamini–Hochberg false discovery rate (FDR)**:
A correction for testing many hypotheses together: it turns each p into a q that bounds the
expected share of false findings among the results called significant. Study 09 applies it to a
family of feature correlations.

<a id="method-correlation"></a>**Pearson and Spearman correlation**:
Pearson r measures how close two variables lie to a straight line. Spearman ρ is Pearson r on
ranks, so it measures any steadily rising or falling relation and resists outliers. Both run
from −1 to 1. Watch: n is the number of units correlated (subjects or cohorts); with 7–12
cohorts even a large ρ is uncertain.

<a id="method-leave-one-cohort-out"></a>**Leave-one-cohort-out**:
Repeat an analysis once per cohort, each time without that cohort, and report the range of the
results. It shows whether one cohort drives a correlation across cohorts.

<a id="method-intraclass-correlation"></a>**Intraclass correlation (ICC)**:
The share of the total variance that lies between groups (for example sessions) rather than
within them: 0 means the groups do not differ, 1 means all variation is between groups.

### Fits and decoders

<a id="method-scaling-law-fit"></a>**Scaling-law fit**:
Fit a saturating curve, such as `L(D) = L∞ − A·D^(−α)`, to held-out likelihood against the
number of training subjects D (or model size N), to estimate the ceiling L∞ and how fast the
curve approaches it. Watch: five values of D constrain three parameters weakly, so read L∞ as a
rough guide, do not extrapolate far, and check that the code fits the form the report prints.

<a id="method-aic"></a>**Akaike information criterion (AIC)**:
Twice a fit's negative log-likelihood plus twice its number of parameters; lower is better. For
a least-squares fit the first term becomes n·ln(RSS / n). Compare AIC only between models fitted
to the same data.

<a id="method-cross-validation"></a>**Cross-validation**:
Split the data into k folds, fit on k − 1 of them, score on the one left out, and rotate. Watch:
when one subject gives many rows, keep all its rows in one fold (grouped folds), or the score is
too high.

<a id="method-linear-decoder"></a>**Ridge and logistic-regression decoders**:
Linear maps from an embedding to a number (ridge regression: least squares with a penalty on
large weights) or to a class (logistic regression). Studies use them to ask what an embedding
encodes. Watch: a decoder that fails does not prove the information is absent, only that it is
not linearly readable.

<a id="method-theil-sen"></a>**Theil–Sen slope**:
The median of the slopes between all pairs of points: a line fit that a few extreme points
cannot pull.

<a id="method-mahalanobis-distance"></a>**Mahalanobis distance**:
The distance of a point from the centre of a reference cloud, in units of the cloud's spread in
each direction. A distance of 1 along a wide direction is farther in raw units than 1 along a
narrow one.

<a id="method-pca"></a>**Principal component analysis (PCA)**:
A rotation of many variables (for example embedding dimensions) so that the first axes carry the
most variance. Reports use it to draw embeddings in two dimensions. Watch: a 2-D picture can hide
differences in the other dimensions.

<a id="method-winsorization"></a>**Winsorization**:
Clip extreme values at a fixed bound before computing a statistic, so that a few runaway values
cannot dominate it. Study 04 clips fitted forager parameters at or near the top of the true range.
