---
schema_version: 1
id: r3
slug: rollout-grid
status: draft
authors:
- han
issue: 94
tags:
  model:
  - cognitive
  - gru
  question:
  - comparison
  - generative-run
wandb_groups: []
inputs:
  script: analysis/r3_rollout_grid.py
  data:
  - analysis/p2_task_specs.json
  - analysis/p2_gru_contexts.json
  - analysis/p2_rollouts.json
  - analysis/p2_rollout_gaps.json
  - analysis/p2_holds.json
  - analysis/r2_replay_grid.json
  - analysis/p1_glm.json
  - analysis/p1_gru.json
  figure: analysis/fig_r3_rollout_grid.svg
  freezes:
  - analysis/freeze_p2_tasks.py
  - analysis/freeze_p2_gru_contexts.py
  - analysis/freeze_p2_rollouts.py
  - analysis/freeze_p2_holds.py
  - analysis/freeze_p2_gaps.py
upstream:
- study: 12-cognitive-literature-leaderboard
  report: r1
- study: 12-cognitive-literature-leaderboard
  report: r3
- study: 01-gru-scaling-law
  report: r2
reproduce: make -C studies/15-behavioral-fingerprints r3
---

# r3 — P2: when the models play the task, do their sessions look like the mice's?

## Summary

When each model plays the task itself, in gym sessions built from the mice's own sessions, no
model's switching comes as close to a mouse as that mouse's own sessions come to each other. The
best are about 1.4 times that split-half floor (Figure 1). Three models tie: CompareToThreshold
with its per-subject parameters, the GRU, and Bari2019 as 2-level EM. Across all models, the
rollout ranking follows the replay ranking and the likelihood (Figure 2). Among the good models
it follows them only weakly, because the fit that wins replay often loses rollout. Fitting
2-level EM, so that each session draws its own parameters, improves replay for 16 of 18 models
but improves rollout for only 9. For the best models it bends the losses curve the wrong way:
after more consecutive losses, the mice switch more, but the 2-level models switch less
(Figure 3), and several also switch back too often right after a switch (Figure 4). The GRU
follows the mice's curves (Figures 4 and 5). Two limits of the simulator do not
explain these results. The early coupled stages, where the gym cannot hold a block, score like
the headline sessions (Figure 6). Where the real task records a minimum-reward hold, turning it
on moves the fingerprints by less than half the floor (Figure 7).

## Question

Replay (r2) asks how well a model predicts each trial from the mouse's real history. A model can
pass that test and still drift away from the mice when it makes its own history, because its
small errors then feed back into its next choices. So, when each model plays session-matched
tasks from scratch, how close are the switching statistics of its sessions to the mice's, as a
multiple of how much the mice's own sessions differ from each other?

## Approach

**Models.** The same models as r2: study 12's 19 ranked literature models, each with its
per-subject parameters and as 2-level EM (except Rescorla-Wagner, whose 2-level replay did not
reproduce), the per-subject GLM, and the GRU (three seeds, on the 122 test subjects with a gym
task that its checkpoint did not train on).

**Tasks** (ADR 0006). Each of a subject's score sessions becomes one gym task with the session's
recorded block parameters, baiting, reward probabilities and engaged-trial count. A model plays
each task 10 times. A 2-level model draws new session parameters for every rollout from its fitted
subject distribution (ADR 0005). A task needs recorded block parameters, so 373 of 2,138 score
sessions are left out, and with them every score session of 24 of the 149 test subjects: 125
subjects remain.

**Statistics and the floor.** The five statistics of fingerprint spec v1: r2's four
switch-probability statistics, and the distribution of run lengths (how many trials a mouse
stays on one side). For each subject and statistic, the gap is the trial-weighted mean |model −
mouse| over the subject's bins, as in r2. In rollout, a model cannot track the mouse's particular
sessions, only its typical one. So the floor is no longer the binomial noise of r2: it is the
split-half floor (ADR 0003), the gap between two random halves of the subject's own sessions,
scaled to the full data. It contains trial noise and real session-to-session change. The floor
ratio is the mean gap over subjects divided by the mean floor: 1 means the model's sessions are as
close to the mouse's as the mouse's own sessions are to each other. The rollout fingerprint score is
the geometric mean of the five ratios (ADR 0009). The headline uses uncoupled and final-stage
coupled sessions, where the gym's task rules are the real ones (ADR 0006).

## Results

### 1. No model plays like the mice yet; the best come within 1.4 times the floor

<!-- BEGIN result-1 -->
![rollout grid](../fig_r3_rollout_grid.svg)

**Figure 1.** **a**, floor ratio (rollout gap ÷ split-half floor, each averaged over the 125 test subjects with a gym task) per model, regime and statistic, on the headline sessions (set `headline`: uncoupled and final-stage coupled, ADR 0006). 1 means the model's sessions differ from the subject's no more than the subject's own sessions differ from each other. **score** is the rollout fingerprint score, the geometric mean of the five ratios (ADR 0009). The GRU's row is the mean over its three seeds, on the 122 subjects it holds out. ▾ the rollouts visit less than 95% of the subjects' bin weight. \* the 2-level fit failed (study 12 r3); n.r., not rolled out (its replay did not reproduce, r2). **b**, p(switch) by the last trial's reward and the run length: the subjects' mean and four models' rollouts, over the 122 subjects every model holds out that keep the bin (2-level EM where the model has it).

| model | per subject: score | worst statistic | its floor ratio | 2-level EM: score | worst statistic | its floor ratio |
|---|---|---|---|---|---|---|
| Lebedeva-PR | **1.5** | hazard | 1.7 | **1.6** | losses | 1.9 |
| Lopez-DoubleTrace | **1.5** | hazard | 1.6 | **1.9** | losses | 2.4 |
| Bari2019 | **1.6** | run lengths | 1.7 | **1.5** | last 1–3 trials | 1.6 |
| Miller-RHG | **1.7** | hazard | 1.9 | **1.8** | losses | 2.4 |
| CompareToThreshold | **1.4** | last 1–3 trials | 1.5 | **1.7** | losses | 2.1 |
| Hattori2019 | **1.6** | hazard | 1.7 | **1.7** | losses | 1.9 |
| Zid-HistoryKernel2-ForagingRL | **1.7** | losses | 1.8 | **2.0** | losses | 2.3 |
| Eckstein-BI | **1.8** | run lengths | 2.1 | **1.8** | last 1–3 trials | 2.1 |
| Beron-RFLR | **1.9** | run lengths | 2.1 | **1.7** | last 1–3 trials | 1.9 |
| Chen-RLCK | **1.8** | last 1–3 trials | 2.0 | **1.8** | last 1–3 trials | 2.1 |
| Eckstein-RL | **1.8** | run lengths | 2.2 | **1.8** | last 1–3 trials | 2.2 |
| Grossman-MetaLearning \* | **1.6** | run lengths | 1.8 | **2.4** | last 1–3 trials | 2.7 |
| FeedbackDependent-RL+Bias+CK1 | **2.8** | run lengths | 4.2 | **2.2** | run lengths | 2.7 |
| Alsio-DualRateSticky-RL | **3.0** | run lengths | 4.5 | **2.4** | run lengths | 3.0 |
| LossCounting \* | **4.1** | hazard | 5.6 | **4.2** | hazard | 5.8 |
| ActorCritic | **5.3** | last 1–3 trials | 6.7 | **4.2** | run lengths | 4.7 |
| FeedbackDependent-RL+Bias | **10.9** | last 1–3 trials | 19.6 | **6.9** | last 1–3 trials | 10.1 |
| FeedbackDependent-RL | **11.3** | last 1–3 trials | 20.9 | **8.2** | last 1–3 trials | 13.7 |
| Rescorla-Wagner \* | **11.7** | last 1–3 trials | 21.1 | — | — | — |
| GLM-Su2022 | **1.7** | last 1–3 trials | 1.8 | — | — | — |
| GRU-D614 | **1.4** | last 1–3 trials | 1.5 | — | — | — |
<!-- END result-1 -->

The good models fall short in every statistic, not just one: their worst ratio is about 1.5 to 2.
The poor models fail mainly on the losses and on the last trials. The single-learning-rate models
(Rescorla-Wagner and FeedbackDependent-RL) reach 11 to 21 there, as in replay.

### 2. Rollout broadly agrees with replay and likelihood, but not among the good models

<!-- BEGIN result-2 -->
![scores](../fig_r3_scores.svg)

**Figure 2.** Each model's rollout fingerprint score (Figure 1a) against **a**, its pooled held-out likelihood on the test subjects (r2 Figure 2c), and **b**, its replay fingerprint score (r2), per regime; the grey line marks equal scores. The models of Figure 1b are coloured.
<!-- END result-2 -->

Over all cells, the rollout score ranks the models much as the replay score does (Spearman
ρ = 0.79) and as the likelihood does (ρ = −0.79). But among the 28 cells with likelihood above
0.70, the agreement with likelihood is weak (ρ = −0.43). The clearest disagreement is the fitting
regime. Below the grey line in panel b, a model's rollout score is better than its replay score:
20 of 21 per-subject cells sit there, against 10 of 18 2-level cells. So a 2-level fit gains much
less from replay to rollout, and for the best models it loses its replay lead.

### 3. Drawing parameters per session bends the losses curve the wrong way

<!-- BEGIN result-7 -->
![regimes](../fig_r3_regimes.svg)

**Figure 3.** p(switch) by the count of consecutive unrewarded trials on the current side, in rollout: the subjects' mean (dots), three literature models with their per-subject parameters (dashed) and as 2-level EM, each session drawing its own parameters (solid), and the GRU, over the subjects every model holds out that keep the bin. Floor ratios of this statistic: CompareToThreshold per subject 1.4; CompareToThreshold 2-level EM 2.1; Lopez-DoubleTrace per subject 1.5; Lopez-DoubleTrace 2-level EM 2.4; Bari2019 per subject 1.6; Bari2019 2-level EM 1.6.
<!-- END result-7 -->

The mice switch a little more after each further loss, from 0.135 after one loss to 0.16 after
five or more. With per-subject parameters, the three models rise too (CompareToThreshold a little
above the mice, Bari2019 too steeply). As 2-level EM, CompareToThreshold falls instead, and
Lopez-DoubleTrace falls from 0.19 to 0.09; their floor ratios on this statistic grow from 1.4 to
2.1 and from 1.5 to 2.4. Bari2019's 2-level curve rises up to five losses, and its ratio does not
grow. One explanation, not yet tested: a long run
of losses is most likely in a rollout session that drew low switching parameters, so the long-loss
bins over-sample those sessions. The mice have session-to-session change too, but this result
suggests the fitted 2-level spread is larger than theirs, or of a different shape. The test is to
compare the spread of per-session switch rates between the rollouts and the mice.

<!-- BEGIN result-8 -->
![curves](../fig_r3_curves.svg)

**Figure 4.** The four switch statistics as curves, in rollout: the subjects' mean (dots), the five models of Figure 5 as 2-level EM (thin) and the GRU (thick, the mean of its three seeds), over the subjects every model holds out that keep the bin. **a**, by consecutive losses; **b**, by the last two trials (S same side, O other side, + rewarded, − not; most recent first); **c**, **d**, the hazard, by the current run's length after an unrewarded or a rewarded trial; **e**, **f**, right after a switch, by the length of the run before it and the switch trial's reward.
<!-- END result-8 -->

The losses are not the only miss. Right after a switch (Figure 4e, f), Lopez-DoubleTrace,
Lebedeva-PR and Miller-RHG as 2-level EM switch back too often, by about 0.05 to 0.08, and
CompareToThreshold stays flat where the mice rise with the length of the run before the switch.
On the hazard and the last two trials (Figure 4b–d), every model is close to the mice.

### 4. The GRU is as close as the best cognitive model, and follows the mice's losses curve

<!-- BEGIN result-3 -->
![scatter](../fig_r3_scatter.svg)

**Figure 5.** Study 01 r9's view, in rollout, for each statistic (rows **a** to **e**): each point is one bin, the subjects' mean against the mean of the model's rollouts, over the subjects every model holds out that keep the bin. Columns as in r2 Figure 3: the five literature models with the best 2-level EM likelihood, as 2-level EM, from the lowest likelihood to the highest, then the GRU. Each panel prints r and RMSE over the bins (each bin once) and the model's floor ratio for the statistic (its cell in Figure 1a, trial-weighted). Run lengths are shares of runs, not p(switch).
<!-- END result-3 -->

In rollout, the GRU (score 1.41; seeds 1.38 to 1.43) ties the best literature cell
(CompareToThreshold per subject, 1.39), as it tied the best 2-level models in replay (r2).
Its bin-level curves are the closest of the six in every statistic (smallest RMSE; highest r in
all but the after-a-switch statistic), the same average-mouse advantage that r2 found. The losses row shows its largest lead:
it is the only one of the six whose curve rises with the mice's (r = 0.96).

### 5. The simulator's missing holds do not drive these results

<!-- BEGIN result-4 -->
![sets](../fig_r3_sets.svg)

**Figure 6.** Rollout fingerprint score per session group, for the models of Figure 5, on groups where at least 8 subjects have a floor (in brackets: the GRU's count; a floor needs two sessions of the group). Left of the line: the headline and its groups. Right: the early coupled stages and the warm-ups, where the gym does not hold a block as the real task's `AdvancedBlockAuto` does (ADR 0006).
<!-- END result-4 -->

The gym cannot hold an early coupled block until the mouse follows the better side
(`AdvancedBlockAuto`), so ADR 0006 kept those stages out of the headline. Yet they score like the
headline groups (Figure 6): no model is clearly worse there. The warm-ups score worse for every
model, around 2.

<!-- BEGIN result-5 -->
![holds](../fig_r3_holds.svg)

**Figure 7.** The `BlockMinReward` check, on the 44 score sessions of 9 subjects that record a positive minimum. The other coupled and warm-up sessions record 0 (coupled with minimum 0: 224, coupled with minimum 1: 34, warm-up with minimum 0: 53, warm-up with minimum 1: 5, warm-up with minimum 10: 5). Per model and statistic: the gap between rollouts with the hold on and off (same seeds), over the subjects' bins, as a multiple of the subjects' split-half floor on those sessions. Largest: 0.45.
<!-- END result-5 -->

The one hold the real task records, `BlockMinReward`, is off (0) in 224 of 258 coupled score
sessions, and in every early-stage one. Where it is on, turning it on in the gym moves every
model's fingerprints by at most 0.45 of the floor (Figure 7). So ADR 0006 stands, and the
simulator stays as it is.

## Caveats

- **The holds check is narrow.** It covers 44 sessions of 9 subjects (6 with a floor), with
  minimum 1 (or 10, in 5 warm-ups). It cannot speak to the early stages where P0 measured held
  blocks, because they record no minimum; those blocks come from `AdvancedBlockAuto` or the
  manual next-block button, which are not recorded. The check also compares a nearly noiseless
  difference (paired rollouts with the same seeds) with a floor that contains real session
  change, so it passes any small effect.
- **Coverage.** 125 of the 149 test subjects have a gym task, and the per-group panels of Figure 6
  rest on 8 to 23 subjects each (the headline on 118 to 121).
- **Real blocks are longer than the gym's** (Gates): partly ignored trials, which the gym never
  plays, and partly held blocks.
- **The gym's version is not in the freezes.** The rollouts ran with `aind-behavior-gym` 0.4.5
  (the PyPI release, in the mevolve env); a later freeze should record it.
- **The mechanism in Result 3 is a hypothesis.** The rollouts were frozen as per-subject tallies,
  not per session, so the spread test needs a new freeze.

## Gates

<!-- BEGIN result-6 -->
**Gates.** Each agent, replayed on the first rollout it played, returns the probabilities it played with (largest difference 0.0e+00). The numpy GRU replays every held-out session to the wrapper's stored p(right) within 7.3e-07. Tasks: 1765 score sessions; 373 left out (no recorded block parameters, or one uncoupled reward probability), which removes every score session of 24 of the 149 subjects. Coupled blocks in the gym average 29.3 trials against 35.7 in the real sessions, which count ignored trials and held blocks. GRU rollout scores per seed: 1.42, 1.38, 1.43.
<!-- END result-6 -->
