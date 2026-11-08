# Future directions — after the first production search

Ideas from the first production search, `seeded-islands@20261106-114959` (`seeded-islands-2`,
2026-11-06 to 08), brainstormed with Han while it ran. Each idea names the evidence that prompted it
and where it is tracked. Nothing here changes the running search or a pre-registered rule; anything
adopted goes through an issue, an ADR or a new variant first.

## What the search showed

| observation | evidence |
|---|---|
| A real gain over the literature models | best seed (Miller-RHG) 72.608 → best 72.771 on the 48 validation mice; the top 10 gain 0.12–0.15 over the seed on 100 unused mice too (paired SE 0.01–0.02; #128) |
| The gain came early, then a plateau | stopping-rule rises at proposals 1, 53, 118 and 513; nothing above 72.771 after proposal 513 |
| The top programs are tied | their monitor gains differ by 1–2 SE; the newest best's validation edge (+0.16) shrank to +0.13 on the monitor mice |
| One shared core | dual learning rates (rewarded ≈ 0.55, unrewarded ≈ 0.25), forgetting of the unchosen side (≈ 0.45), a counterfactual update in coupled tasks (≈ 0.5), fast and slow choice habits; in nearly every program after proposal 400 |
| A 12th slot that cycles | reward-rate gating, satiety, training experience, win-stay/lose-shift, bait expectation: gains centred near 0, sign varying across mice |
| The parent pool creeps, children do not | median parent 72.725 (proposals 300–450) → 72.760 (1,800+); median child flat near 72.68; 1–5% of children reach their parent |
| Noisy scoring | the 2-level fit fails at random on single mice; 8 of 12 rescored programs had one or more such mice, moving pooled scores by up to 0.25 |

The GRU comparison is on other mice (the test subjects); its value is read from
`analysis/explainers/explainer_refs.json`, never retyped here.

## 1. Biology: follow-up analyses (brainstormed with Han, 2026-11-08)

The primary goal is solid biological insight, especially **individual differences** and
**task-variant differences**. The search gives something a single fitted model does not: an
ensemble of near-equally good models. A claim is solid when it holds across that ensemble, across
mice, has a model-free signature in the choices, and replicates on mice not used to find it.

### First step for every analysis
Refit a chosen set of programs on all usable mice and keep per-mouse and per-session parameters and
per-trial predictions (the search kept only pooled scores and summaries).

### A. Across programs: what is robust
1. Tag each program's mechanisms from its code; relate mechanisms to score across the whole archive
   (a free ablation study).
2. Report only claims shared by the top ensemble (e.g. "rewards teach about twice as much as
   misses" holds in all 12 top programs).
3. Ablate each core mechanism from the final programs; loss on unseen mice with SEs.

### B. Individual differences
1. **Null-spread test**: simulate each mouse from the core model without an add-on, refit with it,
   and compare the null spread of the fitted gain with the real spread. A wider real spread means
   real individual differences (gate 4's recovery code does most of this).
2. **Which mechanisms each mouse needs**: per-mouse, cross-validated likelihood gain of each
   mechanism; random-effects model selection for the share of mice using it.
3. **Strategy types**: a mice × programs score matrix (~50 representative programs); factor or
   cluster it.
4. **Reliability**: early vs late sessions, even vs odd; a trait must be stable before it is read.
5. **GRU embedding alignment**: which interpretable parameters the GRU's subject embedding encodes,
   and which embedding directions they leave unexplained (pointers to missing mechanisms).
6. **Biological covariates**: sex, genotype, age, rig, date. The data freeze has none of these; they
   must come from the lab's metadata.

### C. Task-variant differences
1. Per-variant parameters within the same mouse (paired). Does the counterfactual update appear in
   uncoupled tasks too (one top program fitted 0.21 there: over-generalised task structure)? In
   baited tasks, do mice expect the unchosen side to fill up?
2. Learning the task: per-session parameters against training index and stage (does the
   counterfactual weight grow as mice learn coupled tasks?).
3. Mice against the optimal agent for each variant's block statistics.
4. Where models fail: per-trial loss by variant and by position after a block change
   (unknown-stage coupled-baited sessions score far below the rest). Trial-level comparison with the
   GRU is only possible on the test subjects, after their single pre-registered scoring, and is
   exploratory.

### D. Rules for solid claims
- Discovery on generation, validation and monitor mice; confirmation on the ~300 still-unused
  mice; test subjects only after ADR 0007's scoring.
- Parameter recovery and null simulations for every parameter claim.
- A model-free signature in the raw choices for every claim (the fingerprint code measures these).
- Questions written down before looking.

### Open decision
Reports r5+ of this study, or a new study for the biology question. The question differs from this
study's ("can search close the gap"), which argues for a new study.

## 2. Search harness (MEvolve)

| idea | why | status |
|---|---|---|
| Rotate the validation mice when the search plateaus; variants: paired random subsets, lexicase selection over mice | the parent-pool creep suggests tuning to one fixed set of 48 mice; ties freeze parent choice. None of the *Evolve harnesses checked so far rotates its evaluation cases, so this is a novel add-on to speed up search | [MEvolve #80](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-mevolve/issues/80) |
| Exploration agent: hypotheses from exploratory data analysis steer the search | Han's preferred harness novelty | [MEvolve #35](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-mevolve/issues/35), planned variant |
| Meta-summary: Shinka's periodic summary of what has worked, fed back into prompts | the search cycles through the same add-ons; a summary could stop repeats | idea |
| Structural rewrites encouraged in the prompt | small edits to Q-learning parents rarely reach state-space models | idea |

## 3. Lower-noise scoring

| idea | effect | cost per evaluation | status |
|---|---|---|---|
| Per-mouse 2-level fallback; diagnostics when the generation fit fails (21% of proposals got none) | removes random jumps of 0.1–0.25 | none | [MEvolve #81](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-mevolve/issues/81) |
| 96 validation mice instead of 48 (more mice drawn by the same seeded rule) | SE ÷ ≈ 1.4 | × ≈ 1.5 | idea |
| Both fit directions (even→odd and odd→even) | SE ÷ ≈ 1.4 | × 2 | ADR 0003 "considered options" |
| More EM iterations (`max_em` 20 → 50) | fewer unfinished fits | × 1.5–2 | idea |

## 4. Model space

| idea | evaluator change | status |
|---|---|---|
| Discrete states (GLM-HMM-like engagement / bias states) as a forward filter inside the latents | none; likelihood stays exact; maybe 8 latents instead of 6 | idea; an island seeded with a state-based program |
| Deterministic drift within a session (satiety, fatigue) | none; already in the search (`satiety_value_sensitivity`) | in use |
| Random drift, approximated by a few discrete levels | none (an HMM over levels) | idea |
| Random drift, exact (particle filter) | new regime: noisy likelihoods, harder fits | PLAN "Later" |
| Three choices (left, right, ignore); from-scratch search with no library | — | PLAN "Later" |

## 5. Generalisation checks for this search

| check | status |
|---|---|
| Top programs rescored on 100 monitor mice | done (#128, PR #129) |
| Late vs early: the final top 10 against the proposals-250–435 top 10 on the monitor mice; tells real late gains from tuning to the validation mice | after the search ends |
| Rescore the archive's 40 parents on a fresh 48 mice; how much would rotation reshuffle the ranking | after the search ends (MEvolve #80) |
| An estimated GRU line on the validation mice: best seed on validation + (GRU − best literature model) on the test subjects, from committed scores | needs regime-matched literature scores (study 12 r2) |

## 6. Compute and operations

| idea | why | status |
|---|---|---|
| Stop path: snapshot at SIGTERM, cancel evaluation jobs on exit; a killed pool worker must fail the evaluation, not hang it | orphan jobs; an OOM-killed worker hung a 148-mouse rescoring | [MEvolve #72](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-mevolve/issues/72) |
| Ask the HPC admins for a QOS bump | fair-share 0.024–0.05; nights ran 1–8 evaluations an hour against ~100 when the cluster was free | idea; draft offered |
| Code Ocean: one large machine, evaluations local | independence from Slurm priority; ~$400 per 2,000 proposals (estimate) | open questions: machine sizes, internet access, who pays |
| A W&B heartbeat during long quiet stretches | the run shows "crashed" after 5 minutes without a log | idea |
| A live progress figure inside MEvolve (now a scratch script) | the hourly figure proved useful | idea |

## Considered and set aside

| idea | why not |
|---|---|
| Score every new best on the test subjects | breaks ADR 0003/0007 (test scored once, never fed back) and inflates the reported score; the monitor mice answer the same question cleanly |
| Score every new best on the selection subjects | spends the cohort that makes the final choice; the monitor mice were added instead (ADR 0003 amendment, 2026-11-07) |
| Stop the search early on the monitor result | would feed monitor scores back into the search; it runs to its pre-registered stopping rule |
