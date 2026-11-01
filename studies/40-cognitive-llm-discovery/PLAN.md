> **Draft, before the design grill.** Copied from the planning session of 2026-10-31. The grill
> (`/grill-with-docs`) settles the open decisions marked **[open]** and records them in `adr/`;
> this file is then trimmed to the build order and the settled design.

# Plan — Study 40 — LLM-driven discovery of interpretable cognitive programs for mouse dynamic foraging (round 1, lean)

Written 2026-10-31 (Seattle); final revision 2026-10-31 22:40 PT. To execute in a new session:
start at step −1 (§6), then follow §8. Both API keys are stored and verified (2026-10-31):
`~/.config/mevolve/wandb_api_key` (W&B user `han-hou`, team NeuralDynamics) and
`~/.config/mevolve/anthropic_api_key` (lists `claude-sonnet-5-5`). **[settled]** = decided with Han; **[recommended]** = proposed in this
session, decided by approving this plan; **[open]** = to settle later. Sources: two codebase
explorations (baselines and splits; data layer and HPC), the CogFunSearch PDF Han uploaded, the
DataDIVER preprint, the ShinkaEvolve source code, two design passes (harness internals; repository
integration), and a branch survey of the lab's GLM-HMM work. Every number below was read from a
committed file, a paper, or code.

## 0. At a glance

- **Question.** Can LLM-driven program search find a short, readable JAX cognitive program that
  closes the gap between the best hand-written cognitive model and the GRU on held-out sessions of
  held-out mice, and generates realistic behaviour in closed loop with the task?
- **Objective [settled].** Hill-climb held-out likelihood on validation mice, then simplify under a
  fit floor; behavioural fingerprints and closed-loop rollouts are feedback and gates, never part of
  the score. Pilot $500 on Sonnet 5.5. Test set = the existing 149 held-out mice, touched once.
- **Engine [recommended].** ShinkaEvolve (Sakana AI, Apache-2.0) runs the search, pinned to one
  commit, with a small tested compatibility shim; we write the evaluator, the seed library and the
  gates in a new repo **`aind-dynamic-foraging-mevolve`** (MEvolve: M for mice and models; package, CLI and env `mevolve`),
  paired with `aind-dynamic-foraging-models`. Fallback if the engine smoke fails: a custom loop on
  the same evaluate contract.
- **Session heterogeneity [settled].** Programs are fitted with a session level: each session gets
  its own parameter values, pulled toward the mouse's typical values (study 13's two-level model, no
  population level); a held-out session is predicted by reweighting candidate parameter sets with its
  own earlier trials. Programs also receive each session's training stage, task variant and session
  number, and every score and feedback line is split by stage and task variant. The program carries
  the shared computation; mouse and session parameters carry individual and day-to-day differences;
  the program's latent variables (for example Q values) carry trial-by-trial internal quantities.
  Terms (Han, 2026-10-31): a *latent variable* is a continuous internal quantity such as a Q value;
  a *state* is a discrete hidden state as in a GLM-HMM (engaged, disengaged, ...). A program may
  have latent variables and no states; a program with states carries its belief over them as
  latent variables.
- **Headline.** Fraction of the gap closed, `(program − HB-CTT) / (GRU − HB-CTT)`: floor = HB-CTT
  0.7225 (the same session-level regime), ceiling = GRU v2 H128 0.7282; per-mouse MLE CTT 0.7170 and
  GRU H256 0.7290 as reference lines. Plus generative match.
- **Gates before real spend.** (0) data parity; (1) Hattori and CTT as programs reproduce the
  committed MLE numbers, and under the session level land near study 13's HB numbers; (2) evaluator
  timing; (3) engine smoke ≈ $7; (4) recovery of a known program from simulated mice ≤ $20; then the
  pilot.
- **First deliverable, before any implementation.** A 3Blue1Brown-style animated explainer (HTML,
  about 2.5–3 minutes, six scenes) and a static schematic of the study's steps, for Han's review.
- **Han decides.** After the explainer: the storyline. Step 0: ADRs (split, parity tolerances, what
  counts as interpretable) and the go-ahead to create `aind-dynamic-foraging-mevolve` under the
  org. After the 100-proposal smoke: go for the pilot. After the pilot: the larger run and the next
  parking-lot item.
- **Effort.** About five weeks of engineering to the first pilot result.

## 1. Context

The BFM project pinned both ends of one axis on one protocol (snapshot `20260603`; 149 held-out mice;
per mouse, even sessions are context and odd sessions are scored; ignored trials excluded; metric =
held-out per-trial likelihood `exp(mean log p(choice))`, pooled over trials; chance 0.50):

| model | held-out likelihood | committed source |
|---|---|---|
| per-mouse MLE Hattori2019 / Bari2019 / CompareToThreshold | 0.7127 / 0.7149 / 0.7170 | study 01 r8 `analysis/rl_baseline.json` (`cross_study.rl_baseline`) |
| two-level HB-Hattori / HB-CTT: one distribution per held-out mouse over its sessions, no population level | 0.7213 / 0.7225 | study 13 r1 / r5 |
| three-level HB with a population level | 0.7098; below per-mouse MLE for 108–121 of 149 mice; chains did not mix | study 08 (verdict negative) |
| GRU H128 v1 / v2; H256 | 0.7268 / 0.7282; 0.7290 | study 01 `report_data.json`, `scaling_results.csv`; study 11 `reference_points.csv` |

The cognitive-to-GRU gap is about 0.011; +0.001 on this axis is about 0.7 nats per 500-trial session
(project rule of thumb). Nothing between the hand-written models and the GRU is interpretable.

**Sessions within a mouse differ, and that is most of the gap.** Of the 22,555 sessions of the 763
cohort mice in snapshot 20260603, 55 % are at the final stage or graduated, 17 % at a recorded
earlier stage and 28 % have no stage recorded; 83 % of the mice ran more than one task variant (read
with `aind_dynamic_foraging_database.select_sessions`, all sessions before the loader's filters). Study
13's ladder (r4, Hattori, same 149 mice) shows what inferring each session's own parameters is worth:

| how a held-out session is predicted | likelihood | step |
|---|---|---|
| one parameter set per mouse (MLE) | 0.7127 | |
| one HB point estimate | 0.7107 | −0.0020 |
| candidate parameter sets reweighted by the session's own earlier trials | 0.7174 | +0.0067, all 149 mice gain |
| the same, with sessions free to sit away from the mouse's typical values | 0.7213 | +0.0039, 95 % of mice gain |
| GRU H128 v1 | 0.7268 | |

A search with one fixed parameter set per mouse could only win that ~0.01 by writing parameter
inference into the program, folding stage, task and day-to-day drift into "mechanisms". Fitting
programs with a session level removes that pressure (decision 7).

**Ultimate goal (Han).** A model that explains the mice's shared computation and their heterogeneity
across mice and across time, so that its parameters and latent variables can later be related to
metadata and to photometry and Neuropixels recordings. Round 1 delivers the shared computation (the
program), mouse-level and session-level parameters (heterogeneity across mice and days) and the
program's per-trial latent variables (and, for programs with discrete states, the per-trial state
probabilities), all exported keyed by session and trial.

**Prior art, verified.**
- *CogFunSearch* (Castro, Tomasev, …, Miller, Stachenfeld; ICML 2025; bioRxiv
  10.1101/2025.02.05.636732). FunSearch outer loop (Gemini 1.5 Flash mutates JAX programs; prompt =
  two parent programs + scoring description) and an inner loop fitting per-subject parameters by
  gradient descent (AdaBelief, lr 0.05, ≤ 10,000 steps, up to 10 restarts). Interface
  `agent(params, choice, reward, agent_state) -> (choice_logits, agent_state)`. Score = per-subject
  two-fold CV over even/odd sessions; half the subjects held out. Best programs after 1.24M (human),
  293k (rat), 196k (fly) samples. Rat: baseline 66.63, RNN 67.41, CogFunSearch 67.22 (normalized
  likelihood, %). Seeding from the published model worked best for rats and flies. A cascade (proxy
  on 20 subjects, keep if ≥ the 90th percentile) reached rat 67.03 with an order of magnitude fewer
  samples. Pathologies: indexing past the parameter array, unused latent variables, parameters that fit to zero,
  many near-equivalent programs.
- *DataDIVER* (Kasenberg, Castro, …, Daw, Stachenfeld, Miller; bioRxiv May 2026). AlphaEvolve +
  Gemini 2.5 Flash; Stage 1 maximize fit, Stage 2 minimize Halstead effort under a fit floor; fit-only
  programs matched the RNN on average and beat hand models by 4.1 points. Code not public.
- *ShinkaEvolve* (Sakana AI; Apache-2.0; 1,490 stars; last push 2026-10-05). Details in §4.4.

## 2. Decisions

1. **Objective [settled].** Stage 1 hill-climbs one scalar: held-out per-trial likelihood on
   validation mice. Stage 2 minimizes Halstead effort (`radon`) under a fit floor. Fingerprints and
   rollouts are feedback, gates and diversity axes, never in the score.
2. **Engine [recommended, changes Han's earlier "custom harness" choice].** Asked what a custom
   harness gains over ShinkaEvolve: for the lean round 1, mainly control and learning by building;
   the evaluator, where the science and most of the risk sit, is ours either way. So Shinka runs the
   search; we own the evaluator, seed library, gates, shim and provenance wrapper. Harness-level
   research (Pareto archive, tempered islands, tree search) moves to the parking lot and would fork
   Shinka or replace it then. The calibration gate and its variant disappear.
3. **Code location [settled + recommended].** New repo `aind-dynamic-foraging-mevolve` (name
   settled by Han). Study 40 in the studies repo holds the explainer, the ADRs, the variant manifests,
   launch records, frozen results and reports; it is scaffolded at step 0 (cheap, claims the number),
   while the studies-tooling fixes land just before the first real launch (step 5) and gates 0–3 run
   as tool tests in mevolve [recommended middle path].
4. **Pilot [settled].** $500 hard cap, `claude-sonnet-5-5`; larger runs only after the pilot's
   best-validation-vs-samples curve is inspected.
5. **Test set [settled].** The 149 held-out mice (studies 01/08/13 split), touched once; generation
   and validation mice carved from the 614 training-pool mice.
6. **Round-1 scope [settled].** Lean: variants `base` and `seeded-ctt`, the gates, four reports,
   latent exports, and the 17-model seed library. Everything else is a parking-lot line (§10).
7. **Fitting regime [settled 2026-10-31].** The search fits programs with a **session level**:
   per mouse, each session's parameters are drawn around the mouse's typical values, with **no
   population level** (study 08 showed pooling across mice did not converge and scored below
   per-mouse fits). Held-out sessions are scored by the session marginal (in-session reweighting, as
   study 13). Programs receive session facts (training stage, task variant, session number), and all
   scores and feedback are split by stage and task variant. Fallback if the session-level scorer
   fails gate 1c: per-mouse fitting with session facts and split diagnostics, elites re-scored with
   the session level afterwards.
8. 2-way choice (ignore trials excluded), choice and reward inputs only.

## 3. Protocol

### 3.1 Metric and comparators
- Score: `exp(Σ log p(choice_t) / Σ n_trials)` over all scored trials of the tier, probabilities
  clipped to [1e-10, 1 − 1e-10] as in the wrapper's `_compute_normalized_likelihood`
  (`code/evaluation/baseline_rl_evaluation.py`); nats per trial beside it; never mixed with
  per-mouse means (differs by ~0.007).
- Headline `gap_closed = (program − HB-CTT) / (GRU v2 H128 − HB-CTT)`, program scored with the
  session level (the HB's own regime); per-mouse MLE CTT and GRU H256 as reference lines, and the
  program's per-mouse score as a ladder rung (§3.7); all read from committed JSON via
  `studies/util/cross_study.py` or the reference registry (readers for HB and GRU v2/H256 to add,
  issue #63); never retyped.
- Paired per-mouse comparison against HB-CTT and MLE CTT on the 149, and against the GRU on the 146
  mice shared with it (study 13 r2 convention).

### 3.2 Three-tier split (snapshot `20260603`, `min_sessions=10`)
| tier | mice | role |
|---|---|---|
| generation **G** | ~48 from the 614 training-pool list (`studies/10-gru-reservoir-ablation/reference/study01-v2-exact-split.json`), stratified by curriculum, fixed seed | their diagnostics go into the LLM's feedback |
| validation **V** | ~48 disjoint mice from the same pool | `combined_score`: selection, ranking, stopping, the Stage-2 floor |
| test **T** | the 149 held-out mice | touched once per variant with a pre-registered top-3 |

Within every mouse: sessions in date order, even positions = context (parameters fitted), odd =
scored (`compute_train_eval_session_ids`, `eval_every_n=2`). On G and V both CV directions are
averaged; on T only the comparators' direction. G and V sizes are fixed by gate 2 **[open]**.
Shinka shows the selection score to the LLM (§4.4), so the V score is visible as one number per
program; V's per-mouse detail never is.

### 3.3 Fitting and cascade (from CogFunSearch, DataDIVER and study 13)
- **Session-level fit, per mouse, in JAX.** Unconstrained parameters θ_s for each context session,
  with θ_s ~ N(μ_m, σ_m²) per parameter; μ_m free; a fixed weak hyperprior on log σ_m (study 13 ADR
  0003's fixed-hyperprior idea; a Gaussian on log σ also keeps the joint maximum away from σ → 0).
  MAP by gradient descent: `optax.adam`, cosine schedule from lr 0.05, fixed step count, `R` restarts
  from U(−2, 2), `vmap` over mice, sessions and restarts; raw parameters mapped into the declared
  bounds by the evaluator; θ clipped to [−10, 10]. `optax.lbfgs` polish only if gate 1b fails.
- **Session-marginal score.** For a held-out session, K candidate parameter sets are drawn from the
  mouse's fitted session distribution (plus a Laplace term for the uncertainty of μ_m if gate 1c
  needs it); each trial is predicted by averaging the candidates weighted by how well each explained
  the session's earlier trials, which equals the log-mean-exp of whole-session likelihoods. Common
  random draws (fixed seed per mouse) across programs; per-session effective sample size reported;
  K fixed by gate 1c (start 256).
- **Per-mouse rung.** The same program with one parameter set per mouse is computed for elites only
  (§3.7) and as the Stage-A screen.
- Cascade inside `evaluate.py`: **Stage A** (24 G mice, 6 + 6 sessions, per-mouse fit, R = 4, 150
  steps, 120 s cap) fails fast; **Stage B** (all G and V, session-level fit, R = 8, 300 steps,
  session-marginal score, cap set by gate 2) gives the score. A cache keyed by the program's AST hash
  returns stored results for duplicates.
- Static checks first: declared parameters, bounds-checked indexing, finite outputs on a probe
  sequence, JIT compile under a time limit; failures return the error text as feedback and no score.

### 3.4 Stage 2: simplify
A second Shinka run seeded with Stage-1 elites whose evaluate returns −Halstead effort when V ≥ floor
(floors at 50 / 75 / 90 % of the seed → best gap) and invalid otherwise; identical per-mouse scores
(within 1e-4) mark a refactor rather than a simplification. Report the fit–complexity curve.

### 3.5 Fingerprints and generative protocol (feedback and gates, never in the score)
Same code on real data and rollouts: win-stay / lose-switch; p(switch) by reward × run length and by
the last N = 1..3 (choice, reward) patterns (study 01 `generative_match.py` statistics); logistic
choice-history kernels (`aind_dynamic_foraging_models.logistic_regression.fit_logistic_regression`);
choice around block transitions; foraging efficiency (`aind_dynamic_foraging_basic_analysis`).
Rollouts: elites run as agents in `aind_behavior_gym` `CoupledBlockTask` (or `UncoupledBlockTask` by
curriculum) with each mouse's fitted parameters, fixed seeds. Example-session figure with
`plot_foraging_session(..., ax=ax, fitted_data=p_right)`, right `"b"`, left `"r"`.

### 3.6 Latent exports
For every elite: per-trial latent-variable trajectories (named by `LATENT_NAMES`), per-trial state
probabilities for programs with discrete states, per-session fitted parameters
and per-mouse typical values and spreads, as parquet keyed by `session_id` and `trial` on scratch,
with a committed summary.

### 3.7 Session facts and split diagnostics
- **Session facts** from the session table, frozen with the data: training stage (`current_stage_actual`,
  ordinal, with an `unknown` level for the 28 % without one), task variant (coupled or uncoupled ×
  baited or not), and the session's index in the mouse's history. They reach the program through
  `init_latents(p, session)` (§4.2), so known drift can be stated as rules ("inverse temperature rises
  with stage"); the session level absorbs the rest.
- **Splits everywhere.** Every score in `public`, every line of `text_feedback`, and every report table
  is split by stage group (final or graduated, earlier, unknown) × task variant, so the LLM and the
  reader see where a program wins.
- **Both-regime test for elites.** Each elite is scored per mouse and with the session level; a
  mechanism whose advantage over its parent appears only per mouse is flagged as absorbing session
  drift rather than describing within-session computation.

## 4. Evaluator and search engine

### 4.0 Repository map
```
+--------------------------------------------------------------------------------------------+
| aind-dynamic-foraging-mevolve   (NEW, pairs with aind-dynamic-foraging-models)         |
|                                                                                            |
| data freeze      split + parquet, built once from the S3 snapshot                          |
| evaluator        session-level fit, held-out scoring, fingerprints, rollouts, latents      |
| seed library     17 published models as programs, each parity-tested                       |
| Shinka adapter   initial.py, evaluate.py, compatibility shim for Claude 5.x                |
| launch wrapper   W&B group, launch records in study 40                                     |
+----------+----------------------+----------------------+------------------------+----------+
           |                      |                      |                        |
   runs on |              imports |            uses once |          reads, writes |
           v                      v                      v                        v
+---------------------+  +-----------------+  +--------------------+  +----------------------+
| ShinkaEvolve        |  | -models         |  | BFM stack          |  | -bfm-studies         |
| (Sakana AI, pinned) |  | numpy foragers: |  | wrapper: loader    |  | study 40: manifests, |
| islands, archive,   |  | parity oracles  |  | dispatcher: data   |  | launch records,      |
| Claude proposals,   |  | behavior-gym:   |  | config, provenance |  | frozen results,      |
| web UI              |  | task simulator  |  | recipes            |  | reports, explainer   |
+---------------------+  +-----------------+  +--------------------+  | studies 01, 13:      |
                                                                      | comparator numbers   |
                                                                      +----------------------+
           |                                             |
           v                                             v
 Anthropic API: Claude                         S3 snapshot 20260603
 W&B: NeuralDynamics / aind-dynamic-foraging-model
```
One search run is one SLURM job on a 112-core node: the Shinka controller keeps up to 8 Claude calls
in flight and 24 evaluator processes busy (4 cores each); its SQLite lives on the node's `/local1`
and is copied to `/allen/aind/scratch/han.hou/mevolve/<group>/`; the web UI opens through an SSH
tunnel.

### 4.1 `aind-dynamic-foraging-mevolve`
Checkout `/home/han.hou/code/aind-dynamic-foraging-mevolve`; package and CLI `mevolve`; own
conda env `mevolve` (py 3.12, jax 0.6.1, optax, numpy, pandas, pyarrow, radon, anthropic, wandb,
pytest, hypothesis, `shinka-evolve @ git+https://github.com/SakanaAI/ShinkaEvolve@<pinned sha>`,
`aind-dynamic-foraging-models[rl]`, `aind-behavior-gym`, `aind-dynamic-foraging-basic-analysis`).
Contents: `scripts/freeze_data.py` (run once with `PYTHONPATH=<wrapper>/code` in the shared env; builds
the split through the wrapper's own loader; writes `data/mice_20260603.parquet`, `data/split.json` and
a sha256 manifest), `mevolve/{program,replay,fit,evaluate,fingerprints,rollout,complexity}.py`,
`seeds/` (§4.5), `shinka/initial.py` + `shinka/evaluate.py` (the adapter), `mevolve/shinka_compat.py`
(§4.4), `mevolve/launch.py` (provenance wrapper), `tests/`. Reviewed elites can later graduate into
`aind-dynamic-foraging-models` as numpy foragers, closing the loop between the paired repos.

### 4.2 Program interface (the code inside Shinka's `EVOLVE-BLOCK`)
```python
"""<model description written by the LLM>"""
import jax, jax.numpy as jnp
PARAMS = {"alpha_rew": (0.0, 1.0), "alpha_unrew": (0.0, 1.0), "forget": (0.0, 1.0),
          "beta": (0.0, 100.0), "bias_left": (-5.0, 5.0)}   # free params per session, name -> bounds
N_LATENT = 2                                                # latent variables, reset each session
LATENT_NAMES = ("q_left", "q_right")                        # names for the latent exports
def init_latents(p, session): ...  # p: dict of this session's bounded parameters; session: facts
                                   #   {stage: int (−1 unknown), coupled: 0/1, baited: 0/1, index: int}
def predict(latents, p): ...                                # -> logits over [left, right]
def update(latents, choice, reward, p): ...                   # choice int in {0,1}, reward float {0,1}
```
`init_latents` may return derived quantities among the latents (for example a stage-scaled temperature),
which is how a program states rules for known drift; the evaluator draws `p` per session (§3.3).
Limits in the task card: ≤ 12 parameters, `N_LATENT` ≤ 8, imports only `jax`, `jax.numpy`, `jax.nn`,
`jax.lax`, `math`; no Python control flow on traced values. The evaluator: AST allowlist, `exec` into a
bare namespace, `jax.eval_shape` before data; `replay_session` = `lax.scan` over trials (trial 0
predicted from `init_latents`, padding masked); `rollout_session` = Python loop around a gym task,
elites only. Static shapes (`T_pad` buckets {512, 1024, 2048}), one compile per (program, stage);
Stage B with the session level and K = 256 scoring draws is expected at a few minutes on 4 cores plus
compile (gate 2 measures it and sets the cap).

### 4.3 Evaluate contract (Shinka's, verified in its source)
`shinka/evaluate.py` calls `mevolve.evaluate` through `run_shinka_eval(num_runs=1)` and returns
`combined_score` = **100 × V likelihood** (percent points as in CogFunSearch's tables; Shinka prints
the score with two decimals, so 0.7217 would show as "0.72" and hide every difference that matters,
while 72.17 resolves 1e-4), `public` (G likelihood ×100, split by stage group × task variant, paired
delta vs the CTT and Hattori seeds, n_params, n_latent, Halstead effort), `private` (per-mouse parquet
path, fitted typical values and spreads, effective sample sizes, fit diagnostics, timing),
`extra_data` (AST hash, score-vector hash), `text_feedback` (≤ ~60 lines: where the program loses by
stage and task variant, worst and best five G mice by anonymized index with fitted parameters and
spreads, parameters at bounds, fingerprint mismatches, error text translated into advice). Failures: invalid, never score 0. A standalone
`mevolve evaluate --program p.py --cohort {gen,val,test}` serves the gates; `test` needs
`--allow-test`.

### 4.4 Search engine: ShinkaEvolve, pinned, with a compatibility shim
- **What it provides (read in its source):** `initial.py` with `EVOLVE-BLOCK-START/END`; islands with
  migration and an archive (`num_islands`, `migration_interval`, `parent_selection_strategy`
  weighted / power-law / beam, `archive_size`); diff, full-rewrite and crossover patches; a novelty
  filter (code embeddings, optional LLM judge); meta-recommendations (`meta_rec_interval` = 10: an LLM
  summarizes what worked and later prompts see it); UCB selection over an LLM ensemble; optional
  system-prompt evolution (`evolve_prompts`); text feedback (`use_text_feedback`); a dollar cap
  (`max_api_costs`); asynchronous proposals and evaluations (`max_proposal_jobs`,
  `max_evaluation_jobs`); resume from the results directory; W&B logging; CLIs `shinka_run`,
  `shinka_launch`, `shinka_visualize`.
- **Web UI:** `shinka_visualize --port 8888` serves a live metrics page, a genealogy tree of programs
  and a compare view from the run's SQLite; on HPC it runs on the controller node and opens through an
  SSH tunnel via the login node.
- **Round-1 configuration:** `task_sys_msg` = our task card (2-armed baited block task; inputs = this
  session's past choices and rewards; latent variables reset per session; parameters fit per mouse by the
  harness; chance 0.5; the seed scores); `llm_models: [claude-sonnet-5-5]`, `meta_llm_models` the
  same; `use_text_feedback: true`; 4 islands; `max_api_costs: 450`; `job_type: local` with
  `max_evaluation_jobs: 24` (4 cores each) and `max_proposal_jobs: 8` inside one SLURM job on a
  112-core node; novelty filter off unless an embedding model is available (the default is OpenAI's
  `text-embedding-3-small`); duplicates are caught by the evaluator's AST cache.
- **Compatibility shim `mevolve/shinka_compat.py`** (applied at import, unit-tested against the
  pinned SHA): (1) Claude 5.x requests omit temperature and `thinking.budget_tokens` (both rejected
  with an error; Shinka sends them for models in its reasoning list and samples a temperature
  otherwise) and send `output_config.effort: medium`; (2) `cache_control: ephemeral` on the system
  block (Shinka's Anthropic client sends none, so every call would pay full input price); (3) price
  entries for `claude-sonnet-5-5` (input 2.0, cache write 2.5, cache read 0.20, output 10.0 USD per
  MTok) and `claude-opus-5-5`, cache tokens included in cost, and an unknown model raises instead of
  costing $0 (Shinka's current behaviour, which would silently disable the cap); (4)
  `stop_reason == "refusal"` billed and counted as a failed proposal. If the shim outgrows ~150
  lines or breaks on an upgrade: fork Shinka, or offer the patches upstream to Sakana (both
  outward-facing; Han's go-ahead).
- **Not used:** Shinka's SLURM launcher (one sbatch per candidate, `--ntasks=1` hard-coded, extra
  flags joined onto one line, endless polling when `squeue` fails).
- **Run directory:** Shinka's SQLite on the node's `/local1` (SQLite over NFS is unsafe), copied to
  `/allen/aind/scratch/han.hou/mevolve/<group>/` every N minutes and at exit; resume copies it back.
- **Provenance wrapper `mevolve launch`:** builds Shinka's config from the variant's
  `harness.yaml`, pins the mevolve and Shinka SHAs in `harness_lock.yaml`, refuses dirty trees,
  and from step 5 writes the initial launch record (§5.3).
- **W&B (one run per launch, read in `shinka/wandb_logging.py`).** Shinka's logger is off by
  default; the wrapper turns it on and sets `wandb_entity` = `NeuralDynamics` (the Allen octo
  account; Han, 2026-10-31: not the personal account with its 200 GB cap), `wandb_project` =
  `aind-dynamic-foraging-model` (Han's choice; empty when checked 2026-10-31, so it is dedicated to
  study 40 and one-project-per-study holds; the key's W&B user is `han-hou`, a member of
  NeuralDynamics, verified on both `api.wandb.ai` and `api.forge.coreweave.com`), `wandb_group` = `<variant>@<launch_id>`, `wandb_name` and `wandb_run_id`
  deterministic (Shinka stores the id in the results directory, so a resume continues the same run),
  `wandb_notes` = the launch note, `wandb_config` = the `meta.*` block and SHAs. Shinka then creates
  one run per evolution, logs every evaluated program as one history row (with our `public`
  metrics), logs population snapshots and a final table of all programs; it uploads no artifacts,
  so the wrapper attaches the SQLite database and elite sources to the same run id at the end. Its
  logging is best-effort (failures only warn), so freezes read the database artifact, not the
  history rows.

### 4.5 Seed library
The 17 presets in `ForagerCollection.FORAGER_PRESETS` (Hattori2019, Bari2019, CompareToThreshold,
Rescorla-Wagner, Win-Stay-Lose-Shift, ActorCritic, Chen-RLCK, Zid-HistoryKernel2-ForagingRL,
Grossman-MetaLearning, Lebedeva-PR, Beron-RFLR, Miller-RHG, Eckstein-RL, Eckstein-BI,
FeedbackDependent-RL (+Bias, +CK1), Alsio-DualRateSticky-RL, Lopez-DoubleTrace) ported to the program
interface, each with a parity test against its numpy forager (`tests/test_seed_parity.py`,
parametrized; Findling's stochastic particle filter excluded). Uses: seeds (`seeded-ctt` in round 1),
two or three library programs with their G scores in the task card as inspirations, and a
published-model leaderboard on the 149 test mice (r1).

### 4.6 Sandboxing (accidental, not adversarial; tools verified on the nodes)
AST import allowlist; Shinka runs each evaluation in its own process; the evaluator adds
`RLIMIT_CPU`, `RLIMIT_NOFILE`, `RLIMIT_FSIZE`, memory via `RLIMIT_DATA` (8 GB) or the SLURM cgroup
(not `RLIMIT_AS`: XLA reserves large virtual mappings), `unshare -rn` for no network where the
launcher allows it, and a per-job working directory as the only writable path.

## 5. Study integration

### 5.1 Fixes in the studies tooling (verified blockers, minimal fixes)
| # | blocker | fix |
|---|---|---|
| 1 | no `model` tag for LLM-discovered programs | add `cognitive/mevolve` to `schemas/tags.yaml` in the scaffold PR |
| 2 | nothing in the studies repo writes an **initial** launch record; `write_intervention` refuses without one; freezes fail `artifact-group-unrecorded` | add `write_initial()` to `studies/util/launch_record.py` (schema-validated, fills `_meta`), small PR with a test, before the first real launch |
| 3 | `revisions` allows only studies / dispatcher / wrapper / foraging_models SHAs | mevolve and Shinka SHAs in the variant manifest `harness_lock.yaml` and top-level record keys; follow-up issue for an optional `revisions.harness` |
| 4 | `job_refs.type` has no `wandb_run` | `slurm_job` now; `wandb_run_id` / `wandb_run_url` at the record's top level; same follow-up issue |
| 5 | only `extend` may change a manifest key containing `n_steps` | name the proposal budget `n_steps` in `harness.yaml` |
| 6 | `write_intervention` YAML-parses every manifest | only YAML in `manifests`; `harness.yaml` pins `initial.py` and the task card by sha256 |
| 7 | `_meta` knows four source repos | freezes write the mevolve and Shinka SHAs into the frozen JSON body |
| 8 | CI runs `make all` without JAX, W&B or the LLM SDK | every program run is a `freeze-*`; producers read frozen JSON only |
| 9 | 10 MiB per committed file | per-proposal scalars; per-(program, mouse) rows for elites and T only |
| 10 | strict figure checks from the first SVG | prototype the session figure first; `scripts/check_figures.py --study 40-...` |
| 11 | `environment.lock` required with a freeze | written at the first freeze |
| 12 | no reader for study-13 HB or GRU v2/H256; the registry is on unmerged `feat/63-reference-registry` | registry entries after #63 merges; interim: read the JSON by path, declared in `inputs` and `upstream` |
| 13 | the 149 RL/HB mice ≠ the 149 GRU mice (146 shared) | `freeze_split.py` records both sets and the intersection; r4 pairs accordingly |

### 5.2 Scaffold at step 0 (claims the number; dispatcher #194)
Accession **40** (Han's choice; verified free: no directory, branch or issue claims it;
`scripts/scaffold.py` refuses only taken numbers). From a fresh worktree branch off `origin/main`:
```bash
# schemas/tags.yaml: add facets.model.values.cognitive/mevolve (label "Cognitive › MEvolve")
make new-study NAME=40-cognitive-llm-discovery \
  QUESTION='Can LLM-driven evolutionary search over small JAX cognitive programs, with parameters that vary by mouse and by session, find an interpretable model whose held-out likelihood on the 149 held-out mice closes the gap from the best hierarchical cognitive baseline (HB-CTT) to the GRU, and that generates realistic behaviour in closed loop with the task?' \
  PROJECT=aind-dynamic-foraging-model VARIANT=base MOTIVATED_BY=13/r5,13/r4,01/r8 \
  MODEL=cognitive/mevolve QUESTION_TYPE=comparison,interpretation ISSUE=<studies issue number>
make new-variant STUDY=40-cognitive-llm-discovery NAME=seeded-ctt
# + analysis/explainers/ (the approved explainer and schematic), CONTEXT.md, adr/ seeds
make validate && git add -A && git commit -m "feat(study-40): scaffold 40-cognitive-llm-discovery and the cognitive/mevolve tag"
git push -u origin study/40-cognitive-llm-discovery
```
The study's `wandb.entity` is `NeuralDynamics`, not `AIND-disRNN`: the Makefile's `new-study` target
does not pass an entity and `scripts/scaffold.py` defaults to `AIND-disRNN`, so run
`scripts/scaffold.py study ... --entity NeuralDynamics` directly (same arguments as above).
`motivated_by` notes (no retyped numbers): 13/r5 "the best hierarchical cognitive baseline still
trails the GRU; no hand-designed model has closed that gap"; 13/r4 "inferring each held-out session's
own parameters from its earlier trials covers most of the distance from the per-mouse fit to the GRU,
so programs are fitted the same way"; 01/r8 "the GRU beats every per-mouse classical fit on the 149
held-out mice; this study keeps that protocol and its comparators".
Tags `model: [cognitive/mevolve]`, `question: [comparison, interpretation]`; reports add
`methods` (r1–r3) and `generative-run` (r4). Study-local `CONTEXT.md` (program, proposal, elite,
generation / validation / test mice, evaluator, fit floor, fingerprint, gate, rollout, search run) and
`adr/` after a grill session: 0001 objective (decided), 0002 three-tier split (to grill), 0003
evaluation protocol and parity tolerances (to grill), 0004 engine and code ownership (this plan), 0005
what counts as interpretable (to grill), 0006 model and budget (decided).

### 5.3 Variants and launch records
One search run = one launch = one W&B group `<variant>@YYYYMMDD-HHMMSS`; a replicate is a second
initial launch of the same variant; a changed search knob is a new variant. Resume after a crash or
wall time (`resubmit`) and budget extension (`extend`) stay in the launch. The Stage-2 simplify run
changes the objective, so it is its own variant (e.g. `base-simplify`) reading the Stage-1 elites by
digest.

| variant | what | `data.source` |
|---|---|---|
| `base` | minimal seed template; library programs as inspirations | snapshot |
| `seeded-ctt` | seeded from CompareToThreshold as a program (the lab's best per-mouse model; CogFunSearch's published-model seed did best for rats and flies) | snapshot |
| `recovery-ctt`, `recovery-hattori` | gate 4: search on data simulated from a known program | `generated` |

Variant directory: `notes.md`, `harness.yaml` (`n_steps`, `usd_cap`, model id, Shinka settings,
evaluator settings, the V set's sha256, sha256 pins of `initial.py` and the task card),
`harness_lock.yaml` (mevolve and Shinka SHAs, `pip_freeze_sha256`), `initial.py`, `task_card.md`;
records under `launch_record/<label>/` with the submitted copies beside them, later `results.md`.
`mevolve launch` writes the initial `hpc` record via `write_initial()`: `kind`, `platform`,
`wandb_group`, `label`, `note` (≥ 10 chars), `job_refs` (`slurm_job`), `revisions` with genuine full
SHAs, `manifests`, `n_tasks`, `data` (`snapshot 20260603`, `uri`, `manifest_sha256` recomputed with
the dispatcher's `launch_inputs.fetch_manifest` recipe, `build_id`, `n_sessions`), `image`
(`platform: hpc`, env path, `lock_sha256` by the dispatcher's `image_record_hpc` recipe), plus
`wandb_run_id`, `wandb_run_url`, `harness_git_commit`, `shinka_git_commit`, `llm {model, usd_cap}`.
Interventions via `write_intervention`: walltime or crash → `resubmit`; budget extension → `extend`
(only `n_steps` and `usd_cap` change); an unreadable SHA → `backfill`.

### 5.4 Freezes and reports
Run outputs stay on scratch and in W&B; the repo commits frozen inputs keyed by program sha256, W&B
run id and file sha256 (`build_meta(kind="freeze", ...)`, run on a compute node): `split.json`,
`parity.json` (the 17 library programs on the 149), `recovery.json`, `trajectory_<variant>.json` (one
row per proposal from Shinka's database), `elites.json` (≤ 50), `test_scores.json` (pre-registered
top-3 per variant on T), `rollouts.json` + `example_session.json`, a latent-export summary.

| id | slug | inputs | upstream | tags |
|---|---|---|---|---|
| r1 | `evaluator-parity`: Hattori, CTT and Bari as programs reproduce the per-mouse MLE numbers, and under the session level land near study 13's HB numbers; the 17 published models as a leaderboard on the 149, in both regimes | `parity.json`, `split.json`; study 13 `r1_hattori_pooled.json`, `r5_ctt_pooled.json` | `{01, r8}`, `{10, r1}`, `{13, r1}`, `{13, r5}` | `[cognitive]`, `[methods]` |
| r2 | `program-recovery`: recovery of a known program on simulated mice | `recovery.json` | none | `[cognitive/mevolve]`, `[methods]` |
| r3 | `search-curve`: best validation likelihood vs proposals and USD for `base` and `seeded-ctt` | `trajectory_*.json` | none | `[cognitive/mevolve]`, `[methods]` |
| r4 | `heldout-gap-closed`: gap closed on T against HB-CTT, the ladder (per-mouse and session-level rungs beside study 13's), splits by stage and task variant, paired per mouse, fingerprints and generative match, the example-session figure, the fit–complexity curve | `test_scores.json`, `rollouts.json`, `example_session.json`, `split.json`, `elites.json`; registry sources `rl_baseline.json`, `report_data.json`, `nxd_scaling.json`, `generative_match.json`; study 13 `r5_ctt_pooled.json`, `r6_ctt_ladder.json` | `{01, r8}`, `{01, r1}`, `{01, r9}`, `{13, r5}`, `{13, r6}` | `[cognitive/mevolve]`, `[comparison, generative-run, interpretation]` |

Makefile: `all: r1 r2 r3 r4` offline; `freeze-*` never in `all`; `.gitignore` adds `analysis/_cache_*/`,
`*.sqlite`; `environment.lock` at the first freeze; README notes that launches are not dispatcher
launches.

### 5.5 Issue and PR sequence
Issue on project 184 (In progress, P1, XL decomposed): "Study 40: LLM-driven discovery of
interpretable cognitive programs vs the cognitive baselines and the GRU on held-out mice". Child
issues: (S) scaffold + tag + explainer; (S) `write_initial()`; (S) registry HB entries after #63; (L,
mevolve) evaluator + seed library + Shinka adapter; (M) variant manifests and records; (M)
freezes + reports; (S) schema follow-up. PRs: Conventional Commits, `Closes #n`, `make validate`
green, Copilot review on code and rules PRs, threads resolved, merge commit only, Han merges each
after his own go-ahead.

### 5.6 Compute logistics
- Env on a compute node: `conda create -p .../envs/mevolve python=3.12` + `pip install -r
  requirements.lock`; never touch `dynamic-foraging-bfm-cpu`.
- Pilot: one sbatch job on a 112-core node (`-c 112 --mem=256G -t 7-00:00:00`, `--signal=B:USR1@900`)
  running `mevolve launch`; chain with `sbatch --dependency=afterany:<jobid> ... --resume`, each
  link a `resubmit` record. Web UI through an SSH tunnel.
- Smoke tests through `srun --ntasks=1 -p aind`; then a real 100-proposal `smoke` launch
  (`usd_cap 20`) that replaces every estimate with measurements.
- Secrets: `ANTHROPIC_API_KEY` in `~/.config/mevolve/anthropic_api_key` (mode 600), exported inside the sbatch
  script, never in a repo or log; the `NeuralDynamics` W&B key in `~/.config/mevolve/wandb_api_key`
  (mode 600, directory 700), exported as `WANDB_API_KEY` in mevolve's sbatch scripts and freezes
  only, leaving `~/.netrc` (the personal key) for the rest of the stack; backstop = a
  workspace spend limit in the Claude Console; one model per launch, no server-side fallbacks.
- Budget, 10k-proposal Sonnet 5.5 pilot (to be replaced by the smoke's numbers): input with caching ≈
  $80, output incl. thinking ≈ $250, meta-recommendations ≈ $15 → **≈ $350 (range $330–490)**;
  evaluator 330–1,000 core-hours; wall time 2–4 days.

## 6. First deliverable: animated explainer and study schematic (before any implementation)

**House precedent.** Study 13's explainer `studies/13-hb-subject-vs-gru-heldout/analysis/explainers/study13_animated_explainer.html`
(PR #71, open, all CI checks green): one self-contained HTML file, a 960 × 460 canvas, background
`#14171f` with the 3Blue1Brown palette (blue `#58c4dd`, red `#fc6255`, yellow `#ffd866`, green
`#83c167`, purple `#9a72ac`), scene buttons, Play / Replay / Prev / Next, a progress bar, serif
captions with yellow key terms. Study 40 reuses that engine and look.

**Files** (built in a scratch folder, committed with the scaffold PR once Han approves):
`studies/40-cognitive-llm-discovery/analysis/explainers/study40_animated_explainer.html` (about
2.5–3 minutes, six scenes, no network, no external fonts or libraries) and
`study40_schematic.html` (static map, archify skill, dark theme: data freeze → three tiers → seed
library → evaluator → gates → Shinka search with Claude → simplify → finalize on the test mice once →
reports r1–r4, Han's decision points as diamonds, the parking lot as a side box).

| # | scene | about | what moves on screen (right = blue, left = red) |
|---|---|---|---|
| 1 | A mouse at work | 20 s | two lick ports; a toy session drawn trial by trial: choices as ticks (right above, left below), rewards filled, block reward probabilities as a stepped line; 614 training mice, 149 held-out mice |
| 2 | A model is a small program | 25 s | a 10-line program types itself in; its latent variables (two values) as bars updating each trial; its p(right) over the session; parameters as dials |
| 3 | Fit on even, score on odd | 35 s | one mouse's sessions as tiles, coloured by training stage; on even tiles the dials settle to slightly different values per session around the mouse's typical values; on an odd tile a cloud of candidate dial settings is reweighted as its trials arrive while a log-probability counter accumulates; exp(mean log p); a number line from chance through MLE CTT and HB-CTT to the GRU, the remaining gap highlighted |
| 4 | One round of the search | 45 s | four islands of program cards; parent and inspiration glow; the prompt assembles (cached task card in grey, parent + feedback + inspiration in colour); Claude writes a child with the changed lines highlighted; a quick screen kills a broken child; a full fit scores the survivor; it joins the island and the island re-ranks; migration; every tenth generation a short "what worked" note is added; best-so-far curve and a dollar counter |
| 5 | Three groups of mice and the gates | 25 s | mice split into generation (feedback to Claude), validation (the score), test (149, locked); gates light in order, then the pilot |
| 6 | What comes out | 25 s | the best program shrinks along a fit-versus-complexity curve; it lands on the number line ("gap closed"); real and simulated switching curves overlap; latent trajectories are exported for the later physiology joins |

Scenes 1–5 use toy data. The reference values in scenes 3 and 6 (chance, MLE CTT, HB-CTT, GRU) are
read from the committed JSON (study 01 `rl_baseline.json` via `cross_study.rl_baseline`, study 13
`r5_ctt_pooled.json`, study 01 `report_data.json`) by a small build step that fills the template;
the footnote names sources and dates. Checks before sending: frames rendered with the cached
headless Chromium (`~/.cache/ms-playwright/chromium-1134`) at three time points per scene, no console
errors, legible text, `prefers-reduced-motion` honoured, archify's own validation for the schematic.
Then both files go to Han (`SendUserFile`, render) and are revised until he approves the storyline.

## 7. Verification gates (in order; nothing downstream starts before the gate passes)
0. **Data parity.** The split built through the wrapper's loader gives 149 test mice, 2,138 scored
   sessions and 1,013,172 scored trials; ids equal the keys of study 01's `rl_baseline.json` and
   study 10's reference.
1a. **Dynamics parity.** With the committed DE parameters (W&B runs `unhmbrk4` / `lmg1i9yd`,
   `heldout_test/subject_fit_metrics.csv`), the Hattori and CTT programs reproduce the numpy foragers'
   per-trial probabilities to 1e-6 and the pooled 0.7127 / 0.7170 within 1e-4 (oracle = numpy
   forager with the CTT 1e-12 clip, not the HB JAX code).
1b. **Fitter parity.** The per-mouse gradient fitter on the 149 reaches the DE pooled likelihood
   within 0.0005 for both seeds; otherwise L-BFGS polish or more restarts.
1c. **Session-level parity.** The Hattori and CTT seeds fitted with the session level and scored by
   the cheap session marginal land near study 13's HB-Hattori 0.7213 and HB-CTT 0.7225 (tolerance set
   in ADR 0003; study 13 used NUTS draws and an importance-sampling rescue, so an exact match is not
   expected), with per-session effective sample sizes reported, at a cost compatible with the search.
   If it cannot be met, the search falls back to per-mouse fitting with session facts and split
   diagnostics (decision 7).
2. **Timing and determinism.** Stage A and B times for the seeds and three heavy programs; identical
   text gives identical score vectors; 24 concurrent evaluations within 1.5× single-evaluation time;
   the memory limit spares healthy evaluations. Target Stage B ≤ 3 min.
3. **Engine smoke (≈ $7).** First assertion: `mevolve launch` imports Shinka's runner
   (`shinka/core/async_runner.py`) in-process, so the shim applies (shelling out to `shinka_run` would
   bypass it), and the system prompt goes out as content blocks carrying `cache_control`. Then
   (i) Shinka's own example with `claude-sonnet-5-5` through the shim, ≈ $2:
   no API errors, cache-read tokens > 0 after the first call, Shinka's cost within 5 % of the Console,
   the web UI shows the genealogy; (ii) 20 proposals on our evaluator ≤ $5: ≥ 10 valid, feedback
   visible in the prompts, resume after a forced kill without duplicate proposals. Failure → custom
   loop on the same evaluate contract (§10).
4. **Program recovery (≤ $20).** Mice simulated from the Hattori seed in `CoupledBlockTask`, with
   session parameters spread around each mouse's typical values: the session-level fitter recovers the
   typical values and spreads, and a 1-hour search seeded from CTT reaches within 0.002 of the
   generating program's validation score.
Then the **pilot** (`usd_cap 450`, reserve 50; stop at budget or 2,000 proposals without
improvement); the pre-registered top-3 per variant scored once on T.

## 8. Timeline (effort, not calendar)
| step | what | effort |
|---|---|---|
| −1 | explainer and schematic (§6), rendered and checked, revised until Han approves | 1–2 days |
| 0 | grill session (ADRs 0002, 0003, 0005); go-ahead to create `aind-dynamic-foraging-mevolve` and add it to project 184; issue + child issues; scaffold study 40 with the explainer (claims 40); env; API key | 1–2 days |
| 1 | data freeze through the wrapper loader, split lists, three seed programs, DE parameters | 3–4 days |
| 2 | evaluator: per-mouse and session-level fits, session-marginal scorer, session facts, split diagnostics; gates 1b, 1c, 2 | 6–8 days |
| 3 | seed library (14 more programs with parity tests) | 2–3 days |
| 4 | Shinka integration: pinned install, shim + tests, `initial.py` / `evaluate.py` adapter, `mevolve launch`, engine smoke | 2–3 days |
| 5 | `write_initial()` PR, recovery variants, 100-proposal smoke launch, r1 + r2 | 2–3 days |
| 6 | `base` and `seeded-ctt` pilots (≤ $500 total), r3 | 2–4 days wall per pilot |
| 7 | finalize: T once, stage 2, rollouts, latent exports, r4; decide the larger run | 3–4 days |

About five weeks of engineering to the first pilot result. Decision points: after step −1, after
step 0, after gate 1c (session level or fallback), after step 5 (go for the pilot), after step 7.

## 9. Risks
- Selecting among thousands of programs on V overfits V slightly: T is touched once with a
  pre-registered top-3, and the number of programs tested is reported.
- The GRU's 149 mice differ from the MLE/HB 149 by three each way: paired analyses use the 146 shared
  mice; pooled numbers state their cohort.
- Shinka is someone else's moving code: pinned SHA, shim tests in CI, upgrades only on purpose; its
  dollar cap is trusted only after gate 3 matches the Console, with the Console spend limit as backstop.
- The cheap session marginal is noisy for sessions far from the mouse's typical values (weights
  collapse onto few draws; study 13 needed 6,000 × 24 draws and an importance-sampling rescue for 256
  of 2,138 Hattori sessions): common random draws across programs keep rankings fair, effective sample
  sizes are reported, and gate 1c bounds the error before any search.
- Fitting a two-level model by its joint maximum tends to understate the spread between sessions: the
  fixed hyperprior on log σ and the gate-1c comparison with study 13 check it.
- 28 % of sessions have no recorded stage: an explicit `unknown` level, never dropped silently.
- Snapshot session count disagreement (23,868 vs 23,585): pin to the manifest (`e5fd51da…`).
- The simulator uses default block schedules: generative comparisons are curriculum-matched, not
  session-matched, and say so.
- Autowater trials are not filtered by the loader: parity holds, but the physiology interpretation
  must account for it.
- Near-equivalent programs flood the archive: AST cache, score-vector hashes, fingerprints in the
  feedback; watch the equivalence rate.
- `RLIMIT_AS` would kill healthy XLA evaluations: `RLIMIT_DATA` or the cgroup (gate 2).
- Creating `aind-dynamic-foraging-mevolve` under the org is outward-facing: Han's go-ahead.
- Whether Beaker reaches the Anthropic API is unverified; this study runs on HPC CPU only.

## 10. Parking lot (later rounds; one line each)
- **Final-stage discovery variant:** search on final-stage sessions of one task family, then test on
  all sessions (comparators re-scored on the subset).
- **A latent variable that persists across sessions:** a slow program variable carried from one day to the next,
  to model learning across training.
- **HMM seed and three-way variant:** a two-state belief-filter seed (parity against the GLM-HMM
  forward filter); ignore as a third class with the 3-way GRU (0.6273 H128) and GLM-HMM K = 3 (0.6148)
  as comparators; recovery on study 15's hurdle K = 3 truth.
- **Dynamic parameters:** Han's particle-filter draft (models PR #56) as a fitter-level rung for elites.
- **Teacher-guided residual feedback** and **distillation to the GRU** (teacher predictions on G/V are
  per-session held-out but per-mouse in-sample).
- **Harness-level research** on a Shinka fork or a custom loop: Pareto complexity archive, tempered
  islands (EvoDiverse), tree search (MCTS-AHD); Shinka's system-prompt evolution; a local embedding
  model for the novelty filter.
- **Heterogeneity ladder report** (population → mouse → session → within-session latent variables and states) and
  parameter relations to metadata for the elites.
- **Exact HB estimator for the headline:** re-score the pre-registered top 3 on T with study 13's
  NUTS pipeline (~30 CPU-hours each as a 149-task array; needs a NumPyro adapter for a program).
- **Graduation:** reviewed elites ported into `aind-dynamic-foraging-models` as numpy foragers.
- Timing inputs (reaction time, licks); the remaining ~500 pool mice for larger G and V.

## 11. References
- CogFunSearch: https://proceedings.mlr.press/v267/castro25a.html ; https://www.biorxiv.org/content/10.1101/2025.02.05.636732v1
- DataDIVER: https://pmc.ncbi.nlm.nih.gov/articles/PMC13228651/ (bioRxiv 10.64898/2026.05.18.725921)
- ShinkaEvolve: https://github.com/SakanaAI/ShinkaEvolve (arXiv 2509.19349); source files read: `shinka/core/config.py`, `shinka/llm/providers/anthropic.py`, `shinka/llm/kwargs.py`, `shinka/llm/providers/pricing.py`, `shinka/launch/slurm.py`, `shinka/prompts/prompts_base.py`, `shinka/webui/`
- GeCCo: https://arxiv.org/abs/2502.00879 ; ModelSMC: https://arxiv.org/abs/2602.18266 ; MCTS-AHD: https://arxiv.org/abs/2501.08603 ; EvoDiverse: https://arxiv.org/abs/2606.10587
- FunSearch code: https://github.com/google-deepmind/funsearch ; OpenEvolve: https://github.com/codelion/openevolve
- Claude API notes: prices and 5.x constraints from the claude-api skill (cached 2026-10-06)
- Lab GLM-HMM: models repo `glm_hmm/` (merged); study 14 results on branches `feat/48-r1-round1`, `feat/70-r2-state-spread`; designs in models #84 and studies #50
