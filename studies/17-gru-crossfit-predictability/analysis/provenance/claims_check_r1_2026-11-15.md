# Claims check — r1-cv-predictability, 2026-11-15

| field | value |
|---|---|
| report | `studies/17-gru-crossfit-predictability/analysis/reports/r1-cv-predictability.md` |
| version checked | commit `2ac2ec7` (report digest `20e8b67aad15e70ad2a6b3322ba1fe69f559bc950f0d8dd3ed3ce1b5066f9316`) |
| inputs digest | `8f8d4eec38e07c859e19bc71bfc6d6c817a848cd1d84334509e3f14da6784377` (44 files, `current/` and `upstream/` roles) |
| rubric | `claims-check/1 @ c234bff89422af05fc784786431fb4dd5a63c620` (skills pack 0.16.1, `posthoc-reporting/references/claims-check-rubric.md`) |
| reviewer | a new general-purpose agent (not a fork), model opus, effort high, read-only tools |
| fixes | commit `586c1e8`; PR comment https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies/pull/151#issuecomment-6907666690 |

The reviewer's output describes the report; it never feeds the frontmatter, INDEX or README.

## 1. Prompt given to the reviewer (verbatim)

```text
You are the claims check for a study report. Your instructions are the rubric file below; read it first and follow it exactly. Use read-only tools only (Read, Grep, Glob, and read-only Bash such as cat, sha256sum, python3 -c to resolve JSON pointers or compute digests). Do not edit, create, commit or run producers.

Rubric: /home/han.hou/.claude/plugins/cache/aind-behavior-foundation-model/aind-behavior-foundation-model-skills/0.16.1/skills/posthoc-reporting/references/claims-check-rubric.md
rubric_version to record: `claims-check/1 @ c234bff89422af05fc784786431fb4dd5a63c620` (skills pack 0.16.1).

Repository root (git worktree): /home/han.hou/code/aind-dynamic-foraging-bfm-studies/.claude/worktrees/bridge-cse_01XAC3mCG9zmJCgJVYKaNFCP
Study: studies/17-gru-crossfit-predictability (call it S below).

Files to check (all paths under the repository root):
- The report: S/analysis/reports/r1-cv-predictability.md (frontmatter claims F1–F4 with JSON pointers into S/analysis/r1_cv_predictability.json; resolve each).
- Its producer, which writes the methods, result-1..4 and supp-1..4 blocks and chooses the verdict words: S/analysis/r1_cv_predictability.py, output S/analysis/r1_cv_predictability.json.
- Inputs that producer reads: S/analysis/scores.json, S/analysis/cv_headline.json, S/analysis/cv_ignore_split.json, S/analysis/cv_2way_vs_s01.json, S/analysis/cv_fold_spread.json, S/analysis/cv_crosscheck.json, S/analysis/cv_lambda.json, S/analysis/cv_rebuild_check.json, S/analysis/cv_schematic.json, S/reference/roles.json, S/reference/folds.json; frozen inputs S/analysis/ignore_split.json, S/analysis/rebuild_check.json.
- Embedded figures (SVG, text readable) and the producer that draws each: S/analysis/fig_cv_schematic.svg (cv_schematic.py), fig_cv_headline.svg and fig_cv_map.svg (cv_headline.py), fig_cv_ignore_split.svg (cv_ignore_split.py, freeze_ignore_split.py), fig_cv_2way_vs_s01.svg (cv_2way_vs_s01.py), fig_cv_fold_spread.svg (cv_fold_spread.py), fig_cv_crosscheck.svg (cv_crosscheck.py), fig_cv_lambda.svg (cv_lambda.py), fig_cv_rebuild_check.svg (cv_rebuild_check.py, freeze_rebuild_check.py); freeze of the scores: S/analysis/freeze_scoring.py.
- Upstream (named in the report's `upstream` and read by producers): studies/02-gru-scaling-law-ignore/analysis/reports/r4-three-way-engagement.md and studies/02-gru-scaling-law-ignore/analysis/frozen_runs.json; studies/01-gru-scaling-law/analysis/reports/r7-nxd-joint-scaling-grid.md, studies/01-gru-scaling-law/analysis/nxd_scaling.json and nxd_runs.json.
- Study context: S/README.md, S/PLAN.md, S/CHANGELOG.md, the glossary CONTEXT.md at the repository root.
- There is no previous live version of this report (it is a first draft), so C8 has no diff to check; say so.

Note on structure, so you do not flag it as a defect: the report keeps the 8 required sections and adds a `## Supplementary` section (S1–S4: validation and housekeeping, blocks supp-1..4) between Results and What it means; the skeleton rules allow extra `##` sections there.

Return exactly the rubric's JSON (compute report_digest and inputs_digest as it specifies), then the short markdown summary. Quote line numbers and exact text for every finding.
```

## 2. Reviewer's response (verbatim)

Line numbers refer to the report at `2ac2ec7`.

```json
{
  "report": "studies/17-gru-crossfit-predictability/analysis/reports/r1-cv-predictability.md",
  "report_digest": "20e8b67aad15e70ad2a6b3322ba1fe69f559bc950f0d8dd3ed3ce1b5066f9316",
  "inputs_digest": "8f8d4eec38e07c859e19bc71bfc6d6c817a848cd1d84334509e3f14da6784377",
  "rubric_version": "claims-check/1 @ c234bff89422af05fc784786431fb4dd5a63c620",
  "model": "claude-opus-5-5 (effort level not set explicitly by the caller)",
  "claims": [
    {"id": "F1", "verdict": "supported",
     "evidence_pointer": "analysis/r1_cv_predictability.json#/f1/gain_full/mean = 0.012250 (sem 0.000729, n_folds 5); #/f1/gain_full_per_fold all > 0 (0.0107 to 0.0140); #/f1/verdict = 'rises in every fold'; Fig. 1b",
     "note": "+0.0123 ± 0.0007 matches 0.0122505 ± 0.000729. The SEM over 5 folds keeps the interval far above 0. Every fold also rises at every step of D (cv_headline.json#/per_fold), so the heading on line 107 holds too. The test fold's 'full' is D = 614 and the pool folds' is 609–610, which the report labels ≈610. See the missing caveat on subset rules."},
    {"id": "F2", "verdict": "supported",
     "evidence_pointer": "analysis/r1_cv_predictability.json#/f2/passA_passB_r = 0.96053; #/f2/per_mouse/p5 = 0.62327, /p95 = 0.82473; #/f2/n_mice = 763; Fig. 2a title 'r = 0.96'",
     "note": "The numbers match. The split-half r shows that a mouse's score is stable across its two interleaved halves of sessions. The wording 'a property of the mouse' says more than that, because the same per-mouse value would also arise from conditions that stay fixed within a mouse (see missing_caveats)."},
    {"id": "F3", "verdict": "supported",
     "evidence_pointer": "analysis/r1_cv_predictability.json#/f3/first_after_ignore/share_of_loglik_gain = 1.1789; /f3/first_after_ignore/diff_mean = 0.03369; /share_of_trials = 0.04452; /f3/diff_mean = 0.001622, /diff_sem = 0.000106, /n_folds_3way_higher = 5; Fig. 3a title '+0.0016 ± 0.0001 (5 of 5 folds higher)'",
     "note": "All the numbers match, and the bin shares sum to about 100%. The 2nd–4th bin also gains (+0.0015 ± 0.00018, 5 of 5 folds), but 'sits on the first' holds at 118%. The test fold in F3 and Fig. 3 uses 146 of the 149 mice; this is not stated (see estimand_issues)."},
    {"id": "F4", "verdict": "supported",
     "evidence_pointer": "analysis/r1_cv_predictability.json#/f4/verdict = 'within which-mice sampling'; /f4/real_sd = 0.002861; /f4/perm_sd_median = 0.006136; /f4/p_sd_ge_real = 0.9526; /f4/test_minus_all = -0.002448; Fig. 5",
     "note": "The value at the pointer and the numbers match. The verdict comes from code (p >= 0.05) and is not a fixed string. The real SD (0.0029) sits near the lower 2.5% bound of the null (0.0024): the real folds are more alike than random splits. The null is unstratified, but the real pool folds are stratified (see missing_caveats). The '0.0024 below' number has no interval (see estimand_issues)."},
    {"id": "C8", "verdict": "cant_verify",
     "evidence_pointer": "git log of the report: e3130ec, 2ac2ec7; both have status draft",
     "note": "There is no earlier live version, so C8 has no diff to check. Nothing to report."},
    {"id": "C9-fixed-strings", "verdict": "supported",
     "evidence_pointer": "r1_cv_predictability.py lines 180, 196, 209-210",
     "note": "The verdict words for F1, F3 and F4 come from code. Three conclusions in the supplementary blocks are fixed strings that cannot change with the data: line 189 'Study 17's scoring reproduces study 02', line 221 'A fine-tuned held-out model rebuilt from W&B alone reproduces its scoring run', and line 270 'cancelled because λ cannot change those (their session weights are zero; Fig. S2)'. The current data agree with all three: max |Δ| is 4.7e-05; 2,334,957 of 2,334,957 trials match with max diff 1.5e-07; and all 10 missing jobs are for λ = 0 checkpoints in Table S4. None of them contradicts its table."}
  ],
  "stale_sentences": [
    {"line": 91, "text": "The test fold is the freeze's 149 test mice; its models are study 02's (`nxd-3way`, H256) and study 01's (2-way).",
     "why": "This describes the plan before 2026-11-13, when the CHANGELOG added '1 test-fold model' at D=300. Table S4 line 265 lists test-3way-d300 from `NeuralDynamics/dfbfm_17-gru-crossfit-predictability/scaling-folds-20261113-015420-118d9c82`, a study 17 run (README variants table: 'plus the test fold at D=300')."},
    {"line": 303, "text": "The test fold's models are study 02's and study 01's runs; the pool folds' were trained for this study with the same recipe",
     "why": "Table S4 (line 265) and the README contradict this: one test-fold model (D=300) was trained in study 17, on a different platform and code version from study 02's. This is the very fold effect the caveat discusses."}
  ],
  "unsourced_numbers": [],
  "missing_caveats": [
    {"claim": "F1",
     "why": "PLAN.md lines 109-115 ('Caveat: two subset rules on one curve'): the test fold's D=10/30/100/300 subsets come from the wrapper's subject_ratio sampling, and the pool folds' subsets come from freeze_scaling_subsets.py. The two rules pick different mice, and a small curriculum can get 1 mouse under one rule and 0 under the other. The PLAN says 'The report states this'. The r1 Caveats section does not."},
    {"claim": "F4",
     "why": "reference/folds.json#/strata = 'tier x curriculum x session-count third, dealt round-robin into 4 pool folds'. The real folds are therefore balanced on factors that track predictability, while the permutation null (cv_fold_spread.py lines 50-55) draws unstratified random splits. This alone makes the real SD tend to fall below the null, which fits P = 0.95 and a real SD (0.0029) near the null's 2.5th percentile (0.0024). So the test has little power to show a model or platform fold effect. The caveat on lines 304-306 ('F4 finds none larger than mouse sampling') does not mention this."},
    {"claim": "F2",
     "why": "The split-half r = 0.96 shows that a mouse's score is stable between its fit and score sessions. PLAN.md lines 75-78 says these sessions alternate by date, so the two halves are interleaved in time. Folds.json#/balance records that mice differ in curriculum (Coupled Baiting / Uncoupled Baiting / Uncoupled Without Baiting) and in session count. A per-mouse value that is stable across interleaved sessions is also what task conditions fixed within a mouse would produce. The Caveats section does not limit 'a property of the mouse' (line 67, line 125) to this."}
  ],
  "estimand_issues": [
    {"line": 146, "text": "**Fig. 3. Where the 3-way GRU beats the 2-way GRU.** Full-size models, both passes, scored engaged trials.",
     "why": "In Fig. 3, the table on lines 148-156 and F3, the test fold covers only the 146 mice that both models score (freeze_ignore_split.py does an inner join; scores.json has no 2-way scores for 722683, 795395, 820243). As a result, the test fold's 3-way value in Fig. 3a is 0.72965 (cv_ignore_split.json#/per_fold/test/lik_3way), while Fig. 1 and Fig. 5 give 0.73034 for the same model on 149 mice. The F3 text and the Fig. 3 caption do not state the 146 / 760-mouse restriction. Only the F2 table on line 138 shows n = 760."},
    {"line": 279, "text": "so the level differences are about which mice a fold holds, not about the models.",
     "why": "This is a 'no model effect' conclusion from a permutation test that only failed to detect a larger spread. Caveat lines 305-306 say F4 'cannot rule out a small one', and the null is unstratified (see missing_caveats F4)."},
    {"line": 69, "text": "the original test fold scores 0.0024 below all 763 mice",
     "why": "This number (f4/test_minus_all = -0.00245) has no interval or test. It is also used on lines 296-297 ('Its absolute level is slightly below the all-mice value, so an absolute \"GRU predictability\" is better quoted as the 5-fold mean'). F4's own null (median fold SD 0.0061) suggests that an offset of 0.0024 is within which-mice sampling."},
    {"line": 282, "text": "The spread across mice (Fig. 2c) is several times the whole gain from 10 to about 610 training mice (F1).",
     "why": "This compares two different estimands: a between-mouse spread of per-mouse trial-pooled values (SD 0.062, or 5th–95th range 0.20; f2/per_mouse) and a fold-mean shift in trial-pooled likelihood (0.0123; f1/gain_full). 'Several times' is about 5× using the SD and about 16× using the range, and the sentence does not say which. 'Most of the variation' on the same line does not name the quantity that varies."},
    {"line": 205, "text": "Models whose best_eval checkpoint has a session-curriculum λ below 1, scored at their own λ (the study's rule) and at λ = 1",
     "why": "Table S4 lists 10 models with checkpoint λ < 1 (8 at 0, 2 at 0.5). Fig. S2 and its table show 5. The other 5 (pool2-3way-d10, pool2-3way-d30, pool3-3way-d10, pool4-3way-d10, pool4-3way-d30) were not scored at λ = 1 (scores.json#/missing). The heading on line 200 ('hardly matters') rests on n = 2 models where λ can matter (λ = 0.5; max |Δ| 0.00019), and the heading does not give n."}
  ]
}
```

Reviewer's summary:

> All four claims are supported: every number matches its pointer and its figure. There are 2 stale
> sentences, 3 missing caveats and 5 estimand issues to answer. No numbers are unsourced.
>
> **Not checked from these files**
> - Line 291, "This fits the earlier reading that mice change strategy after an ignore (for example, a
>   reset of action values)": no source is given. Study 02 r4 has no such reading, and no other
>   markdown file under `studies/` has it.
> - Line 304, "Beaker H200" for every non-Cedar pool-fold model: PLAN gives H200 only for the
>   full-size models. D=10/30/300 ran on "Beaker tier 2", and the GPU type there is not stated.

## 3. Responses (what shipped in `586c1e8`)

| # | Check | Finding | Response |
|---|---|---|---|
| 1 | C4 | L91/L303: test fold's models are not only study 02/01 runs; D = 300 was trained here | Fixed: Methods and the model-origin caveat say so |
| 2 | C5 (F1) | PLAN promises a caveat on the two subset rules; missing | Fixed: new caveat "Two rules picked the training subsets" |
| 3 | C5 (F4) | Pool folds are stratified; the permutation null was not | Fixed: froze each pool mouse's stratum (`reference/pool_strata.json`, rebuilt as `freeze_folds.py` dealt them, checked against `folds.json` balance) and added a within-strata re-deal null (Fig. 5c): P = 0.81, so F4 holds |
| 4 | C5 (F2) | Split-half r shows stability within a mouse, not that the mouse causes it | Fixed: F2 now says "stable within a mouse"; new caveat on within-mouse fixed conditions |
| 5 | C6 | Fig. 3 / F3 use 760 mice (test fold 146 of 149), unstated | Fixed: F3 claim and Fig. 3 caption state it (counts from code) |
| 6 | C7 | "not about the models" from a non-rejecting test | Fixed: "most simply explained by which mice … a small model or platform effect is not ruled out" |
| 7 | C7 | Test-fold offset had no interval, yet drove a recommendation | Fixed: interval from random 149-mouse sets (P = 0.69). The "slightly hard test split" reading is dropped; the 5-fold-mean advice now rests on precision only |
| 8 | C6 | "several times the whole gain" mixed estimands | Fixed: "SD of per-mouse likelihood is about 5 times the gain"; the ratio is stored in the report JSON (`f2/sd_over_full_gain`) |
| 9 | C6 | Fig. S2 covers 5 of the 10 λ < 1 models; heading rested on n = 2 | Fixed: caption names the 5 and why; heading gives the λ = 0.5 result and n = 2 |
| 10 | C9 | Fixed-string conclusions in S1/S3 captions and the S4 note | Fixed: neutral captions; the S4 reason is chosen by code from the missing jobs' checkpoint λ |
| 11 | can't verify | "earlier reading … reset of action values" unsourced | Reworded as the lab's working hypothesis |
| 12 | can't verify | "Beaker H200" for all non-Cedar pool models | Fixed: "full size on H200" |

Not yet re-run: the claims check has not been repeated on `586c1e8`. The rubric keys a check by report
and input digests, so the fixed version needs its own run before the report goes live.
