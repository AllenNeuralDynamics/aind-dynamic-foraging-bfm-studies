---
schema_version: 1
report: analysis/reports/r1-cv-predictability.md
checked_on: '2026-11-18'
content_digest: 73f710cb4448992523ef2b825446af0a69dcc619de274df44edf3f17b1518321
inputs:
  studies/17-gru-crossfit-predictability/analysis/cv_2way_vs_s01.json: 55e987f797d76429a119a7b24cf7640eb42177a3bf6e37866933aba6e07a2931
  studies/17-gru-crossfit-predictability/analysis/cv_crosscheck.json: 1f915e8618636d6aa2b552a729c3239123c14992048a76e7f2bec10483a0f4ad
  studies/17-gru-crossfit-predictability/analysis/cv_fold_spread.json: a3f674aa2e6658504982c48946b8db37e34ceb03a05c35001fbe9cf3821fcde6
  studies/17-gru-crossfit-predictability/analysis/cv_headline.json: e2042214d8b43b07e5d48fbed800d4a678eee55f63bbd286ed9b2d1688fb0ae9
  studies/17-gru-crossfit-predictability/analysis/cv_ignore_split.json: 6c96e26f6512776949a0e49b511ab3c233f97884e8c8ff03e382a1700b241f93
  studies/17-gru-crossfit-predictability/analysis/fig_cv_2way_vs_s01.svg: 6aa789d9e3e8ab17dc57b129dfdb8b387c665f48be7677ebd38b5eb985a415f5
  studies/17-gru-crossfit-predictability/analysis/fig_cv_crosscheck.svg: 69b32c50b0ec0ebfce7dc318290d16b050dc83eb6c8317c3739ece72d10aafd7
  studies/17-gru-crossfit-predictability/analysis/fig_cv_fold_spread.svg: 986e4067f09e00c093596f49656500ba9f235f52b7aa377091fe7579e7e3b34f
  studies/17-gru-crossfit-predictability/analysis/fig_cv_headline.svg: 65c54afbff077fc9f94449cc01e28a3cbdadf40f32468d2c059cbcd1fd7c2a25
  studies/17-gru-crossfit-predictability/analysis/fig_cv_ignore_split.svg: b1e59d0942d29b34345ddf89996b99004648cbf1082636f3402bdfa70cb635c5
  studies/17-gru-crossfit-predictability/analysis/fig_cv_map.svg: 700d458526412b15234fbb8a981d4deeca413c8d091af1c80ea5d75044b8882b
  studies/17-gru-crossfit-predictability/analysis/gru_map_index.json: 79f042681ae3ad68261eceb31993b77cb2993e0b508f21ae3839b6e806d4f0c4
  studies/17-gru-crossfit-predictability/analysis/ignore_split.json: 8ed0978fab63276f219c9c34c8680eb515bb1eed7999cad9ff095b16ac82cce7
  studies/17-gru-crossfit-predictability/analysis/r1_cv_predictability.json: a267b24c112386e984479b993922d628d9aab7058737cecd0fb32da9111e4b41
  studies/17-gru-crossfit-predictability/analysis/scores.json: 213665f01e16969b18f8bab743d675906589ad829089cc426ae103c9888cfce6
  studies/17-gru-crossfit-predictability/reference/pool_strata.json: acbea92605dec6702544cca3bbf6a0c2187df3d7a897768bbd757b163abf8d87
  studies/17-gru-crossfit-predictability/reference/roles.csv: 9ba14be8b154d9ed7fececac2dc8f04f9360480ce4bab28b3c675e819b46e198
context:
  studies/17-gru-crossfit-predictability/PLAN.md: 36eaff45e2271bbc4e7d50498de37ab20f1f2b8559a6a06e305db5055b036959
code:
  studies/17-gru-crossfit-predictability/analysis/cv_fold_spread.py: 2709d176f3ec73a20b20bcae34f171f069a0e52221d09e73b0cd52ddbfba996c
  studies/17-gru-crossfit-predictability/analysis/cv_headline.py: 6892777cf7048173059b64114c5e2c2b7772551591e5627ae450d25ae4bc81c5
  studies/17-gru-crossfit-predictability/analysis/cv_ignore_split.py: 6443a74c505a151f707813854ad73a64ffc0664a3ce61a16dfaf9d2f5e1f7bb9
  studies/17-gru-crossfit-predictability/analysis/freeze_folds.py: 225c37ba983dd9e684ac9e8019ef4df13a4b81df650bcd5ae71d53c52652b038
  studies/17-gru-crossfit-predictability/analysis/freeze_ignore_split.py: adc1de543c55cf5b49747c4f0604b6cf27a71de7835d99f365a526f93b2b2637
  studies/17-gru-crossfit-predictability/analysis/freeze_scoring.py: ab431c62bb5ca2bceea4446744bc57741fb1622b48c8f4692da42feb0264b1be
  studies/17-gru-crossfit-predictability/analysis/r1_cv_predictability.py: 200cd9ce5dbaf0892769550eabaefc4a89742580be7a9f0f0cae4e38b153dc0c
  studies/util/references.py: f115fc413cf9d3087c3ad673558a6ffcf8864d840f176ca780002c7395963882
rubric_version: claims-check/3 @ 31fc2ddda2a15d7cb4aa5f5ca8fd710361c1d9e1
model: claude-opus-5-5, effort high (general-purpose agent, read-only)
prompt_sha256: 108113a8dc0234b2616dbb2d250d3df1b877b52642e76651a19e9cdc6ac75366
response_sha256: 20807bbe8f3ebe198342051c6b66f99b8dd351ee5abb129cc606ab58b5639664
severity_counts:
  High: 0
  Medium: 0
  Low: 7
---
# Claims check — r1, 2026-11-18

The record describes the report; it never feeds the frontmatter, INDEX or README (ADR 0003).

## Prompt given to the reviewer

~~~~text
You are the claims check for a study report. Your instructions are the rubric file below; read it first and follow it exactly.

Rubric: /home/han.hou/.claude/plugins/cache/aind-behavior-foundation-model/aind-behavior-foundation-model-skills/0.17.1/skills/posthoc-reporting/references/claims-check-rubric.md
rubric_version to record: `claims-check/3 @ 31fc2ddda2a15d7cb4aa5f5ca8fd710361c1d9e1`

Read-only checkout pinned to the report's commit (a detached git worktree; do not fetch, do not read any other ref or commit):
  /home/han.hou/code/wt/s17-cc5-ro    (studies commit 33ecc0a1160d7d53c8691eb8cc306fa259be5d36)
Read only files in this tree. Use read-only tools (Read, Grep, Glob, and read-only Bash such as cat, sha256sum, python3 -c to resolve JSON pointers or compute digests). Do not edit, create or commit anything, and do not run producers.

The one file you may write: /allen/aind/scratch/han.hou/s17_scoring/claims_check/run5_response.md
Write the rubric's JSON and then the markdown summary there, and nothing anywhere else. Then reply with only that path and one line of counts (claims by verdict; findings by severity).

Starting points (paths relative to the checkout root):
- The report: studies/17-gru-crossfit-predictability/analysis/reports/r1-cv-predictability.md (status: live).
- Its last live version is commit 0cdd9ef; the diff `git diff 0cdd9ef..33ecc0a -- <report>` is empty (the report is unchanged since it went live). This run re-checks the live report under rubric claims-check/3; C8 has no changed claim to check.
- Claim values, identical at 0cdd9ef and 33ecc0a:
  F1 analysis/r1_cv_predictability.json#/f1/gain_full/mean = 0.01225049837582226
  F2 analysis/r1_cv_predictability.json#/f2/passA_passB_r = 0.9605310526137287
  F3 analysis/r1_cv_predictability.json#/f3/first_after_ignore/share_of_loglik_gain = 1.1789019465534636
  F4 analysis/r1_cv_predictability.json#/f4/verdict = "within which-mice sampling"
Follow every reference the report makes yourself: its JSON, figures, producers and the helpers they import, frozen inputs, upstream reports and source JSONs.

Not evidence (per the rubric's guards): earlier claims-check records under studies/17-gru-crossfit-predictability/analysis/provenance/claims_check_*.md, commit messages, PR threads, git log text, notes. The study CHANGELOG.md is a list of claims to check, not evidence.

Note on structure, so you do not flag it as a defect: the report keeps the 8 required sections and adds a `## Supplementary` section (S1–S4: validation and housekeeping) between Results and What it means; the skeleton rules allow extra `##` sections there.
~~~~

## Reviewer's response

~~~~text
```json
{
  "report": "studies/17-gru-crossfit-predictability/analysis/reports/r1-cv-predictability.md",
  "report_digest": "45d2c68f17f59254adaea33dcb1dc1fa1d57e34991da399b96b093c2012ba2d1",
  "inputs_digest": "da917104f7fe8188c68e1210a74d965ae00189a005942a683466d6e2d3322eee",
  "rubric_version": "claims-check/3 @ 31fc2ddda2a15d7cb4aa5f5ca8fd710361c1d9e1",
  "model": "claude-opus-5-5, effort as launched by the writing session (not visible to the reviewer)",
  "commit": "33ecc0a1160d7d53c8691eb8cc306fa259be5d36",
  "claims": [
    {"id": "F1", "verdict": "supported",
     "evidence_pointer": "analysis/r1_cv_predictability.json#/f1/gain_full (mean 0.012250, sem 0.000729, n_folds 5); #/f1/gain_full_per_fold (test 0.01399, pool_1 0.01177, pool_2 0.01079, pool_3 0.01074, pool_4 0.01397); #/f1/n_folds_full_gain_positive = 5; fig_cv_headline.svg panel b",
     "note": "Line 26/65: '+0.0123 ± 0.0007 ... (mean ± SEM, n = 5 folds)' matches 0.012250 / 0.000729 to the printed digits. 'in all 5 folds': all 5 per-fold gains are > 0. 'within each fold's own mice': the gain is paired within fold (cv_headline.py:163-170). Each fold also rises monotonically at every D step (cv_2way_vs_s01.json#/study17_3way/*/per_fold), so lines 281-282 ('predicted better as D grows') hold too. Recomputed from scores.json in pure Python (full minus D = 10 per fold, both passes): mean 0.0122505, SEM 0.0007289, identical. The result-1 table agrees (0.0123 ± 0.00073; verdict 'rises in every fold' computed at r1_cv_predictability.py:74). C2 for Fig. 1 rests on provenance, not on a rendered image: the SVG's text carries no data value (axis ticks only), and its sha256 matches cv_headline.json _meta.outputs, whose body gives the same numbers. Single seed per fold and two subset rules are stated in Caveats (lines 311-321).",
     "item": "C1", "severity": null},
    {"id": "F2", "verdict": "supported",
     "evidence_pointer": "analysis/r1_cv_predictability.json#/f2/passA_passB_r = 0.96053; #/f2/per_mouse/p5 = 0.62327, /p95 = 0.82473; #/f2/n_mice = 763; reference/roles.csv; fig_cv_map.svg panel a title 'r = 0.96'",
     "note": "Line 29/66: 0.623 / 0.825 and r = 0.96 match the JSON. Recomputed from scores.json in pure Python: 763 unique mice, p5 0.6232661, p95 0.8247300, SD 0.0621424, pass A vs pass B r 0.9605311 (identical). 'two interleaved halves': reference/roles.csv alternates fit/score by session date for all 763 mice (0 mice with two equal roles in a row). The 'not caused by the mouse' limit is in Caveats (lines 322-324).",
     "item": "C1", "severity": null},
    {"id": "F3", "verdict": "supported",
     "evidence_pointer": "analysis/r1_cv_predictability.json#/f3/first_after_ignore (share_of_loglik_gain 1.17890, share_of_trials 0.044521, diff_mean 0.033687, diff_sem 0.000368); #/f3/diff_mean 0.0016224, /diff_sem 0.0001058, /n_folds_3way_higher 5, /n_mice 760; cv_ignore_split.json#/per_bin/1/diff_per_fold; fig_cv_ignore_split.svg",
     "note": "Line 32/67: +0.0016 ± 0.0001, 'all 5 folds' (per-fold diffs 0.00134 to 0.00186, all > 0), +0.034, 4.5% and 118% all match. k = 1 differences are 0.0328 to 0.0349 in every fold; 20+ and 'before any ignore' bins are negative in every fold, so 'sits on the first engaged trial' holds (2nd-4th carries +8%, also stated in the table). 760 mice: ignore_split.json per-fold trial counts equal the 2-way model's n_engaged sums in scores.json, and its pool-fold 3-way likelihoods equal scores.json exactly; test fold 146 + 154 + 154 + 153 + 153 = 760. Recomputed from ignore_split.json in pure Python: k = 1 share 1.1789019, share of trials 0.0445208, n = 10,761,776 trials, identical. Fig. 3 agrees by its SVG text (title '+0.0016 ± 0.0001 (5 of 5 folds higher)', panel c axis to 1.2) and its digest in cv_ignore_split.json _meta.outputs. C9 on line 160 ('difference +8.3e−05; different mouse count'): a fixed string (r1_cv_predictability.py:172-174), but it states a fact the files show, not an attributed cause: study 17's value covers 146 mice (scores.json test-2way-full-s0), study 01's covers its 149 held-out mice (studies/util/references.py _S01_GRU mouse_set). No finding.",
     "item": "C1", "severity": null},
    {"id": "F4", "verdict": "supported",
     "evidence_pointer": "analysis/r1_cv_predictability.json#/f4 (real_sd 0.002861, perm_sd_median 0.006136, p_sd_ge_real 0.9526, stratified_pool/p_sd_ge_real 0.8128, test_minus_all -0.002448, test_offset_p 0.6858, verdict 'within which-mice sampling'); cv_fold_spread.json; fig_cv_fold_spread.svg",
     "note": "Line 35/68: every number matches to the printed digits. The verdict word is computed (r1_cv_predictability.py:101-103: 'within' when both P >= 0.05), not a fixed string. Fail-to-reject reading, checked in three places: (1) the claim (line 68) places the observed spread inside the null; it does not assert zero fold or model effect, and the real SD is below both null medians (0.0029 vs 0.0061 and 0.0046); (2) the heading (line 163) 'differ no more than random sets of mice would' is literally true of the observed SDs; (3) the bold sentence at line 279 cites '(F1, F4)'; its 'not an artefact' part rests on the F1 gain in every fold, and lines 283-285 say a small model or platform effect is not ruled out. Recomputed from scores.json: 5-fold SD 0.0028613, 4-pool SD 0.0029083, test minus all mice -0.0024478, identical. The null medians, 95% ranges and P values were read from cv_fold_spread.json, not rerun (seeded permutations; producers were not run). Fig. 5 agrees by its SVG text ('fold SD = 0.0029', 'median 0.0061; P(random ≥ real) = 0.95', 'median 0.0046; P(re-deal ≥ real) = 0.81') and its digest. The limited power is stated (lines 302-304, 316-317). The stratified null covers only the 4 pool folds, as the claim says.",
     "item": "C1", "severity": null},
    {"id": "S4 (line 274; not a frontmatter claim)", "verdict": "n/a", "evidence_pointer": "analysis/r1_cv_predictability.json#/supp/lambda/0.0 (n_models 3, max_abs_diff 0.0); Table S4 (8 checkpoints with λ = 0); r1_cv_predictability.py:226-229",
     "note": "Line 274: 'Not scored: 10 λ = 1 jobs (cancelled), of checkpoints with λ = 0: at λ = 0 the session weights are zero, so λ cannot change the score (Fig. S2).' The explanatory clause is chosen at r1_cv_predictability.py:229 only from which jobs are missing (all of λ = 0 checkpoints), not from the measured differences, so it would print even if the λ = 0 rows of Fig. S2 differed. The data agree with it today: all 3 λ = 0 models scored both ways differ by exactly 0.0 (#/supp/lambda/0.0); the other 5 of the 8 λ = 0 checkpoints were not scored at λ = 1, so for them the clause rests on the mechanism (PLAN.md line 63: the session-delta output layer is exactly zero at a λ = 0 checkpoint). Low: consistent with its data; the risk is only that it cannot change with them.",
     "item": "C9", "severity": "Low"}
  ],
  "producer_traces": [
    {"claim": "F1", "producer": "analysis/r1_cv_predictability.py:72 <- analysis/cv_headline.py:163-176 (per-fold full minus D=10, mean and SEM ddof=1 over 5 folds) <- cv_headline.py:49-60 (fold_value: exp(sum ll_lr / sum n_engaged) over both passes) <- analysis/freeze_scoring.py:73",
     "matches": "true", "note": "Estimand as named: trial-pooled held-out likelihood, L/R on engaged trials (scores.json columns ll_lr_sum, n_engaged), both passes pooled, 3-way seed 0, paired within fold; n = 5 folds, unit = fold.",
     "item": "C0", "severity": null},
    {"claim": "F2", "producer": "analysis/r1_cv_predictability.py:81 <- analysis/cv_headline.py:84-87, 101 (np.corrcoef of per-mouse pass A vs pass B values, full-size 3-way seed 0, all 5 folds pooled)",
     "matches": "true", "note": "Per-mouse trial-pooled likelihood per pass; n = 763 mice, each predicted by the fold model that held it out. Percentiles from r1_cv_predictability.py:52-60, 77-80 (both passes pooled per mouse).",
     "item": "C0", "severity": null},
    {"claim": "F3", "producer": "analysis/r1_cv_predictability.py:89 <- analysis/cv_ignore_split.py:52-61 (share = bin's summed 3-way minus 2-way ll_lr over all folds / total) <- analysis/freeze_ignore_split.py:33-37, 49-59 (k = trials since last ignore; scored engaged trials joined on session and trial across both models)",
     "matches": "true", "note": "The per-trial inputs (trial_predictions.csv.gz) live on /allen scratch, outside the tree, so the binning cannot be rerun here. The in-tree tie: ignore_split.json fold totals reproduce scores.json (pool-fold 3-way and 2-way likelihoods identical; trial counts equal the 2-way n_engaged sums; test fold restricted to the 146 mice the 2-way model scores). Quantity matches the claim: share of the summed log-likelihood difference, k = 1 bin, 760 mice, both passes.",
     "item": "C0", "severity": null},
    {"claim": "F4", "producer": "analysis/r1_cv_predictability.py:101-103 <- analysis/cv_fold_spread.py:50-61 (unstratified permutation of mice into folds of real sizes, 5,000 draws), 64-67 (random 149-mouse offsets), 76-83, 92-94 (re-deal of 614 pool mice within 20 strata, as analysis/freeze_folds.py:65-74 dealt them)",
     "matches": "true", "note": "Fold values are full-size 3-way seed 0, both passes, trial-pooled; each mouse keeps its own fold model's score in the nulls. The stratified re-deal mirrors freeze_folds.py (sorted strata, shuffled, round-robin with the counter carried across strata).",
     "item": "C0", "severity": null}
  ],
  "stale_sentences": [
    {"line": 346, "text": "1. Register the per-mouse predictability map (per-trial, per-session, per-mouse) as the shared reference for study 41 and MEvolve scientist mode.", "why": "The tree already registers it: studies/util/references.py has gru_cv_map_3way.s17_pass_a and gru_cv_map_2way.s17_pass_a (report r1, analysis/gru_map_index.json, which lists the per-trial artifacts and each mouse's fold). The Next item reads as not yet done. Outside the claims.", "item": "C4", "severity": "Low"}
  ],
  "unsourced_numbers": [
    {"line": 303, "text": "That range is about as wide as the whole scaling gain (F1)", "why": "Sourced (r1_cv_predictability.json#/f4/test_offset_ci95 = [-0.0118, +0.0116]; #/f1/gain_full/mean = 0.0123), but the comparison does not match its sources as worded: the range's width is 0.0234, about 2x the gain; only each half-width (about 0.0117) is about the gain. The sentence's conclusion (the check cannot rule out an offset of the gain's size) still holds. Low: prose outside the claims, and the reading of F4 does not change.", "item": "C3", "severity": "Low"}
  ],
  "missing_caveats": [
    {"claim": "none (Caveats, lines 325-326)", "why": "Line 325-326: 'Pass B scores lower than pass A for the same models (Fig. S1d)' cites only the 6 test-fold models (+0.0030; cv_crosscheck.json#/summary/passA_minus_passB_lr_mean). scores.json shows pass A above pass B in all 32 models, by +0.0023 to +0.0088 (largest in pool folds 3-4, e.g. pool4-3way-d30 +0.0088, pool4-3way-full +0.0081), up to about 0.7x the full scaling gain. The caveat exists; its stated scope and size are narrower than the data. Low (moved from Medium): no claim depends on the gap. F1 is paired within fold with both passes pooled, F3 compares models on the same trials and passes, F4 pools both passes per fold, and for F2's r the per-fold gaps of the full-size 3-way models (+0.0023 to +0.0081) are small against the per-mouse SD of 0.062 (#/f2/per_mouse/sd).", "item": "C5", "severity": "Low"}
  ],
  "estimand_issues": [
    {"line": 334, "text": "The two offsets are of similar size and opposite sign (pass A scores above pass B, Fig. S1d; the test fold sits below the all-mice value, F4)", "why": "The sentence compares pass A minus pass B (+0.0030, test-fold 3-way models, S1) with the 3-way test fold minus all mice (-0.0024, F4), but Fig. 4's gap is about 2-way models and pass-A-only vs both-passes pooled. For the 2-way test model, pass A alone minus both passes is +0.0014 (cv_2way_vs_s01.json#/study17_test_2way_full_passA 0.72918 vs #/study17_2way_full/per_fold/test 0.72778), and the 2-way test fold sits 0.0032 below the 5-fold 2-way mean (0.73101). Opposite sign holds; 'similar size' is looser than stated. Low (moved from Medium): the caveat's conclusion (the vertical gap is not a clean comparison) holds with either set of numbers, and Fig. 4 carries no claim.", "item": "C6", "severity": "Low"},
    {"line": 67, "text": "+0.034 there, where 4.5% of trials carry 118% of the summed log-likelihood difference", "why": "The claim gives +0.034 without its SEM (the table, line 150, has ± 0.00037, 5 folds), and the 4.5% and 118% shares carry no interval and no n of trials in the claim or the table (n = 10,761,776 trials appears only in the Fig. 3c title). Missing: interval type for the shares; n for the shares in the text. Low (moved from Medium): the per-fold k = 1 differences (cv_ignore_split.json#/per_bin/1/diff_per_fold, 0.0328 to 0.0349, all 5 folds) show the location does not depend on the missing interval.", "item": "C7", "severity": "Low"},
    {"line": 66, "text": "the two interleaved halves of each mouse's sessions agree (Pearson r = 0.96)", "why": "r has n (763 mice, line 66 and line 137) and unit, but no interval type. Low (moved from Medium): with n = 763 and r = 0.96 an interval would not change the reading.", "item": "C7", "severity": "Low"}
  ]
}
```

# Claims check: r1-cv-predictability (studies 33ecc0a, rubric claims-check/3)

The report is unchanged since its last live version (`git diff 0cdd9ef..33ecc0a -- <report>` is empty), so C8 has no changed claim to check. All four claim values trace to producer lines. Recomputed in pure Python from committed files: F1 (gain per fold, from `scores.json`), F2 (percentiles and r, from `scores.json`), F3 (k = 1 share, from `ignore_split.json`) and F4's real SDs and test offset (from `scores.json`); all identical. F4's null medians and P values were read from `cv_fold_spread.json`, not rerun. Every input and figure digest in the producers' `_meta` matches the tree. All 9 result-block digests match `r1_cv_predictability.json` `_meta.outputs`. Figures were checked by their SVG text and by digest, not as rendered images.

**Claims:** F1 supported, F2 supported, F3 supported, F4 supported.
**Findings:** High 0, Medium 0, Low 7.

## C0: producer traces
- F1: `r1_cv_predictability.py:72` <- `cv_headline.py:163-176`, `49-60`. Matches.
- F2: `r1_cv_predictability.py:81` <- `cv_headline.py:84-87, 101`. Matches.
- F3: `r1_cv_predictability.py:89` <- `cv_ignore_split.py:52-61` <- `freeze_ignore_split.py:33-37, 49-59`. Matches. The per-trial inputs are on /allen scratch, outside the tree; the fold totals tie to `scores.json`.
- F4: `r1_cv_predictability.py:101-103` <- `cv_fold_spread.py:50-67, 76-94`. Matches.

## C1/C2/C9: claims
- F1, F2, F3, F4: supported. The figures agree (by SVG text and digest; Fig. 1 by digest only, since its text holds no data value). Every verdict word is computed from the data.
- Low (C9), line 274: "at λ = 0 the session weights are zero, so λ cannot change the score". The clause is chosen from which jobs are missing (`r1_cv_predictability.py:229`), not from measured differences. It agrees with the data today: 3 of 3 λ = 0 models measured differ by exactly 0 (`#/supp/lambda/0.0`); the other 5 λ = 0 checkpoints rest on the mechanism.
- No finding (C9), line 160: "different mouse count" is a fixed string, but it states a fact (146 vs 149 mice), not a cause.

## C3: unsourced numbers
- Low, line 303: "That range is about as wide as the whole scaling gain (F1)". The range width is 0.0234, about 2x the gain (0.0123); only the half-width matches. The conclusion holds.

## C4: stale sentences
- Low, line 346: Next item 1, "Register the per-mouse predictability map ...". `studies/util/references.py` already registers `gru_cv_map_3way.s17_pass_a` / `gru_cv_map_2way.s17_pass_a` from this report.

## C5: missing caveats
- Low, Caveats lines 325-326 (no claim depends on it): the pass B < pass A caveat cites only the test fold (+0.0030). `scores.json` shows the gap in all 32 models, +0.0023 to +0.0088, largest in pool folds 3-4. No claim depends on it.

## C6/C7: estimands and uncertainty
- Low (C6), line 334: "The two offsets are of similar size and opposite sign". It compares 3-way offsets (+0.0030 A-B; -0.0024 test-all) for a 2-way figure. The relevant 2-way offsets are +0.0014 (pass A vs both passes) and -0.0032 (test vs 5-fold mean). The conclusion holds.
- Low (C7), line 67: "+0.034 there, where 4.5% of trials carry 118% ...". There is no SEM for +0.034 in the claim, and no interval or trial n for the shares. The per-fold k = 1 differences (all 5 folds 0.033-0.035) back the location.
- Low (C7), line 66: "Pearson r = 0.96". It has n = 763 but no interval type.

## C8
- Not applicable: the diff against the last live version (0cdd9ef) is empty, and the four claim values are as supplied.

## Inputs digest
`inputs_digest` = sha256 of `json.dumps(map, sort_keys=True, separators=(",", ":"))` over this `role/path -> sha256` (raw file bytes) map. The report itself is excluded.

| role/path | sha256 |
|---|---|
| `current/studies/17-gru-crossfit-predictability/analysis/cv_2way_vs_s01.json` | `16d1944cce524df569344b2ef1d020aadf52fc2a8d5547a3947d972ef842e603` |
| `current/studies/17-gru-crossfit-predictability/analysis/cv_crosscheck.json` | `6709cbc71a96515e47152f26e1688957a615c0b716fb4d9e789cfd93c42398e1` |
| `current/studies/17-gru-crossfit-predictability/analysis/cv_fold_spread.json` | `137001717a7975b71f154e43ca04fad23fc549225900cf65ff8717f3c12af022` |
| `current/studies/17-gru-crossfit-predictability/analysis/cv_fold_spread.py` | `2709d176f3ec73a20b20bcae34f171f069a0e52221d09e73b0cd52ddbfba996c` |
| `current/studies/17-gru-crossfit-predictability/analysis/cv_headline.json` | `a6b823f106c0051cb5f9b82a357c333e134d80e2a0f96277a56aca4622815db5` |
| `current/studies/17-gru-crossfit-predictability/analysis/cv_headline.py` | `6892777cf7048173059b64114c5e2c2b7772551591e5627ae450d25ae4bc81c5` |
| `current/studies/17-gru-crossfit-predictability/analysis/cv_ignore_split.json` | `aaf2ff10f7c13732f4ef738c5f7513968a5bf7a6bae234bce5fb99d65a46b864` |
| `current/studies/17-gru-crossfit-predictability/analysis/cv_ignore_split.py` | `6443a74c505a151f707813854ad73a64ffc0664a3ce61a16dfaf9d2f5e1f7bb9` |
| `current/studies/17-gru-crossfit-predictability/analysis/fig_cv_2way_vs_s01.svg` | `6aa789d9e3e8ab17dc57b129dfdb8b387c665f48be7677ebd38b5eb985a415f5` |
| `current/studies/17-gru-crossfit-predictability/analysis/fig_cv_crosscheck.svg` | `69b32c50b0ec0ebfce7dc318290d16b050dc83eb6c8317c3739ece72d10aafd7` |
| `current/studies/17-gru-crossfit-predictability/analysis/fig_cv_fold_spread.svg` | `986e4067f09e00c093596f49656500ba9f235f52b7aa377091fe7579e7e3b34f` |
| `current/studies/17-gru-crossfit-predictability/analysis/fig_cv_headline.svg` | `65c54afbff077fc9f94449cc01e28a3cbdadf40f32468d2c059cbcd1fd7c2a25` |
| `current/studies/17-gru-crossfit-predictability/analysis/fig_cv_ignore_split.svg` | `b1e59d0942d29b34345ddf89996b99004648cbf1082636f3402bdfa70cb635c5` |
| `current/studies/17-gru-crossfit-predictability/analysis/fig_cv_map.svg` | `700d458526412b15234fbb8a981d4deeca413c8d091af1c80ea5d75044b8882b` |
| `current/studies/17-gru-crossfit-predictability/analysis/freeze_folds.py` | `225c37ba983dd9e684ac9e8019ef4df13a4b81df650bcd5ae71d53c52652b038` |
| `current/studies/17-gru-crossfit-predictability/analysis/freeze_ignore_split.py` | `adc1de543c55cf5b49747c4f0604b6cf27a71de7835d99f365a526f93b2b2637` |
| `current/studies/17-gru-crossfit-predictability/analysis/freeze_scoring.py` | `ab431c62bb5ca2bceea4446744bc57741fb1622b48c8f4692da42feb0264b1be` |
| `current/studies/17-gru-crossfit-predictability/analysis/gru_map_index.json` | `12525d16e94c4ca430366e9796091cd966a02dcc0ac8508ee1d0e52775a850de` |
| `current/studies/17-gru-crossfit-predictability/analysis/ignore_split.json` | `311048d91e8ce94de3ec507fbb1ee208be79c53c478473095db52ab438839746` |
| `current/studies/17-gru-crossfit-predictability/analysis/r1_cv_predictability.json` | `ce2b2862f8f7d91671df669fd819cb0f31d3ee8ef432c08560960dafa45421e7` |
| `current/studies/17-gru-crossfit-predictability/analysis/r1_cv_predictability.py` | `200cd9ce5dbaf0892769550eabaefc4a89742580be7a9f0f0cae4e38b153dc0c` |
| `current/studies/17-gru-crossfit-predictability/analysis/scores.json` | `767767466ae5d51b3f17868c55ac49dc2e721b480c57fd511042e6bd55c8227b` |
| `current/studies/17-gru-crossfit-predictability/PLAN.md` | `36eaff45e2271bbc4e7d50498de37ab20f1f2b8559a6a06e305db5055b036959` |
| `current/studies/17-gru-crossfit-predictability/reference/pool_strata.json` | `4c695a0bff3152db5b164d1ace0d3eb8937d1faa0f1fe577e56a9cbf2634e757` |
| `current/studies/17-gru-crossfit-predictability/reference/roles.csv` | `9ba14be8b154d9ed7fececac2dc8f04f9360480ce4bab28b3c675e819b46e198` |
| `current/studies/util/references.py` | `f115fc413cf9d3087c3ad673558a6ffcf8864d840f176ca780002c7395963882` |
| `last_live/git diff 0cdd9ef..33ecc0a -- studies/17-gru-crossfit-predictability/analysis/reports/r1-cv-predictability.md` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty diff) |
~~~~

## Responses to the findings

Run 5 re-checks the live r1, unchanged since it went live (0cdd9ef), under rubric claims-check/3. Result: 4 claims supported, 4 producer traces match, 0 High, 0 Medium, 7 Low. Under the posthoc-reporting rule "When versions stop" (Low findings never open a revision; a report is done when its last check has no open High and no kept Medium), every finding is closed as won't-fix and the text stays as checked. The overlapping items in studies#163 (run 3/4 deferrals) are for the first author to close or keep.

| # | Check | Severity | Finding | Response |
|---|---|---|---|---|
| 1 | C9 (S4) | Low | Line 274: "at λ = 0 the session weights are zero, so λ cannot change the score" is chosen from which jobs are missing, not from measured differences | Won't fix: it matches every measured λ = 0 model (3 of 3 differ by exactly 0, `#/supp/lambda/0.0`) and follows from the mechanism; no claim depends on it |
| 2 | C3 | Low | Line 303: "about as wide as the whole scaling gain": the range is about 2× the gain; only its half-width matches | Won't fix: sourced (`#/f4/test_offset_ci95`, `#/f1/gain_full/mean`) and the conclusion holds; the word "width" vs "half-width" changes no claim |
| 3 | C4 | Low | Line 346: Next item 1, "Register the per-mouse predictability map", is already done | Won't fix: done after r1 went live (studies#166: `gru_cv_map_3way` / `gru_cv_map_2way`, recorded in the study CHANGELOG); editing Next would stale this record for a wording change |
| 4 | C5 (Caveats) | Low | Lines 325–326: the pass B < pass A caveat cites only the test fold (+0.0030); the gap holds in all 32 models, +0.0023 to +0.0088 | Won't fix: no claim depends on it; the per-model values are in `scores.json` |
| 5 | C6 | Low | Line 334: "two offsets of similar size and opposite sign" compares 3-way offsets for a 2-way figure (2-way: +0.0014 and −0.0032) | Won't fix: the sentence's conclusion (not a clean comparison) holds with either pair |
| 6 | C7 (F3) | Low | Line 67: "+0.034" has no SEM in the claim; the 4.5% and 118% shares carry no interval or trial n | Won't fix: the SEM (± 0.00037, 5 folds) is in the table, and the per-fold k = 1 differences (0.033 to 0.035, all 5 folds) back the location |
| 7 | C7 (F2) | Low | Line 66: Pearson r = 0.96 has n = 763 but no interval type | Won't fix: with n = 763 and r = 0.96 an interval would not change the reading (the reviewer moved it from Medium to Low for this reason) |
