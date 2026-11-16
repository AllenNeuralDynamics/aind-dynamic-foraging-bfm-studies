---
schema_version: 1
report: analysis/reports/r1-cv-predictability.md
checked_on: '2026-11-16'
content_digest: c5e634f5af1cb8b153343cfa249aee0ae7ad29eaba4c2d2cf4d399c35cf5d3b0
inputs:
  studies/17-gru-crossfit-predictability/analysis/r1_cv_predictability.py: 200cd9ce5dbaf0892769550eabaefc4a89742580be7a9f0f0cae4e38b153dc0c
  studies/17-gru-crossfit-predictability/analysis/r1_cv_predictability.json: a267b24c112386e984479b993922d628d9aab7058737cecd0fb32da9111e4b41
  studies/17-gru-crossfit-predictability/analysis/scores.json: 213665f01e16969b18f8bab743d675906589ad829089cc426ae103c9888cfce6
  studies/17-gru-crossfit-predictability/analysis/cv_headline.json: e2042214d8b43b07e5d48fbed800d4a678eee55f63bbd286ed9b2d1688fb0ae9
  studies/17-gru-crossfit-predictability/analysis/cv_ignore_split.json: 6c96e26f6512776949a0e49b511ab3c233f97884e8c8ff03e382a1700b241f93
  studies/17-gru-crossfit-predictability/analysis/cv_2way_vs_s01.json: 55e987f797d76429a119a7b24cf7640eb42177a3bf6e37866933aba6e07a2931
  studies/17-gru-crossfit-predictability/analysis/cv_fold_spread.json: a3f674aa2e6658504982c48946b8db37e34ceb03a05c35001fbe9cf3821fcde6
  studies/17-gru-crossfit-predictability/analysis/cv_crosscheck.json: 1f915e8618636d6aa2b552a729c3239123c14992048a76e7f2bec10483a0f4ad
  studies/17-gru-crossfit-predictability/analysis/cv_lambda.json: 6a7aa4f3f55fe3297d101abaaaa44ec86aef1ed648d2e44899f1485eb56de24b
  studies/17-gru-crossfit-predictability/analysis/cv_rebuild_check.json: fee79aa73bb9367b4de5c6d45ad6c7f03db61968245be4d33afadf869d367ba8
  studies/17-gru-crossfit-predictability/analysis/cv_schematic.json: 708f059e96f97e532a21cb4fc087eca59d95482cc0b8e1d3b6bc9dcc70ac6ce6
  studies/17-gru-crossfit-predictability/analysis/ignore_split.json: 8ed0978fab63276f219c9c34c8680eb515bb1eed7999cad9ff095b16ac82cce7
  studies/17-gru-crossfit-predictability/analysis/rebuild_check.json: 07170c6d590aa50468ad5a1cb6972ea7fdcf6fd47c4beced42e071250c3d2f4e
  studies/17-gru-crossfit-predictability/analysis/fig_cv_schematic.svg: 8286f4e13c82015a1f40f1aa2eecdea71ef4a74f442d7163a7dee0aa1c6229b3
  studies/17-gru-crossfit-predictability/analysis/fig_cv_headline.svg: 65c54afbff077fc9f94449cc01e28a3cbdadf40f32468d2c059cbcd1fd7c2a25
  studies/17-gru-crossfit-predictability/analysis/fig_cv_map.svg: 700d458526412b15234fbb8a981d4deeca413c8d091af1c80ea5d75044b8882b
  studies/17-gru-crossfit-predictability/analysis/fig_cv_ignore_split.svg: b1e59d0942d29b34345ddf89996b99004648cbf1082636f3402bdfa70cb635c5
  studies/17-gru-crossfit-predictability/analysis/fig_cv_2way_vs_s01.svg: 6aa789d9e3e8ab17dc57b129dfdb8b387c665f48be7677ebd38b5eb985a415f5
  studies/17-gru-crossfit-predictability/analysis/fig_cv_fold_spread.svg: 986e4067f09e00c093596f49656500ba9f235f52b7aa377091fe7579e7e3b34f
  studies/17-gru-crossfit-predictability/analysis/fig_cv_crosscheck.svg: 69b32c50b0ec0ebfce7dc318290d16b050dc83eb6c8317c3739ece72d10aafd7
  studies/17-gru-crossfit-predictability/analysis/fig_cv_lambda.svg: 714963306a7b6f865d4e57f6bce372c2414b6b992ad9b783087069d3e5b838e4
  studies/17-gru-crossfit-predictability/analysis/fig_cv_rebuild_check.svg: 401d412cdd66aa8b61210bcafc476ac9da5ccb950dc5180faecda9665d0f87dc
  studies/17-gru-crossfit-predictability/analysis/cv_schematic.py: c4c5dee0c7abdddd9f505215cb857fadd82eb2ff3f91dd00079b61fc025e235e
  studies/17-gru-crossfit-predictability/analysis/cv_headline.py: 6892777cf7048173059b64114c5e2c2b7772551591e5627ae450d25ae4bc81c5
  studies/17-gru-crossfit-predictability/analysis/cv_ignore_split.py: 6443a74c505a151f707813854ad73a64ffc0664a3ce61a16dfaf9d2f5e1f7bb9
  studies/17-gru-crossfit-predictability/analysis/freeze_ignore_split.py: adc1de543c55cf5b49747c4f0604b6cf27a71de7835d99f365a526f93b2b2637
  studies/17-gru-crossfit-predictability/analysis/cv_2way_vs_s01.py: 3756b9d3c13e68d2a4b83d51b8c29312bc8108c4083e839b2fc32ac30da75e67
  studies/17-gru-crossfit-predictability/analysis/cv_fold_spread.py: 2709d176f3ec73a20b20bcae34f171f069a0e52221d09e73b0cd52ddbfba996c
  studies/17-gru-crossfit-predictability/analysis/cv_crosscheck.py: 2b3e2b48622848d0a2590ffcc2c2797d150b05f91c992b5b095f8f4e535de9b5
  studies/17-gru-crossfit-predictability/analysis/cv_lambda.py: 1bb2414bae94aedf922ec814e74db132f2b4d8809641d595486bc96e411ec6f9
  studies/17-gru-crossfit-predictability/analysis/cv_rebuild_check.py: 98ab7fe18299d4d9aff099484bba427efa4ac980e0c81b7156355990194a060c
  studies/17-gru-crossfit-predictability/analysis/freeze_rebuild_check.py: a764ed60895c71230e760065df393c61a306b8269498338478dcaec2837a17aa
  studies/17-gru-crossfit-predictability/analysis/freeze_scoring.py: ab431c62bb5ca2bceea4446744bc57741fb1622b48c8f4692da42feb0264b1be
  studies/17-gru-crossfit-predictability/reference/roles.json: 8fdf246c50de7c50e4f6b93be003dcfb0b33b2aaac5b0588bbffcfa6badab74f
  studies/17-gru-crossfit-predictability/reference/folds.json: 716599d05950b6a3c73c0e4c06b131c4c3dcc02d75e0536819b51fa6726142e8
  studies/02-gru-scaling-law-ignore/analysis/reports/r4-three-way-engagement.md: 763bfe06cbeacede5dde3c451c27e0c94c8f55e0d13345be07525565182a56f0
  studies/02-gru-scaling-law-ignore/analysis/frozen_runs.json: 7680fab2b0b8950ddd3378e915f7867d71c0273d40c5c6e4323f0d7604484093
  studies/01-gru-scaling-law/analysis/reports/r7-nxd-joint-scaling-grid.md: 7e05c01310054987f55d313f1d56b505f14bd3b63e27aec6f1811e9e9c9f5d16
  studies/01-gru-scaling-law/analysis/nxd_scaling.json: 06addf8f7d0c023e1ba435e4fed2dd31fc9020116ea92518d36fbde84bcb2c11
  studies/01-gru-scaling-law/analysis/nxd_runs.json: 43685b2266333cf920384233a346ff086e214c0db867847b31d7e6f63e20868e
  studies/17-gru-crossfit-predictability/reference/pool_strata.json: acbea92605dec6702544cca3bbf6a0c2187df3d7a897768bbd757b163abf8d87
  studies/17-gru-crossfit-predictability/analysis/freeze_pool_strata.py: 2210246b0a8a60987094419c78fb0420b9f32510989116e49995e281348a1c9b
context:
  studies/17-gru-crossfit-predictability/README.md: 9e992d1d01ef3843cc2401822bd3145a53adf917687928b95ae20f310be6f494
  studies/17-gru-crossfit-predictability/PLAN.md: e1eac4170dfeaaf6b7931dd43fa030aae42f9a453b1e3337c643c31cc6efbaff
  studies/17-gru-crossfit-predictability/CHANGELOG.md: f42f56ad77288889b6c8894f04e65bebfeb46e86b5d129efb21171fcdc83bcf4
  CONTEXT.md: d0710daa76b08305c7d3d2eeefec8f63b90d3fe1b61bc791bc19c66e817388d3
rubric_version: claims-check/1 @ c234bff89422af05fc784786431fb4dd5a63c620
model: claude-opus-5-5, effort high (general-purpose agent, read-only)
prompt_sha256: 09a999c0c4c956ea142e30542e279903b0867ce5bcc8e0c6e30f1dba683ab677
response_sha256: 43253fe87b585dc840e1a4227c0406aa5f89e7b1395e4f180136b32d4f6faf44
---
# Claims check — r1, 2026-11-16

The record describes the report; it never feeds the frontmatter, INDEX or README (ADR 0003).

## Prompt given to the reviewer

~~~~text
You are the claims check for a study report. Your instructions are the rubric file below; read it first and follow it exactly. Use read-only tools only (Read, Grep, Glob, and read-only Bash such as cat, sha256sum, python3 -c to resolve JSON pointers or compute digests). Do not edit, create, commit or run producers.

Rubric: /home/han.hou/.claude/plugins/cache/aind-behavior-foundation-model/aind-behavior-foundation-model-skills/0.16.1/skills/posthoc-reporting/references/claims-check-rubric.md
rubric_version to record: `claims-check/1 @ c234bff89422af05fc784786431fb4dd5a63c620` (skills pack 0.16.1).

Repository root (git worktree): /home/han.hou/code/aind-dynamic-foraging-bfm-studies/.claude/worktrees/bridge-cse_01XAC3mCG9zmJCgJVYKaNFCP
Study: studies/17-gru-crossfit-predictability (call it S below).

Files to check (all paths under the repository root):
- The report: S/analysis/reports/r1-cv-predictability.md (frontmatter claims F1–F4 with JSON pointers into S/analysis/r1_cv_predictability.json; resolve each).
- Its producer, which writes the methods, result-1..4 and supp-1..4 blocks and chooses the verdict words: S/analysis/r1_cv_predictability.py, output S/analysis/r1_cv_predictability.json.
- Inputs that producer reads: S/analysis/scores.json, S/analysis/cv_headline.json, S/analysis/cv_ignore_split.json, S/analysis/cv_2way_vs_s01.json, S/analysis/cv_fold_spread.json, S/analysis/cv_crosscheck.json, S/analysis/cv_lambda.json, S/analysis/cv_rebuild_check.json, S/analysis/cv_schematic.json, S/reference/roles.json, S/reference/folds.json, S/reference/pool_strata.json; frozen inputs S/analysis/ignore_split.json, S/analysis/rebuild_check.json (and S/reference/pool_strata.json, written by S/analysis/freeze_pool_strata.py).
- Embedded figures (SVG, text readable) and the producer that draws each: S/analysis/fig_cv_schematic.svg (cv_schematic.py), fig_cv_headline.svg and fig_cv_map.svg (cv_headline.py), fig_cv_ignore_split.svg (cv_ignore_split.py, freeze_ignore_split.py), fig_cv_2way_vs_s01.svg (cv_2way_vs_s01.py), fig_cv_fold_spread.svg (cv_fold_spread.py, which also reads S/reference/pool_strata.json), fig_cv_crosscheck.svg (cv_crosscheck.py), fig_cv_lambda.svg (cv_lambda.py), fig_cv_rebuild_check.svg (cv_rebuild_check.py, freeze_rebuild_check.py); freeze of the scores: S/analysis/freeze_scoring.py.
- Upstream (named in the report's `upstream` and read by producers): studies/02-gru-scaling-law-ignore/analysis/reports/r4-three-way-engagement.md and studies/02-gru-scaling-law-ignore/analysis/frozen_runs.json; studies/01-gru-scaling-law/analysis/reports/r7-nxd-joint-scaling-grid.md, studies/01-gru-scaling-law/analysis/nxd_scaling.json and nxd_runs.json.
- Study context: S/README.md, S/PLAN.md, S/CHANGELOG.md, the glossary CONTEXT.md at the repository root.
- There is no previous live version of this report (it is a first draft), so C8 has no diff to check; say so.

Note on structure, so you do not flag it as a defect: the report keeps the 8 required sections and adds a `## Supplementary` section (S1–S4: validation and housekeeping, blocks supp-1..4) between Results and What it means; the skeleton rules allow extra `##` sections there.

Return exactly the rubric's JSON (compute report_digest and inputs_digest as it specifies), then the short markdown summary. Quote line numbers and exact text for every finding.
~~~~

## Reviewer's response

~~~~text
```json
{
  "report": "studies/17-gru-crossfit-predictability/analysis/reports/r1-cv-predictability.md",
  "report_digest": "920d954866c5a9e603e0e225c744e43e33b864b498853ccfac2abde4ddf4c4d2",
  "inputs_digest": "fdb1e2e90dbac79fe0b688b48fb0a4ed6ace0249cd30293784dac5b4a36abc25",
  "rubric_version": "claims-check/1 @ c234bff89422af05fc784786431fb4dd5a63c620",
  "model": "claude-opus-5-5, effort high",
  "claims": [
    {"id": "F1", "verdict": "supported",
     "evidence_pointer": "analysis/r1_cv_predictability.json#/f1/gain_full/mean",
     "note": "Pointer = 0.012250 (sem 0.000729, n_folds 5), printed as +0.0123 ± 0.0007. f1/gain_full_per_fold is positive in all 5 folds (0.0107 to 0.0140), n_folds_full_gain_positive = 5. Fig. 1a/b and the result-1 table agree. C9 note: the verdict code (r1_cv_predictability.py line 74) checks only full > D=10. The heading at line 108, 'rises with training mice in every fold', implies a rise at every step. cv_headline.json per_fold shows every fold does rise at every step (10<30<100<300<full), so the heading is true now, but the verdict would still print if a middle step fell."},
    {"id": "F2", "verdict": "supported",
     "evidence_pointer": "analysis/r1_cv_predictability.json#/f2/passA_passB_r",
     "note": "Pointer = 0.9605 (printed 0.96), n = 763 mice (f2/n_mice). f2/per_mouse p5 = 0.6233 and p95 = 0.8247 match '0.623 to 0.825'. Fig. 2a title 'r = 0.96' agrees. The interleaving caveat (line 323) is present."},
    {"id": "F3", "verdict": "supported",
     "evidence_pointer": "analysis/r1_cv_predictability.json#/f3/first_after_ignore/share_of_loglik_gain",
     "note": "Pointer = 1.1789 (printed 118%). f3/diff_mean = 0.001622 ± 0.000106 (n_folds 5), n_folds_3way_higher = 5, n_mice = 760. first_after_ignore diff_mean = 0.03369 and share_of_trials = 0.0445 match '+0.034' and '4.5%'. Fig. 3 and its table agree. C9 note: the Fig. 3 caption 'Where the 3-way GRU beats the 2-way GRU' (line 147) is a fixed string (producer line 155) and would print even if the verdict flipped. It is consistent with the data now."},
    {"id": "F4", "verdict": "supported",
     "evidence_pointer": "analysis/r1_cv_predictability.json#/f4/verdict",
     "note": "Pointer = 'within which-mice sampling'. This needs both P >= 0.05: p_sd_ge_real = 0.9526 and stratified_pool.p_sd_ge_real = 0.8128. real_sd = 0.00286, perm median = 0.00614, test_minus_all = -0.00245, test_offset_p = 0.6858. All match the claim and Fig. 5. This is a 'no excess' claim, but the real SD sits below the null median, so it is not an inconclusive interval read as no effect. The limited power is stated (lines 304-305, 317-318). C8: there is no previous live version of this report, so there is no diff to check."}
  ],
  "stale_sentences": [
    {"line": 358, "text": "Training groups: the study README's variants table.",
     "why": "The report sentence is correct, but the table it points to is stale. README line 47 still lists `test-fold-ref` as 'planned' with no W&B group, and line 51 lists `scoring` as 'running (HPC CPU)'. Line 90 links PLAN.md, whose status line 5-7 says 'scoring not started'. These are stale lines in the context files, not in the report body."}
  ],
  "unsourced_numbers": [
    {"line": 316, "text": "(Beaker: full size on H200; two D = 10 models on Cedar A100)",
     "why": "Not in any cited JSON. It traces only to README line 50 and PLAN lines 105-107 (D=10 pool folds 1-2 on Cedar), which are study prose, not a JSON or a registry reference."}
  ],
  "missing_caveats": [
    {"claim": "F1",
     "why": "Table S4 (lines 243-265) shows that every D = 10 model and 3 of 5 D = 30 models use a best_eval checkpoint at λ = 0, with no session conditioning, at step 10k-40k. The other 2 D = 30 models are at λ = 0.5. Every D >= 100 model is at λ = 1, mostly at step 90k. So the D = 10 -> full gain also spans a change in checkpoint type, a result of the best_eval rule. PLAN lines 53-54 promise that the report 'says which points come from a checkpoint without session conditioning'. Table S4 lists this, but no caveat says what it means for reading F1 as a training-mice effect."},
    {"claim": "F3",
     "why": "In the pool folds, the 3-way and 2-way models differ only in ignore_policy (README line 49, PLAN line 120). In the test fold, the 2-way model is study 01's nxd-grid run (Table S4 line 266) and the 3-way model is study 02's. Study 01's run trained on a different mouse list: it includes 3 freeze test mice and lacks 3 pool mice (PLAN lines 36-37, 123-126). So the test fold's 3-way - 2-way difference is not ignore-policy alone. The caveat at lines 314-318 covers fold effects for F1/F4 only, not this pairing."},
    {"claim": "Caveat lines 326-327 / Fig. 4 caveat lines 334-336",
     "why": "The caveat cites pass A > pass B only for the 6 test-fold models in Fig. S1d (+0.0030). scores.json shows pass A > pass B in all 32 models, and the size depends on the fold: about +0.0023-0.0029 in pool_2, +0.0024-0.0041 in test and +0.0080-0.0088 in pool_4. The Fig. 4 caveat uses the test-fold size as if it were general."}
  ],
  "estimand_issues": [
    {"line": 289, "text": "The SD of per-mouse held-out likelihood (Fig. 2c) is about 5 times the whole gain from 10 to about 610 training mice",
     "why": "This compares a between-subject SD of per-subject held-out likelihood (f2/per_mouse/sd = 0.0621) with the 5-fold mean of trial-pooled gains (f1/gain_full = 0.0123). These are two different aggregations (glossary: per subject vs trial-pooled). The ratio 5.07 is sourced (f2/sd_over_full_gain), but the two quantities are not the same kind."},
    {"line": 303, "text": "That range is about as wide as the whole scaling gain (F1)",
     "why": "The 95% range is -0.0118 to +0.0116 (f4/test_offset_ci95). Its width is 0.0234, about 1.9 times the gain of 0.0123. Only its half-width (about 0.0117) matches the gain. The sentence does not say which width it means."},
    {"line": 334, "text": "The two offsets are of similar size and opposite sign (pass A scores above pass B, Fig. S1d; the test fold sits below the all-mice value, F4)",
     "why": "Pass A - pass B (+0.0030, S1) is not the offset of a pass-A-only value from the both-pass value. From cv_headline.json that offset is +0.0012 for the 3-way test-full model (0.73158 vs 0.73034). From cv_2way_vs_s01.json it is +0.0014 for the 2-way test model (0.72918 vs 0.72778). The mouse-set offset is -0.0024 for 3-way (F4) and -0.0032 for 2-way (test 0.72778 vs 5-fold mean 0.73101). So the pass offset is about half the mouse-set offset, not 'similar size'."},
    {"line": 333, "text": "only its 149 test mice (study 17: both passes, 763 mice)",
     "why": "This sentence is about the 2-way models. Study 17's 2-way models cover 760 mice (f2/n_mice_2way = 760; test fold 146 of 149), not 763."},
    {"line": 147, "text": "share of the 3-way − 2-way log-likelihood difference summed over all folds and trials",
     "why": "In fig_cv_ignore_split.svg, the panel c legend says 'share of 3-way − 2-way log-likelihood gain' and the panel b title says 'Gain by trials since the last ignore'. The caption and table (lines 147, 149) use 'summed log-likelihood difference'. Two bins are negative (-13%), so the figure's 'gain' label differs from the text's term (CHANGELOG run 2 says these were named apart in the text)."},
    {"line": 113, "text": "dashed = test fold, pass A only (study 02's held-out half)",
     "why": "PLAN check 1 (lines 130-132) says 3 test mice (788733, 778103, 810758) have session splits that differ from study 02's. Pass A on the freeze's roles is therefore study 02's half for 146 of 149 mice, not all of them. S1 shows the pooled effect is small (max |Δ| 4.7e-05)."},
    {"line": 76, "text": "**b**, nested training subsets per fold (D = 10, 30, 100, 300, full)",
     "why": "cant_verify: PLAN line 100 states nesting for the pool-fold subsets (freeze_scaling_subsets.py). The test fold's subsets come from the wrapper's subject_ratio sampling (study 02, plus the new D=300 run). No supplied file shows whether those are nested."}
  ]
}
```

## Claims check of r1 (cv-predictability)

**Verdict:** all 4 claims are supported. There are no contradictions. There are 3 missing caveats and 7 estimand notes; one of the notes says something cannot be checked.

**C1/C2 (claims and figures)**
- **F1** is supported: gain 0.01225 ± 0.00073 (n = 5 folds), and all 5 per-fold gains are positive. Every fold also rises at every D step.
- **F2** is supported: r = 0.9605 (n = 763 mice). The 5th and 95th percentiles are 0.6233 and 0.8247.
- **F3** is supported: share = 1.1789, overall +0.0016 ± 0.0001, 3-way higher in 5 of 5 folds, n = 760 mice. The first-trial bin is +0.0337 on 4.45% of trials.
- **F4** is supported: P = 0.95 (random splits) and 0.81 (within-strata re-deals); test-fold offset −0.0024, P = 0.69. The limited power of the test is stated at lines 304–305 and 317–318.

**C9 (fixed conclusion text)**
- F1: the verdict code checks only full size against D = 10, but the heading (line 108) says "rises with training mice". The data do rise at every step, so the heading is true now.
- F3: the Fig. 3 caption (line 147) is a fixed string. It agrees with the data now.

**C3 (unsourced numbers)**
- Line 316, "two D = 10 models on Cedar A100" and "H200": these come only from README and PLAN prose, not from a JSON.

**C4 (stale text)**
- Line 358 points to the README variants table. That table still lists `test-fold-ref` as "planned" and scoring as "running". PLAN's status line, linked at line 90, says "scoring not started".

**C5 (missing caveats)**
- **F1:** every D = 10 model and 3 of the 5 D = 30 models use λ = 0 checkpoints from steps 10k–40k. All models at D ≥ 100 use λ = 1. Table S4 shows this, but no caveat says what it means for reading the gain as a training-mice effect.
- **F3:** in the test fold, the 2-way model is study 01's run, trained on a different mouse list. So in that fold the 3-way vs 2-way difference is not only the ignore policy.
- **Pass A > pass B caveat:** it cites only the 6 test-fold models. scores.json shows pass A > pass B in all 32 models, by +0.0023 to +0.0088 depending on the fold. The Fig. 4 caveat uses the test-fold size as if it were general.

**C6/C7 (estimands and uncertainty)**
- **Line 289:** compares the per-subject SD (0.062) with the trial-pooled gain (0.0123). These are different aggregations.
- **Line 303:** "about as wide as the whole scaling gain". The 95% range is 0.0234 wide, about 1.9 times the gain; only its half-width matches.
- **Line 334:** "similar size". The relevant pass offset is +0.0012 (3-way) or +0.0014 (2-way), not the +0.0030 cited from S1. The mouse-set offset is −0.0024 or −0.0032, so the pass offset is about half as large.
- **Line 333:** the 2-way models cover 760 mice, not 763.
- **Fig. 3 SVG:** panels b and c are labelled "gain", but the text says "summed log-likelihood difference".
- **Line 113:** "study 02's held-out half" holds for 146 of 149 mice (PLAN check 1).
- **Line 76 (cannot verify):** no supplied file shows whether the test fold's subsets are nested.

**C8:** there is no previous live version of this report, so there is no diff to check.

**Digest conventions:**
- report_digest is the sha256 of the report file.
- inputs_digest is the sha256 of `json.dumps({role/path: sha256}, sort_keys=True, separators=(",", ":"))` over 46 files. They are keyed `current/` for this study's files plus README, PLAN, CHANGELOG and CONTEXT.md, and `upstream/` for the 5 files from studies 01 and 02. The report is digested on its own and is not in that map.

**Files:**
- Report: /home/han.hou/code/aind-dynamic-foraging-bfm-studies/.claude/worktrees/bridge-cse_01XAC3mCG9zmJCgJVYKaNFCP/studies/17-gru-crossfit-predictability/analysis/reports/r1-cv-predictability.md
- Producer: /home/han.hou/code/aind-dynamic-foraging-bfm-studies/.claude/worktrees/bridge-cse_01XAC3mCG9zmJCgJVYKaNFCP/studies/17-gru-crossfit-predictability/analysis/r1_cv_predictability.py
- Scores: /home/han.hou/code/aind-dynamic-foraging-bfm-studies/.claude/worktrees/bridge-cse_01XAC3mCG9zmJCgJVYKaNFCP/studies/17-gru-crossfit-predictability/analysis/scores.json
~~~~

## Responses to the findings

Final run under the round limit (skills 0.16.2, "Round limit"; this is run 3). Blocking findings are fixed in commit `6a6fcb1`, which needs one more run (run 4); every other finding is deferred to r1's next revision, with the text left as checked.

| # | Check | Finding | Blocking? | Response |
|---|---|---|---|---|
| 1 | C6/C3 | Line 333: "763 mice" for the 2-way models; they cover 760 | **yes** (wrong number) | Fixed: "763 mice for the 3-way models and 760 for the 2-way ones" |
| 2 | C4 | README variants table and PLAN status stale (context files) | no (context, not the report) | Fixed in the context files (README: study `active`, `test-fold-ref` done inside `scoring`, scoring done; PLAN status line) |
| 3 | C3 | "H200 / Cedar A100" sourced only in README/PLAN prose | no (platform facts, not results) | Deferred: the README variants table is the source; no JSON holds hardware |
| 4 | C5 (F1) | D = 10 and most D = 30 points are λ = 0 checkpoints from early steps; D ≥ 100 are λ = 1 | no | Deferred to the next revision: add a caveat that the D axis also changes which checkpoint best_eval picks (Table S4) |
| 5 | C5 (F3) | Test fold's 2-way model is study 01's run, trained on a different mouse list | no | Deferred: add to the F3 caveats; F3 holds in all 4 pool folds, where the pair differs only in ignore policy |
| 6 | C5 | Pass A > pass B holds in all 32 models, size depends on the fold; Fig. 4 caveat uses the test-fold size | no | Deferred: restate with the per-fold range |
| 7 | C6 | Line 289 compares a per-subject SD with a trial-pooled gain | no | Deferred: name the two aggregations |
| 8 | C6 | Line 303 "about as wide as the whole scaling gain": half-width | no | Deferred: say "half-width" |
| 9 | C6 | Line 334 "similar size": the pass-A-only offset is about half the mouse-set offset | no | Deferred: give both offsets |
| 10 | C6 | Fig. 3 panel labels still say "gain" | no | Deferred: relabel in `cv_ignore_split.py` |
| 11 | C6 | Line 113 "study 02's held-out half" holds for 146 of 149 mice | no | Deferred: add "(146 of 149 mice; PLAN check 1)" |
| 12 | can't verify | Whether the test fold's subsets are nested | no | Deferred: state that only the pool-fold subsets are nested by construction |
| — | C9 notes | F1 verdict checks only full vs D = 10; Fig. 3 caption is fixed text | no | Deferred: make the F1 verdict check every step |
