---
schema_version: 1
report: analysis/reports/r1-pilot-decoding.md
checked_on: '2026-11-19'
content_digest: 7d01091b864ebd81a782bf3a203a8a9969043c009266d497f8bbde460350830e
inputs:
  studies/30-gru-vs-glmhmm-latents/analysis/r1_pilot_decoding.json: 349dc65df16bad2fb1488cbd7a32c456368ab470e253f45f661988e1e8f0a43c
  studies/30-gru-vs-glmhmm-latents/analysis/frozen_r1.json: b6b70c9f6ee96d1539c5f5749f89176e2aeb91813a22305b1620517723276c10
  studies/30-gru-vs-glmhmm-latents/analysis/fig_r1_engaged.svg: 505cfe0ef033e0b0f2f5ec023c8a572abd66de8ed8ff00a223d1764c05b0a3fe
  studies/30-gru-vs-glmhmm-latents/analysis/fig_r1_kscan.svg: 847e2df4e32773a893ecb7f4d67fdbadc1b555bc3681272232b757274580b93a
  studies/30-gru-vs-glmhmm-latents/analysis/fig_r1_kselect.svg: abc2d00ad70dda9236e74c962e26e9d67ebd3bddf520370fa56947d560c8f272
  studies/30-gru-vs-glmhmm-latents/analysis/fig_r1_shuffle.svg: bae934eab0fb196dc7960dd0c4eb95b351c4d8cb26cf8c7b991431b09f461b40
  studies/30-gru-vs-glmhmm-latents/CONTEXT.md: b6b41852c1b1e44f52d1a8315a0a2b670796771c31c7d97fc343796236d1876f
  studies/14-glmhmm-vs-gru-heldout/analysis/reports/r2-refit-prior.md: 42653a97feb4a4f4297454f2b31c4356078407c59b28a5f549378b93479af526
  studies/14-glmhmm-vs-gru-heldout/analysis/reports/r1-round-1.md: bbb03622b2f2200eed5fecd4ee6bb201e8fc9b41147bb4acd72f4a9c80a20ab8
context:
  studies/30-gru-vs-glmhmm-latents/PLAN.md: 4bfa88a591698265c62920fd212c5f7d7d0c7a87b12625e0003201c9fc6519d0
code:
  studies/30-gru-vs-glmhmm-latents/analysis/r1_pilot_decoding.py: f57989945f4525cfe12ef10d72224be3c98fa677c756b022dda999866ec5938c
  studies/30-gru-vs-glmhmm-latents/analysis/freeze_r1.py: 5fae3e5cd214038dd2ef0dfe6349d1bb4703e8f37f9167d1ec8f823e2f0b2d45
  studies/30-gru-vs-glmhmm-latents/analysis/oof_r1.py: 669e25bd472e6b5478f98117d2fd9ebe9d34ef058acbd852be46ed6aa370178f
  studies/30-gru-vs-glmhmm-latents/analysis/pilot/50_probes.py: fdf56fb398f7f0e00729b0859afc2ba8e19aea662b8061d3030f58be40aef625
  studies/30-gru-vs-glmhmm-latents/analysis/pilot/99_shuffle_null.py: f9c80a7687d22cd6cf39c419655b84be1d632b2bfc80b3879e041137f6f544b3
  studies/30-gru-vs-glmhmm-latents/analysis/pilot/kscan_glmhmm.py: 5384ce70d67950928e33f566ae4180e2b11d5f67b1559fe25902177825ae59a2
  studies/30-gru-vs-glmhmm-latents/analysis/pilot/kscan_probes.py: b832f12e23aebdf0fd81002cbb6b7183eb9e9ac37d0857341cd98893b54b0cb6
  studies/30-gru-vs-glmhmm-latents/analysis/pilot/kscan_report.py: 7e4e92f8ddeff015c2dea464196370a59d5e0e9cc23c16028871d81ad9c77c47
rubric_version: claims-check/3 @ 31fc2ddda2a15d7cb4aa5f5ca8fd710361c1d9e1
model: claude-opus-5-5 (requested effort high; reviewer saw a low setting)
prompt_sha256: 6591a90153d6ebd6cd30347c42289387131cfac2d81900fc8e48e0a0b9ebaafe
response_sha256: 1a161ad8b6481fe46fef31f9ef50c8bf3855d589ceb2cf114d5a6f35fe7fe1b4
severity_counts:
  High: 0
  Medium: 9
  Low: 4
---
# Claims check — r1, 2026-11-19

The record describes the report; it never feeds the frontmatter, INDEX or README (ADR 0003).

## Prompt given to the reviewer

~~~~text
You are the claims check for one report. Your instructions are the rubric file below; read it first and follow it exactly.

Rubric: /home/han.hou/.claude/plugins/cache/aind-behavior-foundation-model/aind-behavior-foundation-model-skills/0.17.1/skills/posthoc-reporting/references/claims-check-rubric.md
rubric_version: claims-check/3 @ 31fc2ddda2a15d7cb4aa5f5ca8fd710361c1d9e1

Read-only checkout pinned to the report's commit (a detached git worktree; do not fetch, and do not read any other ref or commit):
  /home/han.hou/code/wt/s30-r1-cc-7ff172c   (studies commit 7ff172c96c6dfca307619e4ea3fbc27a78c84362)

Use read-only tools only, except ONE output file, which you write once:
  /allen/aind/scratch/han.hou/pilots/study30-probe/r1live/claims_check_run1_response.md
Write the rubric's JSON, then its short markdown summary, to that file. Reply with only the file's path and one line of counts.

Starting points
- Report: studies/30-gru-vs-glmhmm-latents/analysis/reports/r1-pilot-decoding.md
- Last live version: none. The report has never been live (it was a version-1 draft); there is no diff to a live version, so C8 has nothing to compare against.
- Claim pointers resolved at this commit:
  F1 analysis/r1_pilot_decoding.json#/f1/latents_min_pooled_r2/point = 0.8154362643827469
  F2 analysis/r1_pilot_decoding.json#/f2/latents/engaged_r2_S1_vs_S2/point = 0.881832562468335
  F3 analysis/r1_pilot_decoding.json#/f3/latents_null_max_within = 0.03671516969603783
  F4 analysis/r1_pilot_decoding.json#/f4/latents_K3_minus_K10_pooled_r2/point = 0.08975409085779651
  (paths are relative to studies/30-gru-vs-glmhmm-latents/)

Follow the report's references yourself: its JSON (r1_pilot_decoding.json, frozen_r1.json), its figures (fig_r1_*.svg; read the SVG text and data), the producer (analysis/r1_pilot_decoding.py), the freeze and refit code (analysis/freeze_r1.py, analysis/oof_r1.py, analysis/pilot/), the study README, CONTEXT.md and PLAN.md. The per-trial scratch files the freeze read are not available to you; frozen_r1.json records their sha256 under _meta.sources. Commit messages, PR threads and notes are not evidence; CHANGELOG.md is a list of claims to check.
~~~~

## Reviewer's response

~~~~text
```json
{
 "report": "studies/30-gru-vs-glmhmm-latents/analysis/reports/r1-pilot-decoding.md",
 "report_digest": "71dafb77cd5c58b7db6c9dabb260339cd4d2c6ea50a11e713dbaed20df157c7b",
 "inputs_digest": "ce5eff2126effb64b71007990ba16c2ecbac09245f1ed29908eac6541a261523",
 "rubric_version": "claims-check/3 @ 31fc2ddda2a15d7cb4aa5f5ca8fd710361c1d9e1",
 "model": "claude-opus-5-5 (reasoning effort not visible to the reviewer)",
 "commit": "7ff172c96c6dfca307619e4ea3fbc27a78c84362",
 "claims": [
  {
   "id": "F1",
   "verdict": "supported",
   "evidence_pointer": "analysis/r1_pilot_decoding.json#/f1/latents_min_pooled_r2 (K 9, 0.8154 [0.8026, 0.8285]); #/f1/latents_min_pooled_r2_within (K 9, 0.7154 [0.7002, 0.7318]); #/f1/K10_pooled_r2/logits (0.2453 [0.2081, 0.2771]), /history (0.2845 [0.2440, 0.3220])",
   "note": "Every number in the sentence rounds correctly from the JSON, and both minima fall at K = 9. The latents beat history at every K: the smallest paired gap is #/f1/latents_minus_history_min_over_K = 0.193 [0.173, 0.217] (K = 2). The reference emission gives the same picture (pooled R2 0.817 to 0.942, frozen_r1.json#/kscan/parts/reference__K*__latents). 'Fall to ... at K = 10' is true at K = 10, but K = 10 is not the baselines' minimum (logits 0.23 at K = 9, Fig. 1 table line 135). The untrained GRU is the strongest baseline from K = 3 on; Caveats line 300 covers it. C2: fig_r1_kscan.svg is drawn from the same frozen_r1.json#/bootstrap/kscan entries (r1_pilot_decoding.py:250-254), and its sha256 matches r1_pilot_decoding.json#/_meta/outputs. Checked from the SVG text and the producer code, not from bar or point heights.",
   "item": "C1",
   "severity": null
  },
  {
   "id": "F2",
   "verdict": "supported",
   "evidence_pointer": "analysis/r1_pilot_decoding.json#/f2/latents/engaged_r2_S1_vs_S2 (0.8818 [0.8598, 0.8995]); #/f2/logits/engaged_r2_S1_vs_S2 (0.0291 [-0.0447, 0.0738]); #/f2/history/engaged_r2_S1_vs_S2 (-0.0003 [-0.0704, 0.0586])",
   "note": "The values and intervals match. The latents' interval excludes 0, and it is far above both baseline intervals. The confident-trial S2 -> S2 rate agrees: 0.96 for latents against 0.02 and 0.06 (result-2 table, line 175-177). The 'do not' wording is a separate C7 finding. C2: in fig_r1_engaged.svg the text gives balanced acc. 0.98 / 0.63 / 0.64 and the S2 row as 0.01 0.03 0.96 (latents), 0.01 0.97 0.02 (logits) and 0.01 0.93 0.06 (history). This matches the table and frozen_r1.json#/k3_engaged/sets/*/confusion_confident_counts. Panel d reads frozen_r1.json#/bootstrap/k3_engaged (r1_pilot_decoding.py:223-229); bar heights were not measured.",
   "item": "C1",
   "severity": null
  },
  {
   "id": "F3",
   "verdict": "supported",
   "evidence_pointer": "analysis/r1_pilot_decoding.json#/f3/latents_real_within_min (0.722), #/f3/latents_real_within_max (0.877), #/f3/latents_null_max_within (0.0367), #/f3/n_shuffles_min (5), #/f3/n_shuffles_max (10)",
   "note": "The within-session latents scores and the largest latents null (swap, K = 2, 0.0367) match '0.72 to 0.88' and 'at or below 0.04'. Every latents swap null is near 0, and every circular-shift null is about -0.17 to -0.20 (result-3 table, lines 191-194). The bound holds only for the latents nulls, scored within session. The history swap nulls reach 0.08 (line 195), and the all-trials circular-shift nulls reach 0.43 (#/f3/nulls/latents_K2/circ/mean_all). See the C6 and C7 entries. C2: fig_r1_shuffle.svg plots the per-shuffle values from frozen_r1.json#/shuffle (r1_pilot_decoding.py:353-357); its sha256 matches _meta.outputs. Side note, outside the report: PLAN.md:21, which line 293 links, says the nulls are '<= 0.03 within session', but the JSON maximum is 0.0367.",
   "item": "C1",
   "severity": null
  },
  {
   "id": "F4",
   "verdict": "supported",
   "evidence_pointer": "analysis/r1_pilot_decoding.json#/f4/latents_K3_minus_K10_pooled_r2 (0.0898 [0.0794, 0.1010]); frozen_r1.json#/bootstrap/kscan/*/latents/pooled_r2/point; frozen_r1.json#/kscan/glmhmm/hurdle/*/logged_train_mice_eval_likelihood",
   "note": "The paired difference and its interval match. The latents' readout peaks at K = 2 (0.941), not at K = 3 (0.908), so 'does not single out K = 3' holds. The round-1 refit score in this report's W&B group does peak at K = 3 (0.62887, against 0.62778 at K = 2 and 0.62841 at K = 4: one fit per K, no interval). 'Declines steadily' describes the trend; the point values rise twice (K = 6 -> 7: 0.837 -> 0.844; K = 9 -> 10: 0.815 -> 0.818; the within-session score also rises at 6 -> 7). 'Only 0.09 lower' is a framing word: the paired interval [0.08, 0.10] excludes 0, so K = 3 is read significantly better than K = 10. The other problems with F4 are separate entries (C4, C5, C9, and the C6/C7 entries on lines 206 and 279).",
   "item": "C1",
   "severity": null
  },
  {
   "id": "F4",
   "verdict": "supported",
   "evidence_pointer": "analysis/r1_pilot_decoding.py:618-620, 626; reports/r1-pilot-decoding.md lines 206, 214",
   "note": "C9, line 206: 'a: the latents' decoding of the K-state posterior declines steadily with K. b: ... the global fit on these trials rises with K; study 14's per-mouse refit score, the one it selects K by, peaks at K = 3.' The producer writes this as a fixed string (r1_pilot_decoding.py:618-620) inside the result-4 block, so it cannot change with the data. Today 'rises with K' and 'peaks at K = 3' agree with frozen_r1.json#/kscan/glmhmm/hurdle. 'Steadily' is loose (two rises, see the F4 C1 note). Line 214 'Intervals: 95% over mice (n = 50)' is also hard-coded (line 626), and so is the Fig. 1 K-scan caption's 'n = 50' (r1_pilot_decoding.py:261). Severity is Medium, not High, because none of these strings contradicts its table today.",
   "item": "C9",
   "severity": "Medium"
  }
 ],
 "producer_traces": [
  {
   "claim": "F1",
   "producer": "analysis/r1_pilot_decoding.py:512,520 (selection of K) <- analysis/freeze_r1.py:281,283 (value)",
   "matches": "true",
   "note": "freeze_r1.py:281 computes pooled R2 = 1 - sum over states of SSE / sum over states of SST. It pools all out-of-fold trials of the 50 mice; row 0 of W is the full sample (the point), and rows 1-1000 resample mice. The out-of-fold predictions come from oof_r1.py:112-114: 5 folds by mouse, each probe fitted on 40 mice and used to predict the other 10. r1_pilot_decoding.py:512 picks the K with the lowest point value. This is the claim's estimand: trial-pooled and state-pooled, held out by mouse, with the interval over mice (n = 50). The within-session value comes from freeze_r1.py:281 (session means removed) via r1_pilot_decoding.py:513,521. The baselines come from r1_pilot_decoding.py:514,522.",
   "item": "C0",
   "severity": null
  },
  {
   "claim": "F2",
   "producer": "analysis/r1_pilot_decoding.py:526 <- analysis/freeze_r1.py:296-297,304",
   "matches": "true",
   "note": "Engaged trials are those with target pi_S0 < 0.2 (freeze_r1.py:243,296; n = 234,546). The target is c = P1/(P1+P2) and the prediction is Q1/(Q1+Q2), from the same 3-class out-of-fold probe. R2 is pooled over engaged trials, with denominators re-centred on each resample (line 304). This matches 'S1 vs S2 R2 on engaged trials at K = 3', with the interval over mice (n = 50).",
   "item": "C0",
   "severity": null
  },
  {
   "claim": "F3",
   "producer": "analysis/r1_pilot_decoding.py:517 <- null_cells 496-504 <- analysis/pilot/99_shuffle_null.py:97-104 and analysis/pilot/kscan_probes.py:79-82",
   "matches": "true",
   "note": "The value is the largest per-shuffle within-session pooled R2 over the latents swap and circular-shift nulls at K = 2, 3, 6 and 10. Each shuffle refits the decoder on the shuffled pairs and scores it out of fold. The claim names the same quantity, but only for the latents: the pointer leaves out the history nulls, which the sentence's 'every ... null' could be read to include.",
   "item": "C0",
   "severity": null
  },
  {
   "claim": "F4",
   "producer": "analysis/r1_pilot_decoding.py:535 <- analysis/freeze_r1.py:386-388",
   "matches": "true",
   "note": "The value is the paired difference of bootstrap rows: pooled R2 at K = 3 minus pooled R2 at K = 10, for the latents, with the same resampled mice on both sides (W shared, freeze_r1.py:363). It is the claim's paired 95% interval over n = 50 mice.",
   "item": "C0",
   "severity": null
  }
 ],
 "stale_sentences": [
  {
   "line": 36,
   "text": "Readability does not single out K = 3, the K that study 14's per-mouse refit score favours",
   "why": "The same framing appears at line 73 (Answer), line 206 ('study 14's per-mouse refit score, the one it selects K by, peaks at K = 3') and lines 279-280 ('its score has no peak at the K that study 14's per-mouse refit selects'). Study 14's live r2 has replaced round 1's refit. r2-refit-prior.md:133-136 says: 'Round 1's early peak came from the refit, not from the data. With each mouse pooled toward the global fit, the selection score rises through K = 10 ... Round 1's peaks at K = 3 and K = 2 were the zero-centred refit overfitting each mouse.' r2-refit-prior.md:52 says: 'hurdle: the rule picks K = 10 ... round 1 picked K = 3'. The report describes K = 3 as study 14's selected K without saying that this was round 1 and has been superseded. Under the current rule (K = 10), readability is near its lowest (0.82), not at a non-peak. The K = 3 peak is also small: 0.0005 above K = 4, from one fit per K. Severity is Medium, not the default Low, because the sentence sits in claim F4 and in its figure caption.",
   "item": "C4",
   "severity": "Medium"
  }
 ],
 "unsourced_numbers": [
  {
   "line": 105,
   "text": "the GLM-HMM's 40 lag regressors",
   "why": "No JSON field holds 40. The number comes from code only: analysis/pilot/50_probes.py:41,48 (L = 10 lags x RewC, UnrC, Ignored, Rewarded; x[:, :40]). It agrees with frozen_r1.json#/k3/results/pred|3_history/n_features = 56 (40 lags + 6 EMAs + 10 trial-index spline columns). Low: the number is right and only describes the features.",
   "item": "C3",
   "severity": "Low"
  },
  {
   "line": 310,
   "text": "Repeat on the 149 held-out mice",
   "why": "The number is in neither r1_pilot_decoding.json nor frozen_r1.json. It traces only to PLAN.md:35 and study 14 r1 (r1-round-1.md:68), which is not listed in the report's upstream. Low, not High: it sits in Next and supports no claim.",
   "item": "C3",
   "severity": "Low"
  },
  {
   "line": 284,
   "text": "the 3-way GRU receives the previous choice as one number (0 = left, 1 = right, 2 = ignore), so an ignore enters as an \"extra-right\" choice",
   "why": "cant_verify. No code, config or JSON in the tree shows the GRU's input encoding: frozen_r1.json#/pilot/gru/architecture has no input field, and the wrapper is not in the checkout. Only PLAN.md:86 asserts it. The sentence is the report's explanation for the cosine of 0.90 (SupFig. 1).",
   "item": "C3",
   "severity": "Medium"
  }
 ],
 "missing_caveats": [
  {
   "claim": "F1",
   "why": "'Recovers the GLM-HMM's predictive state posterior ... at every K' rests on pooled R2, which variance-rich states dominate. The per-state R2 in frozen_r1.json#/kscan/parts/hurdle__K8__latents/state_r2 includes 0.37 for a state with 1.7% occupancy (#/kscan/glmhmm/hurdle/8/occupancy). At K = 10 (#/kscan/parts/hurdle__K10__latents/state_r2), a state with 2.1% occupancy scores 0.33. These are point values with no interval. The Caveats do not say that rare states at large K are decoded much worse than the pooled score suggests.",
   "item": "C5",
   "severity": "Medium"
  },
  {
   "claim": "F4",
   "why": "F4 reads the fall of pooled R2 with K as a property of readability. The evidence shows that the target changes with K. Its entropy rises from 0.22 to 0.91 nats (frozen_r1.json#/kscan/glmhmm/hurdle/*/target_entropy), its within-session share rises from 0.40 to 0.51 (#/within_session_share_pooled), and rare states appear (occupancy 0.9% and 2.1% at K = 10). Part of the decline comes from those poorly decoded rare states (see the F1 caveat); most states stay at 0.73 or above at K = 10. CONTEXT.md:34 asserts pooled R2 is 'comparable across K' but gives no support. The Caveats do not name this alternative reading of the K trend.",
   "item": "C5",
   "severity": "Medium"
  }
 ],
 "estimand_issues": [
  {
   "line": 206,
   "text": "b: the GLM-HMM's predictive likelihood: the global fit on these trials rises with K; study 14's per-mouse refit score, the one it selects K by, peaks at K = 3.",
   "why": "C6. Fig. 4b puts two curves on one axis that differ in their mice as well as their fit. 'These trials (global fit)' is the 50 pilot mice's odd sessions (analysis/pilot/kscan_glmhmm.py:99). 'Study 14 logged (per-mouse refit)' is W&B glmhmm/select/eval_likelihood over all 614 training mice (kscan_glmhmm.py:100; kscan_report.py:65 labels it 'logged (all 614 mice)'). The caption and the legend name only the fit difference. A reader cannot tell whether the rise and the peak differ because of the fit or because of the mice.",
   "item": "C6",
   "severity": "Medium"
  },
  {
   "line": 33,
   "text": "while every session-swap and circular-shift null stays at or below 0.04",
   "why": "C6. The bound (#/f3/latents_null_max_within) covers only the latents nulls, scored within session. The history session-swap nulls reach 0.08 (line 195: '0.07 (range 0.06 to 0.08, n = 5)'), and the latents circular-shift nulls reach 0.43 on all trials (#/f3/nulls/latents_K2/circ/mean_all). The sentence does not say 'latents' before 'null'. Low: the subject of the sentence implies the scope, and the claim's meaning is unchanged.",
   "item": "C6",
   "severity": "Low"
  },
  {
   "line": 282,
   "text": "Choice and outcome dominate the latents' variance; the state variables add a small unique share on held-out mice.",
   "why": "C6. The table behind this (lines 228-234) reports drop-one unique R2, which is not a share of variance. Per variable, the state's unique R2 overlaps the choice and outcome terms: p(ignore-prone) 0.05 [0.02, 0.08], choice history 0.05 [0.04, 0.07], reward t-1 0.05 [0.04, 0.06] (frozen_r1.json#/bootstrap/tdr/unique_r2). 'Dominate' holds only when six choice/outcome variables are summed against two state variables, and the text does not say that.",
   "item": "C6",
   "severity": "Medium"
  },
  {
   "line": 30,
   "text": "but the GRU's logits (0.03 [−0.04, 0.07]) and linear history (0.00 [−0.07, 0.06]) do not",
   "why": "C7. This is a no-effect statement from intervals that include 0, with no margin stated. The same reading appears at line 71 and at line 273 ('carry engagement but not that split'). The intervals are narrow (upper bounds 0.07 and 0.06, against 0.88 for the latents), so stating the bound ('at most 0.07') would be enough. Medium is the rubric's default for this case.",
   "item": "C7",
   "severity": "Medium"
  },
  {
   "line": 279,
   "text": "The readout recovers the K = 10 posterior nearly as well as the K = 3 one",
   "why": "C7. This is a near-equality statement with no stated margin. The paired difference is 0.09 [0.08, 0.10] on all trials and 0.10 [0.09, 0.12] within session (#/f4/latents_K3_minus_K10_pooled_r2, #/f4/latents_K3_minus_K10_pooled_r2_within); both intervals exclude 0. F4's 'only 0.09 lower' (lines 36, 73) has the same framing.",
   "item": "C7",
   "severity": "Medium"
  },
  {
   "line": 33,
   "text": "within session the latents score 0.72 to 0.88 (K = 2, 3, 6, 10)",
   "why": "C7. The claim sentence gives point values with no interval and no n. The intervals are in the result-3 table (lines 191-194, 95% over n = 50 mice). Low: the gap to the nulls is about 0.7, so the missing intervals do not change the reading.",
   "item": "C7",
   "severity": "Low"
  }
 ]
}
```

## Claims check summary (r1-pilot-decoding, studies 7ff172c, claims-check/3)

Claims: 4 supported, 0 overstated, 0 contradicted, 0 cant_verify. Findings: 0 High, 9 Medium, 4 Low. C8: not applicable (no live version).

**C0 (producer traces):** F1, F2, F3 and F4 each trace to a producer line that computes the named quantity (no findings). The F3 pointer covers latents nulls only.
**C1:** F1, F2, F3 and F4 are supported by their pointers.

**C9**
- Medium: line 206 (and 214). The Fig. 4 caption ('declines steadily', 'rises with K', 'peaks at K = 3') and 'n = 50' are fixed strings in result-4 (r1_pilot_decoding.py:618-620, 626). They agree with the data today; 'steadily' is loose (rises at K 6->7 and 9->10).

**C4**
- Medium: lines 36/73/206/279-280. Study 14's 'K = 3' is round 1's refit. Study 14's live r2 says that peak came from refit overfitting and that its rule now picks K = 10 (r2-refit-prior.md:52, 133-136).

**C3**
- Medium (cant_verify): line 284. The GRU's choice input as one number 0/1/2: only PLAN.md:86 says so; no code or JSON in the tree shows it.
- Low: line 105. '40 lag regressors' traces to code only (50_probes.py:48), consistent with n_features 56.
- Low: line 310. '149 held-out mice' is in PLAN.md and study 14 r1 only, not in the report's JSON or upstream.

**C5**
- Medium: F1. Pooled R2 hides rare states that decode poorly (K = 8: state R2 0.37 at 1.7% occupancy; K = 10: 0.33 at 2.1%).
- Medium: F4. The target changes with K (entropy 0.22 -> 0.91 nats, rarer states), an alternative reading of the decline that the Caveats do not name.

**C6**
- Medium: line 206 / Fig. 4b. The two likelihood curves differ in mice (50 pilot mice vs all 614 training mice) as well as in fit; only the fit is named.
- Medium: line 282. 'Choice and outcome dominate the latents' variance' is not shown by drop-one unique R2; p(ignore-prone) 0.05 equals choice history 0.05 and reward t-1 0.05.
- Low: line 33. F3's '<= 0.04' bound covers latents, within-session nulls only (history swap reaches 0.08).

**C7**
- Medium: lines 30/71/273. 'Do not' / 'not that split' rests on intervals that include 0, with no stated margin (upper bounds 0.07, 0.06).
- Medium: line 279. 'Nearly as well' (and F4's 'only 0.09'): the paired difference 0.09 [0.08, 0.10] excludes 0 and no margin is stated.
- Low: line 33. '0.72 to 0.88' in F3 has no interval or n in the sentence (the table has them).

Inputs hashed for inputs_digest (role/path: sha256):

- current/studies/30-gru-vs-glmhmm-latents/CONTEXT.md: b6b41852c1b1e44f52d1a8315a0a2b670796771c31c7d97fc343796236d1876f
- current/studies/30-gru-vs-glmhmm-latents/PLAN.md: 4bfa88a591698265c62920fd212c5f7d7d0c7a87b12625e0003201c9fc6519d0
- current/studies/30-gru-vs-glmhmm-latents/analysis/fig_r1_engaged.svg: 505cfe0ef033e0b0f2f5ec023c8a572abd66de8ed8ff00a223d1764c05b0a3fe
- current/studies/30-gru-vs-glmhmm-latents/analysis/fig_r1_kscan.svg: 847e2df4e32773a893ecb7f4d67fdbadc1b555bc3681272232b757274580b93a
- current/studies/30-gru-vs-glmhmm-latents/analysis/fig_r1_kselect.svg: abc2d00ad70dda9236e74c962e26e9d67ebd3bddf520370fa56947d560c8f272
- current/studies/30-gru-vs-glmhmm-latents/analysis/fig_r1_shuffle.svg: bae934eab0fb196dc7960dd0c4eb95b351c4d8cb26cf8c7b991431b09f461b40
- current/studies/30-gru-vs-glmhmm-latents/analysis/freeze_r1.py: 5fae3e5cd214038dd2ef0dfe6349d1bb4703e8f37f9167d1ec8f823e2f0b2d45
- current/studies/30-gru-vs-glmhmm-latents/analysis/frozen_r1.json: 2fe7fc74d31207cd1a131b56feb46f1be85f611aaf413d1cc3387fe3ede111e8
- current/studies/30-gru-vs-glmhmm-latents/analysis/oof_r1.py: 669e25bd472e6b5478f98117d2fd9ebe9d34ef058acbd852be46ed6aa370178f
- current/studies/30-gru-vs-glmhmm-latents/analysis/pilot/50_probes.py: fdf56fb398f7f0e00729b0859afc2ba8e19aea662b8061d3030f58be40aef625
- current/studies/30-gru-vs-glmhmm-latents/analysis/pilot/99_shuffle_null.py: f9c80a7687d22cd6cf39c419655b84be1d632b2bfc80b3879e041137f6f544b3
- current/studies/30-gru-vs-glmhmm-latents/analysis/pilot/kscan_glmhmm.py: 5384ce70d67950928e33f566ae4180e2b11d5f67b1559fe25902177825ae59a2
- current/studies/30-gru-vs-glmhmm-latents/analysis/pilot/kscan_probes.py: b832f12e23aebdf0fd81002cbb6b7183eb9e9ac37d0857341cd98893b54b0cb6
- current/studies/30-gru-vs-glmhmm-latents/analysis/pilot/kscan_report.py: 7e4e92f8ddeff015c2dea464196370a59d5e0e9cc23c16028871d81ad9c77c47
- current/studies/30-gru-vs-glmhmm-latents/analysis/r1_pilot_decoding.json: e579f0e901edeb87ef47c4fb7f8548811fd3ae8561fa4f800a9ceae223527c17
- current/studies/30-gru-vs-glmhmm-latents/analysis/r1_pilot_decoding.py: f57989945f4525cfe12ef10d72224be3c98fa677c756b022dda999866ec5938c
- upstream/studies/14-glmhmm-vs-gru-heldout/analysis/reports/r1-round-1.md: bbb03622b2f2200eed5fecd4ee6bb201e8fc9b41147bb4acd72f4a9c80a20ab8
- upstream/studies/14-glmhmm-vs-gru-heldout/analysis/reports/r2-refit-prior.md: 42653a97feb4a4f4297454f2b31c4356078407c59b28a5f549378b93479af526
~~~~

## Responses to the findings

Run 1 of at most 3 for this version. Reviewer usage: 211,649 tokens, 35 tool calls, about 20 min.

Note on the run: the brief and the Agent call asked for Opus at high effort; the reviewer reports its context showed a low reasoning-effort setting (15). The run is recorded as it ran. The final run will be checked for the effort it gets.

Errata reported by the reviewer (its file left as written): the F3 producer trace should cite analysis/pilot/99_shuffle_null.py lines 88-96, not 97-104.

Answers (fixes follow in the next commit, minimal: narrow or delete rather than add):

| # | Check | Severity | Finding | Response |
|---|---|---|---|---|
| 1 | C4 | Medium | F4, What it means and Fig. 4b cite study 14's per-mouse refit as favouring K = 3; study 14's live r2 replaced that refit and its selection score rises through K = 10 | Fix: every reference to study 14's K selection is removed (F4 text, What it means, Fig. 4 caption), and Fig. 4b keeps only the global fit on the same 50 mice |
| 2 | C9 | Medium | Fig. 4 caption states directions ("rises", "peaks at K = 3") as a fixed string | Fix: the caption only names what each panel plots; no direction words |
| 3 | C6 | Medium | Fig. 4b puts two curves with different mice (50 pilot mice vs 614 training mice) on one axis | Fix: the 614-mouse logged curve is dropped (see 1) |
| 4 | C5 | Medium | F1: pooled R² hides rare states at large K that decode much worse (state R² 0.37 at 1.7% occupancy, K = 8) | Fix: one caveat sentence: pooled R² is dominated by common states; rare states at large K decode worse |
| 5 | C5 | Medium | F4: the target changes with K (entropy, within-session share, rare states), so the fall in pooled R² is not only readability | Fix: covered by the same caveat, which says pooled R² across K compares different targets; F4's interpretation narrowed to "decoding does not peak at any K" |
| 6 | C7 | Medium | F2 "do not" from intervals that include 0 | Fix: F2 states the bound: logits and history recover at most 0.07 and 0.06 (upper 95% bounds) |
| 7 | C7 | Medium | "nearly as well" (K = 10 vs K = 3) without a margin; both intervals exclude 0 | Fix: replaced by the paired difference via F4; "only" removed from F4 |
| 8 | C6 | Medium | "Choice and outcome dominate the latents' variance" not supported by per-variable unique R² | Fix: reworded to "each state variable adds a unique share about as large as single choice or outcome variables" |
| 9 | C3 | Medium | 0/1/2 previous-choice encoding is cant_verify from the tree | Fix: the sentence cites the pilot code that builds the GRU inputs as the GRU trainer does (analysis/pilot/10_load_trials.py) |
| 10 | C3 | Low | "40 lag regressors" in no JSON | Fix: written as the design constant in inline code |
| 11 | C3 | Low | "149 held-out mice" in Next has no source in r1 | Fix: number removed ("the held-out mice") |
| 12 | C6 | Low | F3 "every null" scope (latents only, within session) | Fix: F3 says "every within-session null of the latents decoder" |
| 13 | C7 | Low | F3 gives point values without intervals | Fix: F3 points to the Fig. 3 table for the intervals |
