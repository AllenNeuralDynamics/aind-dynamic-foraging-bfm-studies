"""Producer for r1: the cross-validated GRU predictability report, from the study's committed JSONs.

Reads the frozen scores (scores.json), the fold and role references, and the JSON of each figure
producer (cv_*.py); computes the values the report's claims point at; rewrites the report's
methods, result-1..4 (main findings) and supp-1..4 (validation and housekeeping) blocks. Draws
no figure: each figure belongs to its own producer and is only embedded here.

    python analysis/r1_cv_predictability.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

REPORT = HERE / "reports" / "r1-cv-predictability.md"
OUT = HERE / "r1_cv_predictability.json"
WANDB_GROUPS = ["scoring@20261114-034950", "scoring@20261114-081940", "scoring@20261114-092226"]
SRC = {k: HERE / f"{k}.json" for k in ("scores", "cv_headline", "cv_ignore_split", "cv_2way_vs_s01", "cv_fold_spread",
                                        "cv_crosscheck", "cv_lambda", "cv_rebuild_check", "cv_schematic")}
SRC["roles"] = STUDY / "reference" / "roles.json"
SRC["folds"] = STUDY / "reference" / "folds.json"
FOLDS = ["test", "pool_1", "pool_2", "pool_3", "pool_4"]
FULL3 = {"test": "test-3way-full-s0", **{f"pool_{k}": f"pool{k}-3way-full" for k in range(1, 5)}}
DS = ["10", "30", "100", "300", "full"]
BIN_LABEL = {"1": "1st", "2-4": "2nd–4th", "5-19": "5th–19th", "20+": "20th or later", "none": "before any ignore"}


def f(x: float, spec: str = ".4f") -> str:
    return format(x, spec).replace("-", "−")


def sem(x: float) -> str:
    """An SEM to 2 significant digits, so a small one does not print as 0.0000."""
    return f(x, ".2g") if x else "0"


def rewrite(text: str, block: str, body: str) -> str:
    pat = re.compile(rf"(<!-- BEGIN {re.escape(block)} -->\n)(.*?)(<!-- END {re.escape(block)} -->)", re.S)
    assert pat.search(text), f"no marker pair for {block}"
    return pat.sub(lambda m: m.group(1) + body.rstrip() + "\n" + m.group(3), text, count=1)


def per_mouse_full(models: dict) -> dict[str, float]:
    out = {}
    for fold, name in FULL3.items():
        sc = models[name]["scores"]["lam-checkpoint"]
        for sid in sc["A"]["mice"]:
            ll = sum(sc[p]["mice"][sid][0] for p in "AB")
            n = sum(sc[p]["mice"][sid][1] for p in "AB")
            out[sid] = float(np.exp(ll / n))
    return out


def main() -> None:
    d = {k: json.loads(p.read_text()) for k, p in SRC.items()}
    models = d["scores"]["models"]
    hd, ig, s01, sp, cc, lam, rb = (d[k] for k in ("cv_headline", "cv_ignore_split", "cv_2way_vs_s01", "cv_fold_spread",
                                                   "cv_crosscheck", "cv_lambda", "cv_rebuild_check"))
    # F1: the cross-validated scaling curve
    gains_full = {fo: hd["per_fold"][fo]["full"]["lr_engaged"] - hd["per_fold"][fo]["10"]["lr_engaged"] for fo in FOLDS}
    n_pos = sum(g > 0 for g in gains_full.values())
    f1 = {"curve": hd["summary"]["lr_engaged"], "gain_over_d10": hd["summary"]["lr_engaged_gain_over_d10"],
          "gain_full": hd["summary"]["lr_engaged_gain_over_d10"]["full"], "gain_full_per_fold": gains_full,
          "n_folds_full_gain_positive": n_pos, "n_folds": len(FOLDS),
          "verdict": "n/a" if not gains_full else ("rises in every fold" if n_pos == len(FOLDS) else f"rises in {n_pos} of {len(FOLDS)} folds"),
          "ignore_pr_auc": hd["summary"]["ignore_pr_auc"], "ignore_recall": hd["summary"]["ignore_recall"]}
    # F2: predictability is a property of the mouse
    pm = np.array(list(per_mouse_full(models).values()))
    f2 = {"n_mice": int(len(pm)), "per_mouse": {"p5": float(np.percentile(pm, 5)), "median": float(np.median(pm)),
                                                "p95": float(np.percentile(pm, 95)), "sd": float(pm.std(ddof=1)),
                                                "min": float(pm.min()), "max": float(pm.max())},
          "passA_passB_r": hd["map"]["passA_vs_passB_pearson_r"], "threeway_twoway_r": hd["map"]["3way_vs_2way_pearson_r"],
          "n_mice_2way": hd["map"]["n_mice_2way_vs_3way"],
          "sd_over_full_gain": float(pm.std(ddof=1) / hd["summary"]["lr_engaged_gain_over_d10"]["full"]["mean"])}
    # F3: 3-way vs 2-way, and where the gain sits
    o = ig["overall"]; b1 = ig["per_bin"]["1"]
    f3 = {"n_mice": hd["map"]["n_mice_2way_vs_3way"], "diff_mean": o["diff_mean"], "diff_sem": o["diff_sem"], "n_folds": o["n_folds"],
          "n_folds_3way_higher": o["n_folds_3way_higher"],
          "first_after_ignore": {"diff_mean": b1["diff_mean"], "diff_sem": b1["diff_sem"],
                                 "share_of_trials": b1["share_of_trials"], "share_of_loglik_gain": b1["share_of_loglik_gain"]},
          "far_from_ignore_20plus": {k: ig["per_bin"]["20+"][k] for k in ("diff_mean", "diff_sem", "share_of_trials")},
          "verdict": "n/a" if o["n_folds"] == 0 else ("3-way higher in every fold" if o["n_folds_3way_higher"] == o["n_folds"]
                                                       else f"3-way higher in {o['n_folds_3way_higher']} of {o['n_folds']} folds"),
          "study01_same_run": s01["study01_same_run"], "study17_test_2way_passA": s01["study17_test_2way_full_passA"],
          "study17_minus_study01_same_run": s01["study17_test_2way_full_passA"] - s01["study01_same_run"]["heldout_ll"]}
    # F4: the fold spread and the test split
    f4 = {"real_sd": sp["real_sd"], "perm_sd_median": sp["perm"]["sd_median"], "perm_sd_ci95": sp["perm"]["sd_ci95"],
          "p_sd_ge_real": sp["perm"]["p_sd_ge_real"], "n_perm": sp["perm"]["n"], "test_minus_all": sp["test_minus_all"],
          "all_mice": sp["all_mice_pooled"], "test_fold": sp["fold_values"]["test"],
          "test_offset_ci95": sp["test_offset_null"]["ci95"], "test_offset_p": sp["test_offset_null"]["p_abs_ge_real"],
          "stratified_pool": sp["stratified_pool"],
          "verdict": ("within which-mice sampling" if min(sp["perm"]["p_sd_ge_real"], sp["stratified_pool"]["p_sd_ge_real"]) >= 0.05
                      else "larger than which-mice sampling"),
          "test_offset_verdict": "within random 149-mouse sets" if sp["test_offset_null"]["p_abs_ge_real"] >= 0.05 else "outside random 149-mouse sets"}
    supp = {"crosscheck": cc["summary"], "lambda": lam["by_checkpoint_lambda"], "rebuild": rb}
    result = {"f1": f1, "f2": f2, "f3": f3, "f4": f4, "supp": supp}

    roles, folds = d["roles"], d["folds"]
    n_models = len(models)
    methods = "\n".join([
        "![Fig. M1. Design](../fig_cv_schematic.svg)",
        "",
        "**Fig. M1. Design.** **a**, the 763 mice in 5 folds; each fold model trains on the other 4 folds and scores "
        "its own. **b**, nested training subsets per fold (D = 10, 30, 100, 300, full). **c**, the two scoring passes.",
        "",
        "| fact | value |",
        "|---|---|",
        f"| mice | {roles['n_subjects']} (data snapshot `{roles['snapshot']}`) |",
        f"| folds | test {folds['sizes']['test']}, pool 1–4: {', '.join(str(folds['sizes'][k]) for k in FOLDS[1:])} mice |",
        f"| sessions with a role | {roles['n_sessions']:,} (fit {roles['roles']['fit']:,}, score {roles['roles']['score']:,}) |",
        f"| models scored | {n_models} (3-way: {sum(m['way'] == '3way' for m in models.values())}, "
        f"2-way: {sum(m['way'] == '2way' for m in models.values())}); hidden size 256 |",
        "| checkpoint | `best_eval`, scored at its own session λ (Supplementary S2) |",
        "| passes | A: tune on fit sessions, score on score sessions; B: swapped. Each session is scored once |",
        "| estimand | held-out likelihood, trial-pooled: `exp(Σ log p / n)` over scored trials, L/R on engaged trials; "
        "per fold, both passes pooled |",
        "| uncertainty | mean ± SEM over the 5 folds (n = folds; one model per fold and D, seed 0) |",
    ])

    r1 = ["![Fig. 1](../fig_cv_headline.svg)", "",
          "**Fig. 1. Cross-validated scaling.** **a**, held-out likelihood, trial-pooled (L/R, engaged), vs training mice D; "
          "thin lines = folds, black = mean ± SEM (5 folds); dashed = test fold, pass A only (study 02's held-out half). "
          "**b**, gain over D = 10 within each fold (same mice at every D). **c**, ignore PR-AUC. **d**, ignore recall "
          "(p_ignore ≥ 0.5). c and d: mean of passes A and B.", "",
          "| D | held-out likelihood | gain over D = 10 | ignore PR-AUC | ignore recall |", "|---|---|---|---|---|"]
    for k in DS:
        c, g, pr, rc = f1["curve"][k], f1["gain_over_d10"][k], f1["ignore_pr_auc"][k], f1["ignore_recall"][k]
        r1.append(f"| {k} | {f(c['mean'])} ± {sem(c['sem'])} | {f(g['mean'], '+.4f')} ± {sem(g['sem'])} | "
                  f"{f(pr['mean'])} ± {sem(pr['sem'])} | {f(rc['mean'])} ± {sem(rc['sem'])} |")
    r1 += ["", f"Mean ± SEM over n = {f1['n_folds']} folds. Gain at full size per fold: "
           + ", ".join(f"{fo} {f(g, '+.4f')}" for fo, g in gains_full.items()) + f". Verdict: **{f1['verdict']}**."]

    p = f2["per_mouse"]
    r2 = ["![Fig. 2](../fig_cv_map.svg)", "",
          "**Fig. 2. The per-mouse predictability map (full-size models).** **a**, held-out likelihood per mouse, pass A vs "
          "pass B (two disjoint halves of the mouse's sessions). **b**, 3-way vs 2-way GRU per mouse (both passes). "
          "**c**, per-mouse held-out likelihood by fold (violin: distribution; bar: interquartile range; dot: median).", "",
          "| quantity | value |", "|---|---|",
          f"| mice | {f2['n_mice']} |",
          f"| per-mouse held-out likelihood, 5th / 50th / 95th percentile | {f(p['p5'])} / {f(p['median'])} / {f(p['p95'])} |",
          f"| per-mouse SD | {f(p['sd'])} |",
          f"| pass A vs pass B, Pearson r (n = {f2['n_mice']} mice) | {f(f2['passA_passB_r'], '.3f')} |",
          f"| 3-way vs 2-way, Pearson r (n = {f2['n_mice_2way']} mice) | {f(f2['threeway_twoway_r'], '.3f')} |"]

    r3 = ["![Fig. 3](../fig_cv_ignore_split.svg)", "",
          f"**Fig. 3. Where the 3-way GRU beats the 2-way GRU.** Full-size models, both passes, scored engaged trials of the "
          f"{f3['n_mice']} mice both models score (test fold: {len(models['test-2way-full-s0']['scores']['lam-checkpoint']['A']['mice'])} "
          f"of {len(models['test-3way-full-s0']['scores']['lam-checkpoint']['A']['mice'])}). "
          "**a**, per fold, 2-way and 3-way held-out likelihood (L/R). **b**, 3-way − 2-way by trials since the last "
          "ignore (dots: folds; bar: mean ± SEM, 5 folds). **c**, share of trials and share of the 3-way − 2-way log-likelihood "
          "difference summed over all folds and trials.", "",
          "| engaged trial | share of trials | 3-way − 2-way held-out likelihood (mean ± SEM, 5 folds) | share of the summed log-likelihood difference |", "|---|---|---|---|"]
    for k, lab in BIN_LABEL.items():
        b = ig["per_bin"][k]
        r3.append(f"| {lab} | {100 * b['share_of_trials']:.1f}% | {f(b['diff_mean'], '+.4f')} ± {sem(b['diff_sem'])} | "
                  f"{f(100 * b['share_of_loglik_gain'], '.0f')}% |")
    r3 += ["", f"All engaged trials: {f(f3['diff_mean'], '+.4f')} ± {sem(f3['diff_sem'])} (SEM, n = {f3['n_folds']} folds); "
           f"verdict: **{f3['verdict']}**.", "",
           "![Fig. 4](../fig_cv_2way_vs_s01.svg)", "",
           "**Fig. 4. Study 17's models on study 01's 2-way scaling curves.** Grey: study 01's 2-way GRU per hidden size N "
           "(test mice, pass A, mean ± SE over 3 seeds; read from study 01's `nxd_scaling.json`). Blue: study 17's 3-way curve; "
           "purple: study 17's full-size 2-way models (5 folds, both passes, mean ± SEM). Open square: study 01's N = 256, "
           "D = 614, seed-0 run re-scored by study 17 (pass A, 146 of its 149 test mice): "
           f"{f(f3['study17_test_2way_passA'], '.5f')} vs study 01's value for the same run, {f(f3['study01_same_run']['heldout_ll'], '.5f')} "
           f"(difference {f(f3['study17_minus_study01_same_run'], '+.1e')}; different mouse count)."]

    r4 = ["![Fig. 5](../fig_cv_fold_spread.svg)", "",
          "**Fig. 5. Fold-to-fold spread vs which-mice sampling.** **a**, the 5 fold values (full-size 3-way, both passes) and the "
          f"value over all {sp['n_mice']} mice pooled. **b**, SD of the 5 fold values when the same mice (each scored by its own "
          f"fold model) are reassigned to 5 random folds of the real sizes, {f4['n_perm']:,} times. **c**, SD of the 4 pool fold "
          f"values when the 614 pool mice are re-dealt within the {f4['stratified_pool']['n_strata']} strata the real folds were dealt in "
          "(tier × curriculum × session-count third), the same way. Black lines: real SD.", "",
          "| quantity | value |", "|---|---|",
          f"| SD of the 5 real fold values | {f(f4['real_sd'])} |",
          f"| SD under random splits: median (95% range) | {f(f4['perm_sd_median'])} ({f(f4['perm_sd_ci95'][0])}–{f(f4['perm_sd_ci95'][1])}) |",
          f"| P(random SD ≥ real SD) | {f(f4['p_sd_ge_real'], '.2f')} |",
          f"| SD of the 4 real pool fold values | {f(f4['stratified_pool']['real_sd'])} |",
          f"| SD under within-strata re-deals: median (95% range) | {f(f4['stratified_pool']['sd_median'])} "
          f"({f(f4['stratified_pool']['sd_ci95'][0])}–{f(f4['stratified_pool']['sd_ci95'][1])}) |",
          f"| P(re-deal SD ≥ real SD) | {f(f4['stratified_pool']['p_sd_ge_real'], '.2f')} |",
          f"| test fold − all mice | {f(f4['test_minus_all'], '+.4f')} |",
          f"| same, random 149-mouse sets: 95% range; P(abs. offset ≥ real) | {f(f4['test_offset_ci95'][0], '+.4f')} to "
          f"{f(f4['test_offset_ci95'][1], '+.4f')}; {f(f4['test_offset_p'], '.2f')} |",
          "", f"Verdicts: the fold spread is **{f4['verdict']}**; the test fold's offset is **{f4['test_offset_verdict']}**."]

    s1 = ["![Fig. S1](../fig_cv_crosscheck.svg)", "",
          "**Fig. S1. Study 17's pass-A scores vs study 02's frozen metrics.** **a–c**, study 02's 6 test-fold models (D = 10, 30, 100, "
          "full seeds 0–2) re-scored by study 17 on pass A vs study 02's frozen held-out metrics. **d**, pass B vs pass A.", "",
          "| metric | max abs difference vs study 02 | n models |", "|---|---|---|"]
    for k, lab in (("lr", "held-out likelihood, trial-pooled (L/R)"), ("ignore_pr_auc", "ignore PR-AUC"), ("ignore_recall", "ignore recall")):
        s1.append(f"| {lab} | {f(cc['summary'][k]['max_abs_diff'], '.1e')} | {cc['summary'][k]['n_models']} |")
    s1 += ["", f"Pass A − pass B, held-out likelihood: {f(cc['summary']['passA_minus_passB_lr_mean'], '+.4f')} (mean over the same models)."]

    s2 = ["![Fig. S2](../fig_cv_lambda.svg)", "",
          f"**Fig. S2. Session λ at scoring.** The {len(lam['rows'])} models scored both at their checkpoint's own λ (the study's rule) "
          "and at λ = 1 (the in-training fine-tune's value in studies 01/02). The other λ = 0 models were scored at their own λ only "
          "(Table S4).", "",
          "| model | checkpoint λ | own λ | λ = 1 | λ = 1 − own |", "|---|---|---|---|---|"]
    for r in lam["rows"]:
        s2.append(f"| {r['model']} | {r['checkpoint_lambda']:g} | {f(r['lik_own_lambda'], '.6f')} | {f(r['lik_lambda_1'], '.6f')} | "
                  f"{f(r['lambda1_minus_own'], '+.6f')} |")

    s3 = ["![Fig. S3](../fig_cv_rebuild_check.svg)", "",
          "**Fig. S3. A fine-tuned held-out model rebuilt from W&B alone vs its scoring run's logged predictions** (pool4-3way-full, "
          "pass A; wrapper PR #130). **a**, |rebuilt − logged p(right)| over every logged trial. **b**, rebuilt vs logged. "
          "**c**, rebuilt GRU hidden states of one held-out session (first two principal components).", "",
          "| check | value |", "|---|---|",
          f"| logged trials matched 1:1 | {rb['n_matched']:,} of {rb['n_logged']:,} |",
          f"| max abs difference, p(left/right/ignore) | {f(rb['max_abs_diff_p'], '.1e')} |",
          f"| readout of the states vs the logits, max abs error | {f(rb['readout_max_abs_error'], '.1e')} |"]

    s4 = ["**Table S4. Models scored** (`scores.json`; each scored in passes A and B).", "",
          "| model | D | way | seed | best_eval step | checkpoint λ | held-out mice | source run |", "|---|---|---|---|---|---|---|---|"]
    for name, m in sorted(models.items()):
        s4.append(f"| {name} | {m['D']} | {m['way']} | {m['seed']} | {m['best_eval_step']} | {m['checkpoint_lambda']:g} | "
                  f"{len(m['scores']['lam-checkpoint']['A']['mice'])} | `{m['source']}` |")
    missing = d["scores"].get("missing", [])
    lams = sorted({models[m.split("/")[0]]["checkpoint_lambda"] for m in missing})
    s4 += ["", f"Not scored: {len(missing)} λ = 1 jobs (cancelled), of checkpoints with λ = {', '.join(f'{x:g}' for x in lams)}"
           + (": at λ = 0 the session weights are zero, so λ cannot change the score (Fig. S2)." if lams == [0.0] else ".")]

    text = REPORT.read_text()
    for block, body in (("methods", methods), ("result-1", "\n".join(r1)), ("result-2", "\n".join(r2)), ("result-3", "\n".join(r3)),
                        ("result-4", "\n".join(r4)), ("supp-1", "\n".join(s1)), ("supp-2", "\n".join(s2)),
                        ("supp-3", "\n".join(s3)), ("supp-4", "\n".join(s4))):
        text = rewrite(text, block, body)
    REPORT.write_text(text)
    rel = "reports/r1-cv-predictability.md"
    blocks = ["methods", "result-1", "result-2", "result-3", "result-4", "supp-1", "supp-2", "supp-3", "supp-4"]
    meta = build_meta("analysis/r1_cv_predictability.py", WANDB_GROUPS, study_root=STUDY, code=[__file__],
                      inputs=list(SRC.values()))
    write_artifact(OUT, {"_meta": meta, **result}, outputs=[HERE / f"{rel}#{b}" for b in blocks])


if __name__ == "__main__":
    main()
