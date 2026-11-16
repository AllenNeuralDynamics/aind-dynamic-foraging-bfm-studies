#!/usr/bin/env python
"""Is the spread across folds larger than which-mice sampling explains? From analysis/scores.json.

Full-size 3-way models (seed 0), held-out likelihood, trial-pooled (L/R engaged, both passes).
a: the 5 fold values and the value over all 763 mice pooled.
b: unstratified null: the 763 mice (each scored by its own fold model) are reassigned to 5 random
   folds of the real sizes, N_PERM times; SD of the 5 fold values, with the real SD. The same draws
   give the null of the test fold's offset (a random 149-mouse set minus all mice).
c: stratified null for the 4 pool folds, which freeze_folds.py dealt round-robin within strata
   (tier x curriculum x session-count third; reference/pool_strata.json): the 614 pool mice are
   re-dealt the same way with a new shuffle, N_PERM times; SD of the 4 pool fold values, with the real SD.

    python analysis/cv_fold_spread.py   # writes analysis/fig_cv_fold_spread.svg (+ .json)
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
import figure_style as fs  # noqa: E402
from _meta import build_meta, write_artifact  # noqa: E402

SCORES = HERE / "scores.json"
STRATA = STUDY / "reference" / "pool_strata.json"
OUT_SVG = HERE / "fig_cv_fold_spread.svg"
OUT = HERE / "cv_fold_spread.json"
MODELS = {"test": "test-3way-full-s0", **{f"pool_{k}": f"pool{k}-3way-full" for k in range(1, 5)}}
COLOR = {"test": fs.OKABE_ITO["black"], "pool_1": fs.OKABE_ITO["orange"], "pool_2": fs.OKABE_ITO["sky"],
         "pool_3": fs.OKABE_ITO["green"], "pool_4": fs.OKABE_ITO["vermillion"]}
N_PERM, SEED = 5000, 0


def main() -> None:
    models = json.loads(SCORES.read_text())["models"]
    ll, n, fold, ids = [], [], [], []
    for f, m in MODELS.items():
        sc = models[m]["scores"]["lam-checkpoint"]
        for sid in sc["A"]["mice"]:
            ll.append(sum(sc[p]["mice"][sid][0] for p in "AB")); n.append(sum(sc[p]["mice"][sid][1] for p in "AB"))
            fold.append(f); ids.append(sid)
    ll, n, fold = np.array(ll), np.array(n), np.array(fold)
    real = {f: float(np.exp(ll[fold == f].sum() / n[fold == f].sum())) for f in MODELS}
    pooled = float(np.exp(ll.sum() / n.sum()))
    real_sd = float(np.std(list(real.values()), ddof=1))
    sizes = [int((fold == f).sum()) for f in MODELS]
    rng = np.random.default_rng(SEED)
    sds = []
    for _ in range(N_PERM):
        perm, i, v = rng.permutation(len(ll)), 0, []
        for s in sizes:
            idx = perm[i:i + s]; i += s
            v.append(np.exp(ll[idx].sum() / n[idx].sum()))
        sds.append(np.std(v, ddof=1))
    sds = np.array(sds)
    # Test-fold offset under random 149-mouse sets (the first block of each unstratified draw).
    rng = np.random.default_rng(SEED + 1)
    offs = np.array([np.exp(ll[i].sum() / n[i].sum()) - pooled
                     for i in (rng.permutation(len(ll))[:sizes[0]] for _ in range(N_PERM))])
    test_off = real["test"] - pooled
    # Stratified null for the pool folds: re-deal within strata, round-robin, as freeze_folds.py did.
    strata = json.loads(STRATA.read_text())["strata"]
    idx = {sid: i for i, sid in enumerate(ids)}
    pool_ids = [s for s in ids if fold[idx[s]] != "test"]
    groups: dict[tuple, list[str]] = {}
    for sid in pool_ids:
        st = strata[sid]; groups.setdefault((st["tier"], st["curriculum"], st["third"]), []).append(sid)
    real_pool_sd = float(np.std([real[f] for f in MODELS if f != "test"], ddof=1))
    sds_strat = []
    for it in range(N_PERM):
        r = random.Random(10_000 + it); nxt = 0; members = {k: [] for k in range(4)}
        for key in sorted(groups):
            g = sorted(groups[key]); r.shuffle(g)
            for sid in g:
                members[nxt].append(idx[sid]); nxt = (nxt + 1) % 4
        sds_strat.append(np.std([np.exp(ll[m].sum() / n[m].sum()) for m in members.values()], ddof=1))
    sds_strat = np.array(sds_strat)
    out = {"fold_values": real, "all_mice_pooled": pooled, "n_mice": len(ll), "fold_sizes": dict(zip(MODELS, sizes)),
           "real_sd": real_sd, "perm": {"n": N_PERM, "seed": SEED, "sd_median": float(np.median(sds)),
                                        "sd_ci95": np.percentile(sds, [2.5, 97.5]).tolist(),
                                        "p_sd_ge_real": float(np.mean(sds >= real_sd))},
           "test_minus_all": test_off,
           "test_offset_null": {"n": N_PERM, "ci95": np.percentile(offs, [2.5, 97.5]).tolist(),
                                "p_abs_ge_real": float(np.mean(np.abs(offs) >= abs(test_off)))},
           "stratified_pool": {"real_sd": real_pool_sd, "n": N_PERM, "n_strata": len(groups),
                               "sd_median": float(np.median(sds_strat)), "sd_ci95": np.percentile(sds_strat, [2.5, 97.5]).tolist(),
                               "p_sd_ge_real": float(np.mean(sds_strat >= real_pool_sd))}}

    fs.apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=fs.mm(165, 52), layout="constrained", gridspec_kw={"width_ratios": [0.8, 1.1, 1.1]})
    ax = axes[0]
    for i, f in enumerate(MODELS):
        ax.plot(i, real[f], "o", color=COLOR[f], ms=4)
    ax.axhline(pooled, color="0.5", lw=0.7, ls="--")
    ax.text(5.1, pooled, "all 763 mice", ha="right", va="bottom", fontsize=5, color="0.4")
    ax.set_xticks(range(5)); ax.set_xticklabels(["test", "1", "2", "3", "4"]); ax.set_xlim(-0.5, 5.2)
    ax.set_xlabel("fold"); ax.set_ylabel("held-out likelihood, trial-pooled\n(L/R engaged; full size, both passes)")
    ax.set_title(f"fold SD = {real_sd:.4f}", fontsize=6)
    ax = axes[1]
    ax.hist(sds, bins=50, color="0.75")
    ax.axvline(real_sd, color="k", lw=1)
    ax.set_xlabel("SD of the 5 fold values"); ax.set_ylabel(f"random 5-way splits (n = {N_PERM})")
    ax.set_title(f"5 folds vs random splits (unstratified)\n(median {out['perm']['sd_median']:.4f}; "
                 f"P(random ≥ real) = {out['perm']['p_sd_ge_real']:.2f})", fontsize=6)
    ax = axes[2]; sp = out["stratified_pool"]
    ax.hist(sds_strat, bins=50, color="0.75")
    ax.axvline(real_pool_sd, color="k", lw=1)
    ax.set_xlabel("SD of the 4 pool fold values"); ax.set_ylabel(f"re-dealt within strata (n = {N_PERM})")
    ax.set_title(f"4 pool folds vs re-deals within strata\n(median {sp['sd_median']:.4f}; "
                 f"P(re-deal ≥ real) = {sp['p_sd_ge_real']:.2f})", fontsize=6)
    for a, letter in zip(axes, "abc"):
        fs.panel_label(a, letter)
    fs.save_figure(fig, OUT_SVG)
    meta = build_meta("analysis/cv_fold_spread.py", [], study_root=STUDY, code=[__file__], inputs=[SCORES, STRATA])
    write_artifact(OUT, {"_meta": meta, **out}, outputs=[OUT_SVG])


if __name__ == "__main__":
    main()
