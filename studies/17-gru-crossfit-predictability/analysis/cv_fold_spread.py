#!/usr/bin/env python
"""Is the spread across folds larger than which-mice sampling explains? From analysis/scores.json.

Full-size 3-way models (seed 0), held-out likelihood, trial-pooled (L/R engaged, both passes).
a: the 5 fold values and the value over all 763 mice pooled.
b: permutation test: the 763 mice (each scored by its own fold model) are reassigned to 5 random
   folds of the real sizes, N_PERM times; histogram of the SD of the 5 fold values, with the real SD.

    python analysis/cv_fold_spread.py   # writes analysis/fig_cv_fold_spread.svg (+ .json)
"""
from __future__ import annotations

import json
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
OUT_SVG = HERE / "fig_cv_fold_spread.svg"
OUT = HERE / "cv_fold_spread.json"
MODELS = {"test": "test-3way-full-s0", **{f"pool_{k}": f"pool{k}-3way-full" for k in range(1, 5)}}
COLOR = {"test": fs.OKABE_ITO["black"], "pool_1": fs.OKABE_ITO["orange"], "pool_2": fs.OKABE_ITO["sky"],
         "pool_3": fs.OKABE_ITO["green"], "pool_4": fs.OKABE_ITO["vermillion"]}
N_PERM, SEED = 5000, 0


def main() -> None:
    models = json.loads(SCORES.read_text())["models"]
    ll, n, fold = [], [], []
    for f, m in MODELS.items():
        sc = models[m]["scores"]["lam-checkpoint"]
        for sid in sc["A"]["mice"]:
            ll.append(sum(sc[p]["mice"][sid][0] for p in "AB")); n.append(sum(sc[p]["mice"][sid][1] for p in "AB"))
            fold.append(f)
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
    out = {"fold_values": real, "all_mice_pooled": pooled, "n_mice": len(ll), "fold_sizes": dict(zip(MODELS, sizes)),
           "real_sd": real_sd, "perm": {"n": N_PERM, "seed": SEED, "sd_median": float(np.median(sds)),
                                        "sd_ci95": np.percentile(sds, [2.5, 97.5]).tolist(),
                                        "p_sd_ge_real": float(np.mean(sds >= real_sd))},
           "test_minus_all": real["test"] - pooled}

    fs.apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=fs.mm(110, 52), layout="constrained", gridspec_kw={"width_ratios": [0.8, 1.2]})
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
    ax.set_title(f"real SD vs random splits of the same mice\n(median {out['perm']['sd_median']:.4f}; "
                 f"P(random ≥ real) = {out['perm']['p_sd_ge_real']:.2f})", fontsize=6)
    for a, letter in zip(axes, "ab"):
        fs.panel_label(a, letter)
    fs.save_figure(fig, OUT_SVG)
    meta = build_meta("analysis/cv_fold_spread.py", [], study_root=STUDY, code=[__file__], inputs=[SCORES])
    write_artifact(OUT, {"_meta": meta, **out}, outputs=[OUT_SVG])


if __name__ == "__main__":
    main()
