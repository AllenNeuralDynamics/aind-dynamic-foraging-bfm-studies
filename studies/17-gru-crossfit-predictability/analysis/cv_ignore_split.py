#!/usr/bin/env python
"""Where the 3-way GRU beats the 2-way GRU on held-out L/R prediction, from analysis/ignore_split.json.

a: per fold, held-out likelihood (trial-pooled, L/R engaged, both passes) of the full-size 2-way and
   3-way models, paired by a line.
b: 3-way minus 2-way held-out likelihood by trials since the last ignore (k), fold dots and
   mean +- SEM over the 5 folds.
c: share of scored engaged trials vs share of the total 3-way minus 2-way log-likelihood gain, by k.

    python analysis/cv_ignore_split.py   # writes analysis/fig_cv_ignore_split.svg (+ .json)
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

SRC = HERE / "ignore_split.json"
OUT_SVG = HERE / "fig_cv_ignore_split.svg"
OUT = HERE / "cv_ignore_split.json"
FOLDS = ["test", "pool_1", "pool_2", "pool_3", "pool_4"]
COLOR = {"test": fs.OKABE_ITO["black"], "pool_1": fs.OKABE_ITO["orange"], "pool_2": fs.OKABE_ITO["sky"],
         "pool_3": fs.OKABE_ITO["green"], "pool_4": fs.OKABE_ITO["vermillion"]}
BIN_LABEL = {"1": "1", "2-4": "2–4", "5-19": "5–19", "20+": "≥20", "none": "none\nyet"}


def lik(ll: float, n: int) -> float:
    return float(np.exp(ll / n))


def main() -> None:
    src = json.loads(SRC.read_text())
    bins, folds = src["bins"], src["folds"]
    out: dict = {"per_fold": {}, "per_bin": {}}
    for f in FOLDS:
        tot = {k: sum(folds[f][b][k] for b in bins) for k in ("n_trials", "ll_lr_sum_3way", "ll_lr_sum_2way")}
        out["per_fold"][f] = {"lik_3way": lik(tot["ll_lr_sum_3way"], tot["n_trials"]),
                              "lik_2way": lik(tot["ll_lr_sum_2way"], tot["n_trials"]), "n_trials": tot["n_trials"]}
        out["per_fold"][f]["diff"] = out["per_fold"][f]["lik_3way"] - out["per_fold"][f]["lik_2way"]
    diffs = np.array([out["per_fold"][f]["diff"] for f in FOLDS])
    out["overall"] = {"diff_mean": float(diffs.mean()), "diff_sem": float(diffs.std(ddof=1) / np.sqrt(len(diffs))),
                      "n_folds": len(diffs), "n_folds_3way_higher": int((diffs > 0).sum())}
    n_all = sum(folds[f][b]["n_trials"] for f in FOLDS for b in bins)
    gain_all = sum(folds[f][b]["ll_lr_sum_3way"] - folds[f][b]["ll_lr_sum_2way"] for f in FOLDS for b in bins)
    for b in bins:
        d = np.array([lik(folds[f][b]["ll_lr_sum_3way"], folds[f][b]["n_trials"])
                      - lik(folds[f][b]["ll_lr_sum_2way"], folds[f][b]["n_trials"]) for f in FOLDS])
        n_b = sum(folds[f][b]["n_trials"] for f in FOLDS)
        g_b = sum(folds[f][b]["ll_lr_sum_3way"] - folds[f][b]["ll_lr_sum_2way"] for f in FOLDS)
        out["per_bin"][b] = {"diff_per_fold": dict(zip(FOLDS, d.tolist())), "diff_mean": float(d.mean()),
                             "diff_sem": float(d.std(ddof=1) / np.sqrt(len(d))), "n_trials": n_b,
                             "share_of_trials": n_b / n_all, "share_of_loglik_gain": g_b / gain_all}

    fs.apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=fs.mm(150, 55), layout="constrained",
                             gridspec_kw={"width_ratios": [0.6, 1.2, 1.2]})
    ax = axes[0]
    for f in FOLDS:
        v = out["per_fold"][f]
        ax.plot([0, 1], [v["lik_2way"], v["lik_3way"]], "-o", color=COLOR[f], lw=0.7, ms=2.5,
                label="test fold" if f == "test" else f.replace("pool_", "pool fold "))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["2-way", "3-way"]); ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel("held-out likelihood, trial-pooled\n(L/R, engaged; full size, both passes)")
    o = out["overall"]
    ax.set_title(f"3-way − 2-way\n{fs.num(o['diff_mean'], '+.4f')} ± {o['diff_sem']:.4f}\n"
                 f"({o['n_folds_3way_higher']} of {o['n_folds']} folds higher)", fontsize=6)

    ax = axes[1]
    xs = np.arange(len(bins))
    for i, b in enumerate(bins):
        for j, f in enumerate(FOLDS):
            ax.plot(i + (j - 2) * 0.06, out["per_bin"][b]["diff_per_fold"][f], "o", color=COLOR[f], ms=2, alpha=0.8)
    ax.errorbar(xs, [out["per_bin"][b]["diff_mean"] for b in bins], yerr=[out["per_bin"][b]["diff_sem"] for b in bins],
                fmt="_", color="k", ms=8, capsize=2, lw=1)
    ax.axhline(0, color="0.6", lw=0.5, ls=":")
    ax.set_xticks(xs); ax.set_xticklabels([BIN_LABEL[b] for b in bins])
    ax.set_xlabel("trials since the last ignore (k)")
    ax.set_ylabel("3-way − 2-way held-out likelihood\n(trial-pooled, L/R)")
    ax.set_title("Gain by trials since the last ignore\n(dots = folds; bar = mean ± SEM, 5 folds)", fontsize=6)

    ax = axes[2]
    w = 0.38
    ax.bar(xs - w / 2, [out["per_bin"][b]["share_of_trials"] for b in bins], w, color="0.7", label="share of trials")
    ax.bar(xs + w / 2, [out["per_bin"][b]["share_of_loglik_gain"] for b in bins], w, color=fs.OKABE_ITO["blue"],
           label="share of 3-way − 2-way\nlog-likelihood gain")
    ax.axhline(0, color="0.3", lw=0.5)
    ax.set_xticks(xs); ax.set_xticklabels([BIN_LABEL[b] for b in bins])
    ax.set_xlabel("trials since the last ignore (k)")
    ax.set_ylabel("share (all folds, both passes)")
    ax.legend(frameon=False, fontsize=5, loc="upper right")
    ax.set_title(f"n = {n_all:,} scored engaged trials", fontsize=6)
    for a, letter in zip(axes, "abc"):
        fs.panel_label(a, letter)
    fs.save_figure(fig, OUT_SVG)
    meta = build_meta("analysis/cv_ignore_split.py", [], study_root=STUDY, code=[__file__], inputs=[SRC])
    write_artifact(OUT, {"_meta": meta, **out}, outputs=[OUT_SVG])


if __name__ == "__main__":
    main()
