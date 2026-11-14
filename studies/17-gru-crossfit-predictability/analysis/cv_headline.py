#!/usr/bin/env python
"""Headline figure of the cross-validated scaling curve (draft for review), from analysis/scores.json.

Layout follows study 02's fig_scaling: x = number of training mice D (log), four panels:
held-out left/right likelihood on engaged trials, its within-fold gain over D=10 (paired: each fold
scores the same mice at every D, so the between-fold offset cancels), ignore-class PR-AUC,
ignore-class recall. Here the
replicates are the 5 folds (test + pool 1-4), not seeds: thin lines = folds, thick = mean +- SEM
over folds. Each fold's value pools all its held-out mice and both passes (every session scored
once). Dashed grey (panel A): study 02's published test-fold curve (pass A only, mean of 3 seeds),
read from its committed scaling.json. Models are scored at their best_eval checkpoint and its own
session lambda.

    python analysis/cv_headline.py   # writes analysis/fig_cv_headline.svg (+ .json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
import figure_style as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from _meta import build_meta, write_artifact  # noqa: E402

SCORES = HERE / "scores.json"
STUDY02 = STUDY.parent / "02-gru-scaling-law-ignore" / "analysis" / "scaling.json"
OUT_SVG = HERE / "fig_cv_headline.svg"
OUT_MAP = HERE / "fig_cv_map.svg"
TITLES = ["Held-out choice prediction vs D", "Ignore detection vs D", "Ignore recall vs D"]
OUT = HERE / "cv_headline.json"
DS = [10, 30, 100, 300, "full"]
X = {10: 10, 30: 30, 100: 100, 300: 300, "full": 610}
FOLDS = ["test", "pool_1", "pool_2", "pool_3", "pool_4"]
COLOR = {"test": fs.OKABE_ITO["black"], "pool_1": fs.OKABE_ITO["orange"], "pool_2": fs.OKABE_ITO["sky"],
         "pool_3": fs.OKABE_ITO["green"], "pool_4": fs.OKABE_ITO["vermillion"]}


def fold_value(model: dict) -> dict | None:
    """Pooled held-out metrics of one model over both passes (lambda = the checkpoint's own)."""
    passes = model["scores"].get("lam-checkpoint", {})
    if set(passes) != {"A", "B"}:
        return None
    lr = sum(v[0] for p in passes.values() for v in p["mice"].values())
    n = sum(v[1] for p in passes.values() for v in p["mice"].values())
    out = {"lr_engaged": float(np.exp(lr / n)), "n_mice": len(passes["A"]["mice"])}
    if "ignore_pr_auc" in passes["A"]:
        for k in ("ignore_pr_auc", "ignore_recall"):
            out[k] = float(np.mean([passes[p][k] for p in "AB"]))
    return out


def per_mouse(model: dict, passes: str = "AB") -> dict[str, float]:
    """Held-out L/R likelihood (engaged trials) per mouse, pooled over the given passes."""
    sc = model["scores"]["lam-checkpoint"]
    out = {}
    for sid in sc["A"]["mice"]:
        ll = sum(sc[p]["mice"][sid][0] for p in passes)
        n = sum(sc[p]["mice"][sid][1] for p in passes)
        out[sid] = float(np.exp(ll / n))
    return out


def draw_map(scores: dict) -> dict:
    """The predictability map at full size: pass A vs B per mouse, 3-way vs 2-way, spread by fold."""
    full3 = {m["heldout"]: m for m in scores.values() if m["D"] == "full" and m["way"] == "3way" and m["seed"] == 0}
    full2 = {m["heldout"]: m for m in scores.values() if m["D"] == "full" and m["way"] == "2way"}
    fig, axes = plt.subplots(1, 3, figsize=fs.mm(150, 52), constrained_layout=True)
    summary = {}
    a_all, b_all, x3, y2 = [], [], [], []
    for f in FOLDS:
        if f not in full3:
            continue
        a, b = per_mouse(full3[f], "A"), per_mouse(full3[f], "B")
        ids = sorted(a)
        axes[0].scatter([a[i] for i in ids], [b[i] for i in ids], s=2, color=COLOR[f], alpha=0.7, lw=0)
        a_all += [a[i] for i in ids]; b_all += [b[i] for i in ids]
        if f in full2:
            m3, m2 = per_mouse(full3[f]), per_mouse(full2[f])
            common = sorted(set(m3) & set(m2))
            axes[1].scatter([m3[i] for i in common], [m2[i] for i in common], s=2, color=COLOR[f], alpha=0.7, lw=0)
            x3 += [m3[i] for i in common]; y2 += [m2[i] for i in common]
    groups = [list(per_mouse(full3[f]).values()) for f in FOLDS if f in full3]
    fs.violins(axes[2], groups, [COLOR[f] for f in FOLDS if f in full3],
               labels=[("test" if f == "test" else f[-1]) for f in FOLDS if f in full3])
    lo, hi = 0.55, 0.95
    for ax in axes[:2]:
        ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.5, ls=":")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    if a_all:
        summary["passA_vs_passB_pearson_r"] = float(np.corrcoef(a_all, b_all)[0, 1])
        summary["n_mice_3way_full"] = len(a_all)
        axes[0].set_title(f"Pass A vs pass B per mouse (r = {summary['passA_vs_passB_pearson_r']:.2f})", fontsize=6)
    if x3:
        summary["3way_vs_2way_pearson_r"] = float(np.corrcoef(x3, y2)[0, 1])
        summary["n_mice_2way_vs_3way"] = len(x3)
        axes[1].set_title(f"3-way vs 2-way GRU per mouse (r = {summary['3way_vs_2way_pearson_r']:.2f})", fontsize=6)
    axes[0].set_xlabel("pass A: held-out L/R likelihood"); axes[0].set_ylabel("pass B")
    axes[1].set_xlabel("3-way GRU (both passes)"); axes[1].set_ylabel("2-way GRU (both passes)")
    axes[2].set_xlabel("fold"); axes[2].set_ylabel("per-mouse held-out L/R likelihood")
    axes[2].set_title("Spread of predictability across mice", fontsize=6)
    fs.save_figure(fig, OUT_MAP)
    return summary


def main() -> None:
    scores = json.loads(SCORES.read_text())["models"]
    table: dict = {f: {} for f in FOLDS}
    for name, m in scores.items():
        if m["way"] != "3way" or m.get("seed", 0) != 0:
            continue
        v = fold_value(m)
        if v is not None:
            table[m["heldout"]][str(m["D"])] = v
    s02 = json.loads(STUDY02.read_text())["grid"]
    ref = {d: s02[f"{d}x256"]["LR_engaged"]["mean"] for d in (10, 30, 100, 614)}

    fs.apply_paper_style()
    fig, axs = plt.subplots(1, 4, figsize=fs.mm(183, 52), constrained_layout=True)
    paired_ax, axes = axs[1], [axs[0], axs[2], axs[3]]
    panels = [("lr_engaged", "held-out L/R likelihood\n(engaged trials)"),
              ("ignore_pr_auc", "ignore-class PR-AUC (held-out)"),
              ("ignore_recall", "ignore-class recall (held-out)")]
    summary = {}
    for ax, (key, ylabel) in zip(axes, panels):
        for f in FOLDS:
            pts = [(X[d], table[f][str(d)][key]) for d in DS if str(d) in table[f] and key in table[f][str(d)]]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "-o", color=COLOR[f], lw=0.6, ms=1.8, alpha=0.6,
                        label=("test fold" if f == "test" else f.replace("pool_", "pool fold ")))
        means, sems, xs = [], [], []
        for d in DS:
            vals = [table[f][str(d)][key] for f in FOLDS if str(d) in table[f] and key in table[f][str(d)]]
            if len(vals) >= 2:
                xs.append(X[d]); means.append(np.mean(vals)); sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                summary.setdefault(key, {})[str(d)] = {"mean": means[-1], "sem": sems[-1], "n_folds": len(vals)}
        ax.errorbar(xs, means, yerr=sems, color="k", lw=1.4, capsize=1.5, zorder=5, label="mean ± SEM (folds)")
        if key == "lr_engaged":
            ax.plot([10, 30, 100, 614], [ref[d] for d in (10, 30, 100, 614)], "--", color="0.55", lw=0.8,
                    label="study 02, test fold\n(pass A, 3 seeds)")
        ax.set_xscale("log")
        ax.set_xticks([10, 30, 100, 300, 610])
        ax.set_xticklabels(["10", "30", "100", "300", "full\n(~610)"])
        ax.minorticks_off()
        ax.set_xlabel("# training mice (D)")
        ax.set_ylabel(ylabel)
    # Paired: each fold's gain over its own D=10 point. The same mice are scored at every D within a
    # fold, so the between-fold offset (which mice a fold holds) cancels; the error bar is the SEM of
    # within-fold differences, i.e. how consistently more training mice help.
    gains = {}
    for f in FOLDS:
        base = table[f].get("10", {}).get("lr_engaged")
        pts = [(X[d], table[f][str(d)]["lr_engaged"] - base) for d in DS if base is not None and str(d) in table[f]]
        if pts:
            xs_, ys_ = zip(*pts)
            paired_ax.plot(xs_, ys_, "-o", color=COLOR[f], lw=0.6, ms=1.8, alpha=0.6)
            gains[f] = dict(zip([str(d) for d in DS if str(d) in table[f]], ys_))
    pm, ps, px = [], [], []
    for d in DS:
        vals = [gains[f][str(d)] for f in gains if str(d) in gains[f]]
        if len(vals) >= 2:
            px.append(X[d]); pm.append(np.mean(vals)); ps.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            summary.setdefault("lr_engaged_gain_over_d10", {})[str(d)] = {"mean": pm[-1], "sem": ps[-1], "n_folds": len(vals)}
    paired_ax.errorbar(px, pm, yerr=ps, color="k", lw=1.4, capsize=1.5, zorder=5)
    paired_ax.axhline(0, color="0.6", lw=0.5, ls=":")
    paired_ax.set_xscale("log")
    paired_ax.set_xticks([10, 30, 100, 300, 610])
    paired_ax.set_xticklabels(["10", "30", "100", "300", "full\n(~610)"])
    paired_ax.minorticks_off()
    paired_ax.set_xlabel("# training mice (D)")
    paired_ax.set_ylabel("gain over D=10, same mice\n(held-out L/R likelihood)")
    paired_ax.set_title("Within-fold gain (paired)", fontsize=6)
    axes[0].legend(frameon=False, fontsize=5, loc="upper left")
    for ax, t in zip(axes, TITLES):
        ax.set_title(t, fontsize=6)
    fs.save_figure(fig, OUT_SVG)
    map_summary = draw_map(scores)

    meta = build_meta("analysis/cv_headline.py", [], study_root=STUDY, code=[__file__], inputs=[SCORES, STUDY02])
    write_artifact(OUT, {"_meta": meta, "per_fold": table, "summary": summary,
                         "study02_test_fold_lr_engaged": {str(k): v for k, v in ref.items()},
                         "map": map_summary}, outputs=[OUT_SVG, OUT_MAP])


if __name__ == "__main__":
    main()
