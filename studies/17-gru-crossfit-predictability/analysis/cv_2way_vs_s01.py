#!/usr/bin/env python
"""Study 17's 2-way and 3-way GRUs against study 01's 2-way GRU scaling curves.

x = training mice D (log); y = held-out likelihood, trial-pooled, L/R on engaged trials.
- Study 01 (grey): 2-way GRU (`nxd` grid) per hidden size N, seed mean +- SE over 3 seeds, scored on
  the test mice, pass A only; read from study 01's committed analysis/nxd_scaling.json.
- Study 17, 2-way (purple): full-size models only, one per fold (dots at each fold's D),
  mean +- SEM over the 5 folds; both passes.
- Study 17, 3-way (blue): fold mean +- SEM over 5 folds at every D; both passes.
- Open purple square: study 17's test-fold 2-way model, pass A only, i.e. study 01's own N=256,
  D=614, seed-0 run re-scored (146 of study 01's 149 test mice).

    python analysis/cv_2way_vs_s01.py   # writes analysis/fig_cv_2way_vs_s01.svg (+ .json)
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
FOLDS_JSON = STUDY / "reference" / "folds.json"
S01 = STUDY.parent / "01-gru-scaling-law" / "analysis" / "nxd_scaling.json"
OUT_SVG = HERE / "fig_cv_2way_vs_s01.svg"
OUT = HERE / "cv_2way_vs_s01.json"
FOLDS = ["test", "pool_1", "pool_2", "pool_3", "pool_4"]
NAME3 = {"test": "test-3way-{d}", **{f"pool_{k}": f"pool{k}-3way-{{d}}" for k in range(1, 5)}}
NAME2 = {"test": "test-2way-full-s0", **{f"pool_{k}": f"pool{k}-2way-full" for k in range(1, 5)}}
PURPLE, BLUE = fs.OKABE_ITO["purple"], fs.OKABE_ITO["blue"]


def pooled(model: dict, passes: str = "AB") -> float:
    sc = model["scores"]["lam-checkpoint"]
    ll = sum(v[0] for p in passes for v in sc[p]["mice"].values())
    n = sum(v[1] for p in passes for v in sc[p]["mice"].values())
    return float(np.exp(ll / n))


def sem(v) -> float:
    return float(np.std(v, ddof=1) / np.sqrt(len(v)))


def main() -> None:
    models = json.loads(SCORES.read_text())["models"]
    sizes = json.loads(FOLDS_JSON.read_text())["sizes"]
    s01 = json.loads(S01.read_text())
    total = sum(sizes[f] for f in FOLDS)
    d_full = {f: (614 if f == "test" else total - sizes[f]) for f in FOLDS}   # test fold = study 02's 614 training mice

    two = {f: pooled(models[NAME2[f]]) for f in FOLDS}
    three = {}
    for d in ["d10", "d30", "d100", "d300", "full"]:
        vals = [pooled(models[NAME3[f].format(d=("full-s0" if f == "test" and d == "full" else d))]) for f in FOLDS]
        three[d] = {"mean": float(np.mean(vals)), "sem": sem(vals), "per_fold": dict(zip(FOLDS, vals))}
    xs3 = {"d10": 10, "d30": 30, "d100": 100, "d300": 300, "full": float(np.mean(list(d_full.values())))}
    test_2way_a = pooled(models[NAME2["test"]], "A")
    i256, i614 = s01["Ns"].index(256), s01["Ds"].index(614)
    out = {"study17_2way_full": {"per_fold": two, "mean": float(np.mean(list(two.values()))), "sem": sem(list(two.values())),
                                 "d_full": d_full},
           "study17_3way": three, "study17_test_2way_full_passA": test_2way_a,
           "study01_N256_D614_mean": s01["mean_grid"][i256][i614],
           "study01_source": "studies/01-gru-scaling-law/analysis/nxd_scaling.json"}

    fs.apply_paper_style()
    fig, ax = plt.subplots(figsize=fs.mm(100, 80), layout="constrained")
    greys = {16: "0.85", 64: "0.75", 128: "0.6", 256: "0.35", 512: "0.5"}
    for i, N in enumerate(s01["Ns"]):
        pts = [(D, m, s) for D, m, s in zip(s01["Ds"], s01["mean_grid"][i], s01["se_grid"][i]) if m is not None]
        D, m, s = map(np.array, zip(*pts))
        ax.errorbar(D, m, yerr=s, color=greys[N], lw=1.2 if N == 256 else 0.7, capsize=1, ms=2, marker="o",
                    label=f"study 01, 2-way, N = {N}")
        ax.text(D[-1] * 1.08, m[-1], f"N={N}", fontsize=5, color=greys[N], va="center")
    x3 = [xs3[d] for d in three]
    ax.errorbar(x3, [three[d]["mean"] for d in three], yerr=[three[d]["sem"] for d in three], color=BLUE, lw=1.4,
                capsize=1.5, marker="o", ms=2.5, label="study 17, 3-way, N = 256 (5 folds)")
    for f in FOLDS:
        ax.plot(d_full[f], two[f], "o", color=PURPLE, ms=2.5, alpha=0.6, mec="none")
    ax.errorbar([np.mean(list(d_full.values()))], [out["study17_2way_full"]["mean"]], yerr=[out["study17_2way_full"]["sem"]],
                color=PURPLE, marker="D", ms=3.5, capsize=2, lw=1.2, label="study 17, 2-way, N = 256, full only (5 folds)")
    ax.plot([614], [test_2way_a], "s", mfc="none", mec=PURPLE, ms=4, mew=0.8,
            label="study 17, 2-way test fold, pass A only\n(study 01's N = 256 seed 0, re-scored)")
    ax.set_xscale("log"); fs.plain_log_axis(ax, "x")
    ax.set_xticks([10, 30, 100, 300, 614]); ax.set_xlim(7, 1300)
    ax.set_xlabel("# training mice (D)")
    ax.set_ylabel("held-out likelihood, trial-pooled\n(L/R, engaged trials)")
    ax.set_title("Study 01: test mice, pass A, mean ± SE over 3 seeds.  Study 17: 5 folds, both passes, mean ± SEM.",
                 fontsize=5.5)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=5, loc="outside lower center", ncol=2)
    fs.save_figure(fig, OUT_SVG)
    meta = build_meta("analysis/cv_2way_vs_s01.py", [], study_root=STUDY, code=[__file__], inputs=[SCORES, FOLDS_JSON, S01])
    write_artifact(OUT, {"_meta": meta, **out}, outputs=[OUT_SVG])


if __name__ == "__main__":
    main()
