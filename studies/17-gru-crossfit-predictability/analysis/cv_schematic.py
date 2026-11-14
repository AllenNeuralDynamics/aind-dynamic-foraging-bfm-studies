#!/usr/bin/env python
"""Schematic of study 17's design: fold rotation, scaling subsets, and the two scoring passes.

A: the 763 mice in 5 folds (sizes from reference/folds.json); each fold model trains on every mouse
outside its fold (grey) and scores its own fold (colour). Model 0 is study 02's model.
B: within fold model k's training mice, nested subsets D = 10, 30, 100, 300 (sizes from
reference/scaling_subsets.json), each trained separately and scored on the same held-out fold.
C: one held-out mouse's sessions alternate fit / score (the freeze's roles); pass A tunes the
mouse's embedding on fit and scores score, pass B swaps, so every session is scored once.

    python analysis/cv_schematic.py   # writes analysis/fig_cv_schematic.svg (+ .json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
import figure_style as fs  # noqa: E402
from _meta import build_meta, write_artifact  # noqa: E402

FOLDS = STUDY / "reference" / "folds.json"
SUBSETS = STUDY / "reference" / "scaling_subsets.json"
OUT_SVG = HERE / "fig_cv_schematic.svg"
OUT = HERE / "cv_schematic.json"
ORDER = ["test", "pool_1", "pool_2", "pool_3", "pool_4"]
COLOR = {"test": fs.OKABE_ITO["black"], "pool_1": fs.OKABE_ITO["orange"], "pool_2": fs.OKABE_ITO["sky"],
         "pool_3": fs.OKABE_ITO["green"], "pool_4": fs.OKABE_ITO["vermillion"]}
GREY = "0.85"


def main() -> None:
    sizes = json.loads(FOLDS.read_text())["sizes"]
    sub = json.loads(SUBSETS.read_text())["sizes"]
    total = sum(sizes[f] for f in ORDER)

    fs.apply_paper_style()
    fig = plt.figure(figsize=fs.mm(183, 70))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.9, 1.0], wspace=0.35)

    # A: rotation
    ax = fig.add_subplot(gs[0]); ax.set_axis_off()
    x0 = 0.0
    starts = {}
    for f in ORDER:
        starts[f] = x0; x0 += sizes[f] / total
    ax.text(0, 1.02, f"{total} mice, 5 folds", fontsize=6, va="bottom")
    for f in ORDER:
        ax.add_patch(Rectangle((starts[f], 0.88), sizes[f] / total, 0.08, color=COLOR[f], lw=0))
        ax.text(starts[f] + sizes[f] / total / 2, 0.92, f"{'test' if f == 'test' else 'pool ' + f[-1]}\n{sizes[f]}",
                ha="center", va="center", fontsize=5, color="white")
    for i, held in enumerate(ORDER):
        y = 0.72 - i * 0.14
        for f in ORDER:
            ax.add_patch(Rectangle((starts[f], y), sizes[f] / total - 0.004, 0.09,
                                   color=COLOR[f] if f == held else GREY, lw=0))
        label = "model 0 (study 02, seed 0)" if held == "test" else f"fold model {i}"
        ax.text(-0.02, y + 0.045, label, ha="right", va="center", fontsize=5)
    ax.text(0.5, 0.0, "grey = trains on   colour = held out, scored", ha="center", fontsize=5, color="0.3")
    ax.set_xlim(-0.5, 1.0); ax.set_ylim(-0.05, 1.08)
    fs.panel_label(ax, "A")

    # B: nested subsets
    ax = fig.add_subplot(gs[1]); ax.set_axis_off()
    ds = [("d10", "D=10"), ("d30", "D=30"), ("d100", "D=100"), ("d300", "D=300")]
    full = 763 - sizes["pool_1"]
    ax.text(0, 1.02, "fold model k: nested training subsets", fontsize=6, va="bottom")
    for j, (k, lab) in enumerate(ds + [("full", "full")]):
        n = sub["pool_1"][k] if k != "full" else full
        y = 0.82 - j * 0.16
        ax.add_patch(Rectangle((0, y), n / full, 0.1, color="0.55", lw=0))
        ax.add_patch(Rectangle((n / full, y), 1 - n / full, 0.1, color=GREY, lw=0))
        ax.text(1.03, y + 0.05, lab if k in ("d10", "d30") else f"{lab} ({n} mice)" if k != "full" else f"full ({n} mice)",
                va="center", fontsize=5)
    ax.text(0, 0.0, "same held-out fold scored at every D\n(per curriculum, seed 0; pool fold 1 shown)",
            fontsize=5, color="0.3")
    ax.set_xlim(0, 1.5); ax.set_ylim(-0.05, 1.08)
    fs.panel_label(ax, "B")

    # C: passes
    ax = fig.add_subplot(gs[2]); ax.set_axis_off()
    ax.text(0, 1.02, "one held-out mouse: two passes", fontsize=6, va="bottom")
    roles = ["fit", "score"] * 4
    for r_i, (title, tuned) in enumerate([("pass A", "fit"), ("pass B", "score")]):
        y = 0.7 - r_i * 0.32
        ax.text(-0.02, y + 0.05, title, ha="right", va="center", fontsize=5.5)
        for s_i, role in enumerate(roles):
            tune = role == tuned
            ax.add_patch(Rectangle((s_i * 0.12, y), 0.1, 0.1, facecolor="white" if tune else "0.25",
                                   edgecolor="0.25", lw=0.5))
    ax.add_patch(Rectangle((0, 0.2), 0.06, 0.06, facecolor="white", edgecolor="0.25", lw=0.5))
    ax.text(0.08, 0.23, "tune embedding", va="center", fontsize=5)
    ax.add_patch(Rectangle((0.0, 0.1), 0.06, 0.06, facecolor="0.25", edgecolor="0.25", lw=0.5))
    ax.text(0.08, 0.13, "score", va="center", fontsize=5)
    for s_i, role in enumerate(roles):
        ax.text(s_i * 0.12 + 0.05, 0.86, role, ha="center", va="bottom", fontsize=5, color="0.3", rotation=90)
    ax.text(0, -0.04, "sessions by date; fit/score roles from\nthe data freeze; each session scored once", fontsize=5, color="0.3")
    ax.set_xlim(-0.22, 1.1); ax.set_ylim(-0.05, 1.08)
    fs.panel_label(ax, "C")

    fs.save_figure(fig, OUT_SVG)
    meta = build_meta("analysis/cv_schematic.py", [], study_root=STUDY, code=[__file__], inputs=[FOLDS, SUBSETS])
    write_artifact(OUT, {"_meta": meta, "fold_sizes": {f: sizes[f] for f in ORDER},
                         "subset_sizes_pool_1": sub["pool_1"]}, outputs=[OUT_SVG])


if __name__ == "__main__":
    main()
