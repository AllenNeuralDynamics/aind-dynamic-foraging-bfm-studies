"""Study 40's procedure as a paper method figure (fig_procedure.svg): (a) the four cohorts,
(b) the search loop and what Claude sees, (c) simplification, pre-registration and the one test score.

Cohort sizes come from reference/cohorts_seed0.json; the training-pool and test counts from
analysis/explainers/explainer_refs.json. The gap-closed axis is a schematic: no score is drawn.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
import figure_style as fs  # noqa: E402

W, H = 183, 78                                   # mm
INK, GREY, LIGHT = "#222222", "#888888", "#F2F2F2"
COHORT = {"generation": fs.OKABE_ITO["orange"], "validation": fs.OKABE_ITO["green"],
          "selection": fs.OKABE_ITO["purple"], "test": INK}
CLAUDE = "#FBE9DA"                               # the proposer's fill


def box(ax, x, y, w, h, title, lines=(), edge=GREY, fill="white", lw=0.6, size=6, sub=5):
    """A rounded box at (x, y) lower-left, in mm, with a bold title and smaller lines below it."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=fill, ec=edge, lw=lw))
    n = len(lines)
    top = y + h / 2 + (n * 2.3) / 2
    ax.text(x + w / 2, top, title, ha="center", va="center", fontsize=size, fontweight="bold", color=INK)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, top - 2.6 - 2.3 * i, line, ha="center", va="center", fontsize=sub, color=INK)


def arrow(ax, a, b, color=GREY, lw=0.7, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=5, color=color, lw=lw,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def label(ax, x, y, text, color=INK, size=5, ha="center", **kw):
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=color, **kw)


def main() -> None:
    sizes = json.loads((STUDY / "reference/cohorts_seed0.json").read_text())["sizes"]
    refs = json.loads((HERE / "explainers/explainer_refs.json").read_text())

    fs.apply_paper_style()
    fig = plt.figure(figsize=fs.mm(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()

    for x, letter, title in ((2, "a", "Four cohorts"), (48, "b", "Search loop"), (128, "c", "Choose, then measure once")):
        ax.text(x, H - 3, letter, fontsize=8, fontweight="bold", va="center")
        ax.text(x + 4, H - 3, title, fontsize=7, va="center")

    # (a) cohorts ---------------------------------------------------------------------------
    box(ax, 4, 58, 38, 10, f"{refs['n_train_mice']} training mice", ["stratified draw, fixed seed"])
    rows = (("generation", "Generation", "feedback to Claude", 45),
            ("validation", "Validation", "the search's score", 34),
            ("selection", "Selection", "used once, to choose", 23))
    for key, name, role, y in rows:
        box(ax, 10, y, 32, 8.5, f"{name} · {sizes[key]}", [role], edge=COHORT[key], lw=1.0)
        arrow(ax, (7, y + 4.25), (10, y + 4.25))
    ax.plot([7, 7], [58, 27.25], color=GREY, lw=0.7)
    box(ax, 4, 7, 38, 11, f"Test · {refs['n_test_mice']}", ["held out from every search", "headline on 146 shared with GRU"],
        edge=COHORT["test"], lw=1.0)
    label(ax, 23, 3, "each mouse: fit even sessions, score odd", color=GREY)

    # (b) search loop -----------------------------------------------------------------------
    box(ax, 50, 46, 30, 18, "Program archive", ["5 islands, each seeded:", "template + 4 literature", "models; migration"])
    box(ax, 94, 46, 30, 18, "Claude proposes", ["a program + hypothesis:", "diff, rewrite or", "crossover of parents"],
        fill=CLAUDE, edge="#D9A066")
    box(ax, 72, 14, 30, 20, "Evaluator", ["static checks (≤ 12 params,", "≤ 6 latents)", "2-level EM fit per mouse",
                                         "held-out likelihood"])
    arrow(ax, (80, 58), (94, 58), color=INK)
    label(ax, 87, 61, "parent +\ninspirations")
    arrow(ax, (102, 46), (92, 34), color=INK)
    label(ax, 99.5, 41, "program", ha="left")
    arrow(ax, (102, 26), (118, 46), color=COHORT["generation"], lw=1.0, rad=0.35)
    label(ax, 117, 31, "feedback", color=COHORT["generation"], ha="left", fontweight="bold")
    label(ax, 51, 21, "stop: budget spent, or\n2,000 proposals with\nno gain", ha="left", color=GREY)
    arrow(ax, (75, 34), (63, 46), color=COHORT["validation"], lw=1.0)
    label(ax, 60, 38, "score\n(×100)", color=COHORT["validation"], fontweight="bold")

    # what Claude sees
    ax.add_patch(FancyBboxPatch((50, 1.5), 74, 10, boxstyle="round,pad=0,rounding_size=1", fc=LIGHT, ec="none"))
    label(ax, 52, 9, "Claude sees, per program:", ha="left", fontweight="bold")
    label(ax, 52, 6, "validation mice: one number, the score", ha="left", color=COHORT["validation"])
    label(ax, 52, 3.2, "generation mice: score by stage × task, switch-probability gaps, fitted parameters",
          ha="left", color=COHORT["generation"])

    # (c) choose, then measure -------------------------------------------------------------
    box(ax, 132, 56, 48, 9, "Simplify", ["keep ≥ 90, 75, 50% of the gain over literature"])
    box(ax, 132, 40, 48, 11, "Pre-register 4 programs", ["best of the top 5 on selection mice", "fingerprint gate vs. literature"],
        edge=COHORT["selection"], lw=1.0)
    box(ax, 132, 26, 48, 9, "Score once on test", ["nothing feeds back"], edge=COHORT["test"], lw=1.0)
    ax.plot([65, 65, 128], [64, 69.5, 69.5], color=INK, lw=0.7)
    arrow(ax, (128, 69.5), (132, 62), color=INK)
    label(ax, 96, 71.2, "elites")
    arrow(ax, (156, 56), (156, 51), color=INK)
    arrow(ax, (156, 40), (156, 35), color=INK)
    arrow(ax, (156, 26), (156, 21), color=INK)

    # gap closed, schematic axis from HB-CTT (0) to the GRU (1)
    x0, x1, y = 138, 174, 13
    ax.plot([x0, x1], [y, y], color=INK, lw=0.6)
    for x, name in ((x0, "HB-CTT"), (x1, "GRU")):
        ax.plot([x, x], [y - 1.2, y + 1.2], color=INK, lw=0.6)
        label(ax, x, y - 3.2, name)
    ax.annotate("", xy=(162, y + 0.2), xytext=(x0, y + 0.2),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4, mutation_scale=5))
    ax.plot([162], [y], "o", ms=3, color=INK)
    label(ax, 162, y + 3, "program")
    label(ax, 156, 18.6, "gap closed = (P − HB-CTT) / (GRU − HB-CTT)", fontweight="bold")
    label(ax, 156, 5.5, "best cognitive baseline → GRU ceiling", color=GREY)

    fs.save_figure(fig, HERE / "fig_procedure.svg")


if __name__ == "__main__":
    main()
