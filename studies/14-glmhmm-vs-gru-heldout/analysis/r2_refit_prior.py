"""Producer for r2: study 14 round 2, from ``frozen_round2.json`` and ``frozen_round1.json`` (offline).

Round 2 refits each mouse with its weight prior centred on the global fit (SD σ) and, optionally,
its transition prior centred on the global transition matrix (strength c), reusing round 1's
global fits. Round 1's zero-centred σ = 2 refit is the 17th arm, read from round 1's freeze.

Writes ``r2_refit_prior.json``, ``fig_r2_scores.svg``, ``fig_r2_grid.svg`` and the result blocks
of ``reports/r2-refit-prior.md``:
- result-1: selection, by the rule stated in the variant's notes before launch: K, σ, c and the
  weight centre chosen together by the training mice's refit score;
- result-2: held-out scores of the cells that rule picks, against no refit and study 02's 3-way
  GRU at D = 614 (read from its committed ``frozen_runs.json``).
"""
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
import figure_style as fs  # noqa: E402
from _meta import build_meta, write_artifact  # noqa: E402

WANDB_GROUPS = ["refit-prior@20261101-130226", "hurdle@20261027-212905", "reference@20261027-213006"]
FROZEN2 = HERE / "frozen_round2.json"
FROZEN1 = HERE / "frozen_round1.json"
GRU_SOURCE = HERE.parents[1] / "02-gru-scaling-law-ignore" / "analysis" / "frozen_runs.json"
GRU_GROUP = "nxd-3way@20260707-075401"
GRU_KEYS = {"matched": "eval_likelihood", "LR_engaged": "eval_likelihood_LR_engaged"}
REPORT = HERE / "reports" / "r2-refit-prior.md"
EMISSIONS = ("hurdle", "reference")
COLORS = {"hurdle": fs.OKABE_ITO["blue"], "reference": fs.OKABE_ITO["vermillion"]}
SIGMAS = (0.25, 0.5, 1.0, 2.0)
STRENGTHS = (None, 10.0, 100.0, 1000.0)
GRID_KS = (3, 6, 10)
SCORES = {   # label -> summary key
    "select": "glmhmm/select/eval_likelihood",
    "matched": "heldout/final/eval_likelihood",
    "zero_shot": "heldout/zero_shot/eval_likelihood",
    "LR_engaged": "heldout/final/eval_likelihood_LR_engaged",
    "LR_engaged_zero_shot": "heldout/zero_shot/eval_likelihood_LR_engaged",
    "engage": "heldout/final/eval_likelihood_engage",
    "pr_auc": "heldout/final/engage_ignore_pr_auc",
    "train": "heldout/train_likelihood",
}


def gru_d614():
    """Study 02's 3-way GRU at D = 614: mean over seeds, per hidden size."""
    runs = [r for r in json.loads(GRU_SOURCE.read_text())["runs"]
            if r["D"] == 614 and r["wandb_group"] == GRU_GROUP]
    out = {}
    for H in sorted({r["H"] for r in runs}):
        cell = [r for r in runs if r["H"] == H]
        out[H] = {name: float(np.mean([r[key] for r in cell])) for name, key in GRU_KEYS.items()}
        out[H]["n_seeds"] = len(cell)
    return out


def arm_label(centre, sigma, c):
    """A short name for one prior setting."""
    trans = "transitions free" if c is None else f"transitions c = {c:g}"
    return f"weights → {centre}, σ = {sigma:g}; {trans}"


def candidates(frozen1, frozen2):
    """Every scored cell, both rounds, as flat records."""
    out = []
    for cell in frozen2["cells"]:
        s = cell["summary"]
        out.append({"round": 2, "emission": cell["emission"], "K": cell["num_states"],
                    "centre": "global", "sigma": cell["prior_sigma_subject"],
                    "c": cell["transition_strength_subject"], "wandb_run_id": cell["wandb_run_id"],
                    **{name: s.get(key) for name, key in SCORES.items()}})
    for run in frozen1["runs"]:
        if run["group"].split("@")[0] not in EMISSIONS:
            continue
        s = run["summary"]
        out.append({"round": 1, "emission": run["emission"], "K": run["num_states"],
                    "centre": "zero", "sigma": 2.0, "c": None, "wandb_run_id": run["wandb_run_id"],
                    "global_same_mice": run["global_fit_scores"]["training_mice_eval"]["eval_likelihood"],
                    **{name: s.get(key) for name, key in SCORES.items()}})
    return out


def best(cells):
    """The cell with the highest selection score."""
    return max(cells, key=lambda r: r["select"])


def replace_block(text, n, body):
    """Replace the content between the result-n markers."""
    pattern = re.compile(rf"(<!-- BEGIN result-{n} -->\n).*?(\n<!-- END result-{n} -->)", re.S)
    if not pattern.search(text):
        raise ValueError(f"result-{n} markers missing")
    return pattern.sub(lambda m: m.group(1) + body.strip("\n") + m.group(2), text)


def f4(x):
    """Four decimals, or a dash."""
    return "—" if x is None else f"{x:.4f}"


def signed(x, scale=1e3):
    """A difference in thousandths, with a true minus."""
    return fs.num(x * scale, "+.1f")


def fig_scores(cands, by_k, round1, gru, path):
    """Selection and held-out scores against K: best centred cell per K, round 1, no refit."""
    fig, axes = plt.subplots(2, 2, figsize=fs.mm(150, 100), layout="constrained", sharex=True)
    for row, e in enumerate(EMISSIONS):
        Ks = sorted(by_k[e])
        r2 = [r for r in cands if r["emission"] == e and r["round"] == 2]
        ax = axes[row, 0]
        for sigma in SIGMAS:
            for c in STRENGTHS:
                arm = sorted((r for r in r2 if r["sigma"] == sigma and r["c"] == c), key=lambda r: r["K"])
                if arm:
                    ax.plot([r["K"] for r in arm], [r["select"] for r in arm], color="0.85", lw=0.5)
        ax.plot(Ks, [by_k[e][K]["select"] for K in Ks], "o-", color=COLORS[e], ms=3,
                label="best prior per K (round 2)")
        ax.plot(Ks, [round1[e][K]["select"] for K in Ks], "o--", color="0.3", ms=2,
                label="round 1 (weights → 0, σ = 2)")
        ax.plot(Ks, [round1[e][K]["global_same_mice"] for K in Ks], "s:", color="0.5", ms=2,
                label="global fit only (Faeze's score)")
        ax.set(ylabel=f"{e}\nselection score (training mice)")
        ax = axes[row, 1]
        ax.plot(Ks, [by_k[e][K]["matched"] for K in Ks], "o-", color=COLORS[e], ms=3,
                label="refit, prior picked per K")
        ax.plot(Ks, [round1[e][K]["matched"] for K in Ks], "o--", color="0.3", ms=2, label="round 1 refit")
        ax.plot(Ks, [round1[e][K]["zero_shot"] for K in Ks], "s:", color="0.5", ms=2,
                label="global fit only")
        for H, ls in ((128, ":"), (256, "-.")):
            if H in gru:
                ax.axhline(gru[H]["matched"], color="0.3", ls=ls, lw=0.8, label=f"GRU H{H}")
        ax.set(ylabel="held-out score (149 mice)")
    for ax in axes[-1]:
        ax.set(xlabel="states K", xticks=range(1, 11))
    axes[0, 0].legend(frameon=False, fontsize=5)
    axes[0, 1].legend(frameon=False, fontsize=5)
    for ax, letter in zip(axes.flat, "abcd"):
        fs.panel_label(ax, letter)
    fs.save_figure(fig, path)
    plt.close(fig)


def fig_grid(cands, path):
    """Selection score over σ × c at a few K, relative to the best prior at that K."""
    fig, axes = plt.subplots(len(EMISSIONS), len(GRID_KS), figsize=fs.mm(150, 85), layout="constrained")
    im = None
    for row, e in enumerate(EMISSIONS):
        for col, K in enumerate(GRID_KS):
            ax = axes[row, col]
            cells = {(r["sigma"], r["c"]): r["select"] for r in cands
                     if r["emission"] == e and r["K"] == K and r["round"] == 2}
            if not cells:
                ax.set_axis_off()
                continue
            top = max(cells.values())
            M = np.full((len(SIGMAS), len(STRENGTHS)), np.nan)
            for i, sigma in enumerate(SIGMAS):
                for j, c in enumerate(STRENGTHS):
                    if (sigma, c) in cells:
                        M[i, j] = (cells[(sigma, c)] - top) * 1e3
            im = ax.imshow(M, cmap="viridis", vmin=-3, vmax=0, aspect="auto")
            for i in range(len(SIGMAS)):
                for j in range(len(STRENGTHS)):
                    if np.isfinite(M[i, j]):
                        ax.text(j, i, fs.num(M[i, j], ".1f"), ha="center", va="center", fontsize=5,
                                color="white" if M[i, j] < -1.5 else "black")
            ax.set_xticks(range(len(STRENGTHS)), ["free" if c is None else f"{c:g}" for c in STRENGTHS])
            ax.set_yticks(range(len(SIGMAS)), [f"{s:g}" for s in SIGMAS])
            ax.set(title=f"{e}, K = {K}")
            if col == 0:
                ax.set_ylabel("weight prior σ")
            if row == len(EMISSIONS) - 1:
                ax.set_xlabel("transition prior c")
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, label="selection score − best at this K (×1000)")
    fs.save_figure(fig, path)
    plt.close(fig)


def block_selection(selected, by_k, round1, missing):
    """result-1: the cell the pre-stated rule picks, and the best prior at every K."""
    lines = []
    if missing:
        lines += [f"**Incomplete grid:** {len(missing)} of 320 round-2 cells missing from the freeze.", ""]
    for e in EMISSIONS:
        sel = selected[e]
        r1_pick = max(round1[e], key=lambda K: round1[e][K]["select"])
        lines += [f"**{e}: the rule picks K = {sel['K']}, {arm_label(sel['centre'], sel['sigma'], sel['c'])}** "
                  f"(selection score {f4(sel['select'])}; round 1 picked K = {r1_pick}, "
                  f"{f4(round1[e][r1_pick]['select'])}).", "",
                  "| K | best prior (by selection score) | selection | − round 1 (×1000) | global fit only |",
                  "|---|---|---|---|---|"]
        for K in sorted(by_k[e]):
            b = by_k[e][K]
            mark = "**" if K == sel["K"] else ""
            lines.append(f"| {mark}{K}{mark} | {arm_label(b['centre'], b['sigma'], b['c'])} | {f4(b['select'])} | "
                         f"{signed(b['select'] - round1[e][K]['select'])} | {f4(round1[e][K]['global_same_mice'])} |")
        lines.append("")
    lines += ["![scores](../fig_r2_scores.svg)", "", "Selection score over the prior grid at K = 3, 6 and 10, "
              "relative to the best prior at that K:", "", "![grid](../fig_r2_grid.svg)"]
    return "\n".join(lines)


def block_heldout(by_k, round1, gru):
    """result-2: held-out scores of the cell picked at each K, against no refit and the GRU."""
    lines = ["Held-out scores (149 mice, eval sessions) of the prior the selection score picks at each "
             "K. *Refit − global* > 0 means the refit beats the global fit alone.", ""]
    for e in EMISSIONS:
        lines += [f"**{e}**", "",
                  "| K | refit (round 2) | round 1 refit | global fit only | refit − global (×1000) | "
                  "LR engaged (round 2) |",
                  "|---|---|---|---|---|---|"]
        wins = 0
        for K in sorted(by_k[e]):
            b, r1 = by_k[e][K], round1[e][K]
            wins += b["matched"] > r1["zero_shot"]
            lines.append(f"| {K} | {f4(b['matched'])} | {f4(r1['matched'])} | {f4(r1['zero_shot'])} | "
                         f"{signed(b['matched'] - r1['zero_shot'])} | {f4(b['LR_engaged'])} |")
        lines += ["", f"The picked refit beats the global fit alone at {wins} of {len(by_k[e])} K.", ""]
    lines.append("Study 02's 3-way GRU at D = 614 (mean of 3 seeds): " + "; ".join(
        f"H{H} {f4(g['matched'])} (LR engaged {f4(g['LR_engaged'])})" for H, g in gru.items()) + ".")
    return "\n".join(lines)


def strip(r):
    """A record without its empty fields, for the JSON (c stays: None means transitions free)."""
    return {k: v for k, v in r.items() if v is not None or k == "c"}


def main() -> None:
    frozen2 = json.loads(FROZEN2.read_text())
    frozen1 = json.loads(FROZEN1.read_text())
    cands = candidates(frozen1, frozen2)
    gru = gru_d614()
    round1 = {e: {r["K"]: r for r in cands if r["round"] == 1 and r["emission"] == e} for e in EMISSIONS}
    by_k = {e: {K: best([r for r in cands if r["emission"] == e and r["K"] == K])
                for K in sorted({r["K"] for r in cands if r["emission"] == e})} for e in EMISSIONS}
    selected = {e: best([r for r in cands if r["emission"] == e]) for e in EMISSIONS}
    missing = frozen2["checks"]["missing"]

    fs.apply_paper_style()
    figs = [HERE / "fig_r2_scores.svg", HERE / "fig_r2_grid.svg"]
    fig_scores(cands, by_k, round1, gru, figs[0])
    fig_grid(cands, figs[1])

    text = REPORT.read_text()
    text = replace_block(text, 1, block_selection(selected, by_k, round1, missing))
    text = replace_block(text, 2, block_heldout(by_k, round1, gru))
    REPORT.write_text(text)

    result = {
        "selected": {e: strip(r) for e, r in selected.items()},
        "best_by_k": {e: {K: strip(r) for K, r in g.items()} for e, g in by_k.items()},
        "refit_beats_global_fit": {e: {K: bool(by_k[e][K]["matched"] > round1[e][K]["zero_shot"])
                                       for K in by_k[e]} for e in EMISSIONS},
        "gru_d614": gru,
        "n_round2_cells": len(frozen2["cells"]),
        "missing_round2_cells": missing,
        "cells": [strip(r) for r in cands],
    }
    meta = build_meta("analysis/r2_refit_prior.py", WANDB_GROUPS, study_root=HERE.parent,
                      code=[__file__], inputs=[FROZEN2, FROZEN1, GRU_SOURCE])
    write_artifact(HERE / "r2_refit_prior.json", {"_meta": meta, **result},
                   outputs=figs + [f"{REPORT}#result-{n}" for n in (1, 2)])


if __name__ == "__main__":
    main()
