#!/usr/bin/env python
"""Cross-check with study 02 (PLAN.md check 1, extended to every test-fold model).

The test-fold models scored here are study 02's own `nxd-3way` H256 runs (D=10, 30, 100, full;
full also seeds 1-2). Re-scored with study 17's tool on pass A (study 02's held-out half), they must
reproduce study 02's frozen held-out metrics. Plots study 17 (y) against study 02 (x) for held-out
left/right likelihood on engaged trials, ignore PR-AUC and ignore recall, one point per model, with
the identity line; a fourth panel shows what pass B adds (pass A vs pass B, same models).

    python analysis/cv_crosscheck.py   # writes analysis/fig_cv_crosscheck.svg (+ .json)
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
STUDY02 = STUDY.parent / "02-gru-scaling-law-ignore" / "analysis" / "frozen_runs.json"
OUT_SVG = HERE / "fig_cv_crosscheck.svg"
OUT = HERE / "cv_crosscheck.json"
METRICS = [("lr", "eval_likelihood_LR_engaged", "held-out L/R likelihood"),
           ("ignore_pr_auc", "engage_ignore_pr_auc", "ignore PR-AUC"),
           ("ignore_recall", "engage_ignore_recall", "ignore recall")]


def pooled_lr(mice: dict) -> float:
    return float(np.exp(sum(v[0] for v in mice.values()) / sum(v[1] for v in mice.values())))


def main() -> None:
    scores = json.loads(SCORES.read_text())["models"]
    s02 = {r["wandb_run_id"]: r for r in json.loads(STUDY02.read_text())["runs"]}
    rows = []
    for name, m in sorted(scores.items()):
        if m["heldout"] != "test" or m["way"] != "3way":
            continue
        rid = m["source"].rsplit("/", 1)[1]
        sc = m["scores"].get("lam-checkpoint", {})
        if rid not in s02 or "A" not in sc:
            continue
        r02 = s02[rid]
        row = {"model": name, "D": m["D"], "seed": m["seed"], "run_id": rid,
               "s17_lr": pooled_lr(sc["A"]["mice"]), "s02_lr": r02["eval_likelihood_LR_engaged"],
               "s17_ignore_pr_auc": sc["A"]["ignore_pr_auc"], "s02_ignore_pr_auc": r02["engage_ignore_pr_auc"],
               "s17_ignore_recall": sc["A"]["ignore_recall"], "s02_ignore_recall": r02["engage_ignore_recall"]}
        if "B" in sc:
            row["s17_lr_passB"] = pooled_lr(sc["B"]["mice"])
        rows.append(row)

    fs.apply_paper_style()
    fig, axes = plt.subplots(1, 4, figsize=fs.mm(183, 48), constrained_layout=True)
    summary = {}
    for ax, (k, k02, label) in zip(axes, METRICS):
        x = np.array([r[f"s02_{k}"] for r in rows]); y = np.array([r[f"s17_{k}"] for r in rows])
        ax.scatter(x, y, s=8, color="k", zorder=3)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max()); pad = (hi - lo) * 0.08
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.6", lw=0.5, ls=":")
        ax.set_xlabel(f"study 02: {label}"); ax.set_ylabel(f"study 17, pass A: {label}")
        d = np.abs(y - x)
        summary[k] = {"max_abs_diff": float(d.max()), "n_models": len(rows)}
        ax.set_title(f"{label}: max |Δ| = {d.max():.1e}", fontsize=6)
    ax = axes[3]
    pa = [r for r in rows if "s17_lr_passB" in r]
    xs = [r["s17_lr"] for r in pa]; ys = [r["s17_lr_passB"] for r in pa]
    ax.scatter(xs, ys, s=8, color=fs.OKABE_ITO["vermillion"], zorder=3)
    lo, hi = min(xs + ys), max(xs + ys); pad = (hi - lo) * 0.08
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.6", lw=0.5, ls=":")
    ax.set_xlabel("pass A (study 02's half)"); ax.set_ylabel("pass B (the other half)")
    gap = float(np.mean(np.array(xs) - np.array(ys))) if pa else None
    summary["passA_minus_passB_lr_mean"] = gap
    ax.set_title(f"Pass B is lower by {gap:.4f} on average" if gap is not None else "pass A vs B", fontsize=6)
    for ax, letter in zip(axes, "abcd"):
        fs.panel_label(ax, letter)
    fs.save_figure(fig, OUT_SVG)

    meta = build_meta("analysis/cv_crosscheck.py", [], study_root=STUDY, code=[__file__], inputs=[SCORES, STUDY02])
    write_artifact(OUT, {"_meta": meta, "rows": rows, "summary": summary}, outputs=[OUT_SVG])


if __name__ == "__main__":
    main()
