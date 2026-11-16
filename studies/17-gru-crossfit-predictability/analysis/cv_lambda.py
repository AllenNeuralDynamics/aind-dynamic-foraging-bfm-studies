#!/usr/bin/env python
"""Session lambda at scoring: the checkpoint's own lambda vs lambda = 1, from analysis/scores.json.

Every model is scored at its best_eval checkpoint's own session-curriculum lambda (the study's rule).
The models whose checkpoint lambda is below 1 were also scored at lambda = 1, the value the wrapper's
in-training held-out fine-tune used (studies 01/02). Per model: held-out likelihood, trial-pooled
(L/R engaged, both passes) at each lambda, and the difference.

    python analysis/cv_lambda.py   # writes analysis/fig_cv_lambda.svg (+ .json)
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
OUT_SVG = HERE / "fig_cv_lambda.svg"
OUT = HERE / "cv_lambda.json"


def pooled(sc: dict) -> float:
    ll = sum(v[0] for p in "AB" for v in sc[p]["mice"].values())
    n = sum(v[1] for p in "AB" for v in sc[p]["mice"].values())
    return float(np.exp(ll / n))


def main() -> None:
    models = json.loads(SCORES.read_text())["models"]
    rows = []
    for name, m in sorted(models.items()):
        if "lam-1" not in m["scores"]:
            continue
        own, one = pooled(m["scores"]["lam-checkpoint"]), pooled(m["scores"]["lam-1"])
        rows.append({"model": name, "checkpoint_lambda": m["checkpoint_lambda"], "best_eval_step": m["best_eval_step"],
                     "lik_own_lambda": own, "lik_lambda_1": one, "lambda1_minus_own": one - own})
    by_lam = {}
    for lam in sorted({r["checkpoint_lambda"] for r in rows}):
        d = [r["lambda1_minus_own"] for r in rows if r["checkpoint_lambda"] == lam]
        by_lam[str(lam)] = {"n_models": len(d), "max_abs_diff": float(np.max(np.abs(d))), "min_diff": float(np.min(d))}

    fs.apply_paper_style()
    fig, ax = plt.subplots(figsize=fs.mm(fs.fit_width(groups=len(rows)), 50), layout="constrained")
    col = {0.0: "0.6", 0.5: fs.OKABE_ITO["vermillion"]}
    x = np.arange(len(rows))
    ax.bar(x, [r["lambda1_minus_own"] for r in rows], color=[col[r["checkpoint_lambda"]] for r in rows], width=0.6)
    ax.axhline(0, color="0.3", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"{r['model']}\nλ = {r['checkpoint_lambda']:g}" for r in rows], rotation=30, ha="right")
    ax.set_ylabel("λ = 1 minus own λ\n(held-out likelihood, trial-pooled)")
    ax.set_title("Scoring λ (both passes; bar colour = checkpoint λ)", fontsize=6)
    fs.save_figure(fig, OUT_SVG)
    meta = build_meta("analysis/cv_lambda.py", [], study_root=STUDY, code=[__file__], inputs=[SCORES])
    write_artifact(OUT, {"_meta": meta, "rows": rows, "by_checkpoint_lambda": by_lam}, outputs=[OUT_SVG])


if __name__ == "__main__":
    main()
