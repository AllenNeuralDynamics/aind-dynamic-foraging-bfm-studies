#!/usr/bin/env python
"""Figure: a fine-tuned held-out GRU rebuilt from W&B artifacts reproduces its scoring run.

From analysis/rebuild_check.json (freeze_rebuild_check.py), pool4-3way-full pass A:
a: |rebuilt - logged p(right)| over every logged trial; b: 2,000 random trials, rebuilt vs logged;
c: the rebuilt per-trial GRU states of one scored session (first two principal components),
trials 900-1199, with the ignored trials marked.

    python analysis/cv_rebuild_check.py   # writes analysis/fig_cv_rebuild_check.svg (+ .json)
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

SRC = HERE / "rebuild_check.json"
OUT_SVG = HERE / "fig_cv_rebuild_check.svg"
OUT = HERE / "cv_rebuild_check.json"


def main() -> None:
    d = json.loads(SRC.read_text())
    r = d["wandb_only_rebuild"]
    h, smp, ses = d["abs_diff_p_right_hist"], d["sample_p_right"], d["session"]
    fs.apply_paper_style()
    fig, ax = plt.subplots(1, 3, figsize=fs.mm(150, 50), layout="constrained", gridspec_kw={"width_ratios": [1, 0.8, 1.4]})
    ax[0].stairs(h["counts"], h["edges"], fill=True, color="0.6")
    ax[0].set_xscale("log"); fs.plain_log_axis(ax[0], "x")
    ax[0].set_xticks([1e-12, 1e-10, 1e-8, 1e-6]); ax[0].set_xticklabels(["≤1e−12", "1e−10", "1e−8", "1e−6"])
    ax[0].set_xlabel("|rebuilt − logged p(right)|"); ax[0].set_ylabel("trials")
    ax[0].set_title(f"all {r['n_logged']:,} logged trials\nmax = {fs.num(r['max_abs_diff_p_right'], '.1e')}", fontsize=6)
    ax[1].scatter(smp["logged"], smp["rebuilt"], s=1.5, color="k", lw=0, alpha=0.5)
    ax[1].plot([0, 1], [0, 1], ":", color="0.6", lw=0.5); ax[1].set_aspect("equal")
    ax[1].set_xlabel("logged p(right), scoring run"); ax[1].set_ylabel("rebuilt p(right)")
    ax[1].set_title(f"{len(smp['logged']):,} random trials", fontsize=6)
    t = np.array(ses["trial"]); act = np.array(ses["action"])
    ax[2].plot(t, ses["pc1"], color=fs.OKABE_ITO["blue"], lw=0.6, label="state PC1")
    ax[2].plot(t, ses["pc2"], color=fs.OKABE_ITO["orange"], lw=0.6, label="state PC2")
    lo = min(min(ses["pc1"]), min(ses["pc2"]))
    ax[2].plot(t[act == 2], np.full((act == 2).sum(), lo * 1.15), "|", color="0.4", ms=3, label="ignored trial")
    ax[2].set_xlabel("trial"); ax[2].set_ylabel("GRU hidden state (PC score)")
    ax[2].set_title(f"held-out mouse {ses['subject_id']}, trials {ses['window'][0]}–{ses['window'][1] - 1}\n"
                    f"(PC1+2 = {100 * ses['pc12_variance_share']:.0f}% of the session's state variance)", fontsize=6)
    ax[2].legend(frameon=False, fontsize=5, loc="upper right", ncol=3)
    for a, letter in zip(ax, "abc"):
        fs.panel_label(a, letter)
    fs.save_figure(fig, OUT_SVG)
    meta = build_meta("analysis/cv_rebuild_check.py", [], study_root=STUDY, code=[__file__], inputs=[SRC])
    write_artifact(OUT, {"_meta": meta, "n_logged": r["n_logged"], "n_matched": r["n_matched"],
                         "max_abs_diff_p": max(r[k] for k in r if k.startswith("max_abs_diff_p")),
                         "readout_max_abs_error": r["readout_max_abs_error"]}, outputs=[OUT_SVG])


if __name__ == "__main__":
    main()
