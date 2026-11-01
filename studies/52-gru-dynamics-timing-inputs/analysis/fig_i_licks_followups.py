"""Figures for the licks follow-ups, seeds as rows (offline; reads the stage-A npz/json files).

    python analysis/fig_i_licks_followups.py

  FL1  outcome-flip decomposition: reward input only / licks only / both, per original outcome.
  FL2  consistent-flip effect vs the fed trial's other-side licks.
  FL3  push vs gain for other-side licks: a(k), b(k) and the raw change by current preference.
Writes ``fig_FL_licks_followups.png`` and ``i_licks_followups_summary.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pk03 import ANALYSIS, CACHE, REWARDED_COLOR, UNREWARDED_COLOR, apply_presentation_style, figure_path, write_json  # noqa: E402
import _pk03  # noqa: E402

PRODUCER = "analysis/fig_i_licks_followups.py"
KEYS = ["L-s0", "L-s1"]
OUTCOMES = ((1.0, REWARDED_COLOR, "-", "fed trial rewarded"), (0.0, UNREWARDED_COLOR, "--", "fed trial unrewarded"))
FL1 = [("reward", "reward input only"), ("licks", "licks only"), ("both", "both (consistent)")]


def main() -> None:
    _pk03.use_keys(*KEYS)
    apply_presentation_style()
    fig, axes = plt.subplots(len(KEYS), 4, figsize=(24, 5.2 * len(KEYS)), gridspec_kw={"wspace": 0.32})
    summary = {}
    for i, key in enumerate(KEYS):
        d = np.load(_pk03._read_cache(CACHE / f"i_licks_followups_{key}.npz"))
        fits = _pk03.read_json(ANALYSIS / f"i_licks_followups_{key}.json")["FL3_fits_by_reward"]
        k = d["other_k"]
        summary[key] = {"FL1": {}, "FL2": {}}
        ax = axes[i, 0]
        width = 0.38
        for j, (r, color, _, label) in enumerate(OUTCOMES):
            m = d["reward"] == r
            med = [float(np.median(d[f"fl1_{n}"][m])) for n, _ in FL1]
            q = np.array([np.quantile(d[f"fl1_{n}"][m], [0.25, 0.75]) for n, _ in FL1]).T
            xs = np.arange(len(FL1)) + (j - 0.5) * width
            ax.bar(xs, med, width=width, color=color, label=label)
            ax.errorbar(xs, med, yerr=[np.array(med) - q[0], q[1] - np.array(med)], fmt="none", color="#222222", capsize=3)
            summary[key]["FL1"][label] = dict(zip([n for n, _ in FL1], med))
        ax.axhline(0, color="#888888", lw=1)
        ax.set_xticks(range(len(FL1)), [lab for _, lab in FL1], fontsize=12)
        ax.set_ylabel(f"{key}\nΔ toward switching")
        for r, color, ls, label in OUTCOMES:
            m = d["reward"] == r
            eff = d["s_k_flip"][:, m] - d["s_k"][:, m]
            q = np.quantile(eff, [0.25, 0.5, 0.75], axis=1)
            axes[i, 1].plot(k, q[1], color=color, ls=ls, lw=2, marker="o", ms=5, label=label)
            axes[i, 1].fill_between(k, q[0], q[2], color=color, alpha=0.15, lw=0)
            summary[key]["FL2"][label] = q[1]
            f = fits[str(int(r))]
            axes[i, 2].plot(k, f["a"], color=color, ls=ls, lw=2, marker="o", ms=5, label=f"a(k), {label}")
            axes[i, 2].plot(k, f["b"], color=color, ls=ls, lw=1.2, marker="s", ms=4, alpha=0.7, label=f"b(k), {label}")
            s0 = d["s_k"][0, m]
            for kk, marker in ((2, "o"), (6, "s")):
                jx = int(np.where(k == kk)[0][0])
                bins = pd.qcut(s0, 10, labels=False, duplicates="drop")
                g = pd.DataFrame({"x": s0, "y": d["s_k"][jx, m] - s0, "b": bins}).groupby("b")
                axes[i, 3].plot(g["x"].median(), g["y"].median(), color=color, ls=ls, lw=2, marker=marker, ms=5,
                                label=f"{kk} other-side licks, {label}")
        for ax in axes[i, 1:]:
            ax.axhline(0, color="#888888", lw=1)
        axes[i, 2].axhline(1, color="#888888", lw=1, ls=":")
        axes[i, 1].set_ylabel("consistent-flip effect")
        axes[i, 2].set_ylabel("a(k) push  /  b(k) gain")
        axes[i, 3].set_ylabel("Δ switch-logit vs 0 other-side licks")
    axes[0, 0].set_title("FL1  which input carries the outcome?\n(median; bars = IQR across states)", fontsize=13)
    axes[0, 1].set_title("FL2  outcome effect vs other-side licks\n(median; band = IQR across states)", fontsize=13)
    axes[0, 2].set_title("FL3  s_k = a(k) + b(k)·s₀")
    axes[0, 3].set_title("FL3  raw change by current preference")
    axes[-1, 1].set_xlabel("other-side licks on the fed trial")
    axes[-1, 2].set_xlabel("other-side licks k")
    axes[-1, 3].set_xlabel("s₀: switch-logit with 0 other-side licks")
    axes[0, 0].legend(frameon=False, fontsize=11)
    axes[0, 2].legend(frameon=False, fontsize=9)
    axes[0, 3].legend(frameon=False, fontsize=9)
    fig.suptitle("Licks model follow-ups (50k real states per seed; switch-signed logits)", y=1.0)
    fig.savefig(figure_path("fig_FL_licks_followups.png"), bbox_inches="tight")
    plt.close(fig)
    write_json("i_licks_followups_summary.json", summary, PRODUCER)


if __name__ == "__main__":
    main()
