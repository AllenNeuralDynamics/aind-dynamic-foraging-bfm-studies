"""C1 figure: counterfactual effects of one input in C3, with C's reward kernel beside it.

    python analysis/c1_figure.py

Offline: reads ``c1_impulse.json`` and Study 50's ``c1_impulse.json`` (C, same reward-flip
definition). Panels (rows = readout, side then engage):
  a. reward flip — C3 (and, in the side row, C in its own panel);
  b. engaged trial → ignore, split by whether the replaced trial was rewarded;
  c. ignored trial → unrewarded engaged choice.
Shading: ±1.96 SEM across edits. Writes ``fig_C1_impulse.png``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk02 as s13  # noqa: E402


def band(ax, k, s, color, label, ls="-"):
    mean, sem = np.asarray(s["mean"], float), np.asarray(s["sem"], float)
    ax.plot(k, mean, color=color, marker="o", ms=4, lw=2, ls=ls, label=label)
    ax.fill_between(k, mean - 1.96 * sem, mean + 1.96 * sem, color=color, alpha=0.18, lw=0)


def main() -> None:
    s13.apply_presentation_style()
    c3 = json.loads((s13.ANALYSIS / "c1_impulse.json").read_text())
    c = json.loads((s13.STUDY12 / "analysis" / "c1_impulse.json").read_text())
    k = np.asarray(c3["k"])
    fig, axes = plt.subplots(2, 4, figsize=(24, 11), sharex=True)
    titles = {"side": "side readout logit(R | engaged)", "engage": "engage readout logit(engaged / ignore)"}

    ax = axes[0, 0]
    band(ax, k, c3["reward_flip"]["side"], s13.ENGAGE_COLOR, "C3")
    ax.set(title="a. reward flip — C3\nc·(rewarded − unrewarded)", ylabel=titles["side"])
    ax = axes[0, 1]
    band(ax, np.asarray(c["k"]), c["all"], s13.MODEL_COLOR, "C (Study 50)")
    ax.set(title="a′. reward flip — C (Study 50)\nc·(rewarded − unrewarded)")
    lo = min(axes[0, 0].get_ylim()[0], axes[0, 1].get_ylim()[0])
    hi = max(axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1])
    for a in axes[0, :2]:
        a.set_ylim(lo, hi)
    ax = axes[1, 0]
    band(ax, k, c3["reward_flip"]["engage"], s13.ENGAGE_COLOR, "C3")
    ax.set(title="a. reward flip — C3 (rewarded − unrewarded)", ylabel=titles["engage"])
    axes[1, 1].axis("off")
    axes[1, 1].text(0.5, 0.5, "C has no\nengage readout", ha="center", va="center", fontsize=14, color="0.4",
                    transform=axes[1, 1].transAxes)

    for row, name in enumerate(("side", "engage")):
        ax = axes[row, 2]
        for label, color in (("was_rewarded", s13.RIGHT_COLOR), ("was_unrewarded", s13.LEFT_COLOR)):
            band(ax, k, c3["to_ignore_by_original_reward"][label][name], color, label.replace("_", " "))
        ax.set(title="b. engaged trial → ignore" + ("\n(signed by the original choice)" if name == "side" else ""))
        ax = axes[row, 3]
        band(ax, k, c3["to_engaged"][name], s13.IGNORE_COLOR, "ignore → unrewarded choice")
        ax.set(title="c. ignored trial → unrewarded choice" + ("\n(signed by the inserted side)" if name == "side" else ""))
    for a in axes.flat:
        if a.axison:
            a.axhline(0, color="0.75", lw=0.8)
            a.legend(frameon=False, fontsize=10)
    for a in axes[1]:
        if a.axison:
            a.set_xlabel("steps after the edited trial (k)")
    fig.suptitle("What one input does: counterfactual edits, later inputs as observed (20,000 edits each; ±1.96 SEM)",
                 fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(s13.figure_path("fig_C1_impulse.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
