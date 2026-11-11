"""B3h: the hidden state in PC1–PC2 and PC1–PC4, coloured by P(right) and by the t−1 input.

    python analysis/b3h_pc_overview.py

Rows: the GRU's P(right) on trial t; the four t−1 inputs (previous choice × outcome, the
input the state has just received). Columns: PC1 vs PC2, PC1 vs PC4 (A1's PCA). The same
N_SCATTER-trial subsample as B3 (``s12.subsample_rows`` seed 0); the input classes are drawn in
a shuffled order so none hides another. Each session's first trial (no previous input) is left
out of the bottom row.
Also ``fig_B3h_pc_overview_value.png``: the same two planes coloured by I2's value coordinate v
(I4 ``value_coordinate``; coolwarm centred on 0, red = favours right, full range).
Writes ``fig_B3h_pc_overview.png`` and ``fig_B3h_pc_overview_value.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402

N_SCATTER = 150_000  # as in B3
PAIRS = ((0, 1), (0, 3))
COLORS = {"R rewarded": "#0072B2", "R unrewarded": "#56B4E9", "L rewarded": "#D55E00", "L unrewarded": "#E69F00"}


def main() -> None:
    s12.apply_presentation_style()
    t = s12.load_trials(["p_model1_right", "prev_action", "prev_outcome"])
    scores = s12.load_pca()["scores"][:, :4]
    sub = s12.subsample_rows(len(t), N_SCATTER)
    t, x = t.iloc[sub], scores[sub]
    action = t["prev_action"].map({0.0: "L", 1.0: "R"})
    outcome = t["prev_outcome"].map({0.0: "unrewarded", 1.0: "rewarded"})
    label = (action + " " + outcome).where(action.notna() & outcome.notna()).to_numpy()
    order = np.random.default_rng(0).permutation(len(t))
    colour = np.array([COLORS.get(v, "none") for v in label], dtype=object)
    has = np.array([v in COLORS for v in label])

    fig, axes = plt.subplots(2, 2, figsize=(17, 14), layout="constrained")
    for col, (i, j) in enumerate(PAIRS):
        ax = axes[0, col]
        sc = ax.scatter(x[order, i], x[order, j], s=2, c=t["p_model1_right"].to_numpy()[order], cmap="coolwarm",
                        vmin=0, vmax=1, linewidths=0, rasterized=True)
        ax = axes[1, col]
        o = order[has[order]]
        ax.scatter(x[o, i], x[o, j], s=2, c=list(colour[o]), linewidths=0, alpha=0.6, rasterized=True)
        for ax in axes[:, col]:
            ax.set_xlabel(f"PC{i + 1}", fontsize=22)
            ax.set_ylabel(f"PC{j + 1}", fontsize=22)
            ax.tick_params(labelsize=18)
    cb = fig.colorbar(sc, cax=axes[0, 1].inset_axes([1.05, 0.0, 0.045, 1.0]))  # keys share one column beside the plots
    cb.set_label("P(right)", fontsize=21)
    cb.ax.tick_params(labelsize=18)
    handles = [Line2D([], [], marker="o", ls="none", ms=13, color=c, label=k) for k, c in COLORS.items()]
    axes[1, 1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.03, 0.5), frameon=False, fontsize=20,
                      title="t−1 input", title_fontsize=21, alignment="left")
    fig.savefig(s12.figure_path("fig_B3h_pc_overview.png"), bbox_inches="tight")
    plt.close(fig)

    from i4_distilled_model import value_coordinate  # noqa: PLC0415

    v = value_coordinate()[sub][order]
    lim = float(np.abs(v).max())
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.2), layout="constrained")
    for ax, (i, j) in zip(axes, PAIRS):
        sc = ax.scatter(x[order, i], x[order, j], s=2, c=v, cmap="coolwarm", vmin=-lim, vmax=lim, linewidths=0,
                        rasterized=True)
        ax.set_xlabel(f"PC{i + 1}", fontsize=22)
        ax.set_ylabel(f"PC{j + 1}", fontsize=22)
        ax.tick_params(labelsize=18)
    cb = fig.colorbar(sc, cax=axes[1].inset_axes([1.05, 0.0, 0.045, 1.0]))
    cb.set_label("value coordinate v (> 0 favours right)", fontsize=20)
    cb.ax.tick_params(labelsize=18)
    fig.savefig(s12.figure_path("fig_B3h_pc_overview_value.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
