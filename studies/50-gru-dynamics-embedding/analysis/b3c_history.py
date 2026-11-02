"""B3c (deeper history): fig_B3c_prev_outcome split by the last two and the last three inputs.

    python analysis/b3c_history.py

Like ``fig_B3c_prev_outcome`` (one panel per previous choice × outcome), but conditioned on
the inputs of trials t−1, t−2 (16 panels) and t−1, t−2, t−3 (64 panels); trial t's state has
seen trial t−1. Each panel shows up to N_PER_PANEL trials with that history, drawn from the
full eval set and coloured by P(right), over a grey subsample of all trials. Layout:
  t−2: 4 × 4, row = t−1, column = t−2;
  t−3: 8 × 8 in four 4 × 4 blocks, one block per t−1; within a block row = t−2, column = t−3.
Writes ``fig_B3c_prev_outcome_t2[_pc1_pc4].png`` and ``fig_B3c_prev_outcome_t3[_pc1_pc4].png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402

N_BACKGROUND = 150_000
N_PER_PANEL = 4_000
SEED = 0
INPUTS = ["R+", "R−", "L+", "L−"]  # same order as fig_B3c_prev_outcome: R rewarded, R unrewarded, ...
INPUT_LONG = {"R+": "R rewarded", "R−": "R unrewarded", "L+": "L rewarded", "L−": "L unrewarded"}


def history_codes(trials, depth: int) -> np.ndarray:
    """[N, depth] index into INPUTS of the inputs at t−1 … t−depth (−1 where not available)."""
    g = trials.groupby("ses_idx", sort=False)
    out = np.full((len(trials), depth), -1)
    for k in range(1, depth + 1):
        a = g["action"].shift(k).to_numpy()
        r = (g["reward"].shift(k) > 0).to_numpy()
        ok = ~np.isnan(a)
        code = np.where(a == 1, np.where(r, 0, 1), np.where(r, 2, 3))
        out[ok, k - 1] = code[ok]
    return out


def panel(ax, scores, background, rows, p_right, i, j, title):
    ax.scatter(scores[background, i], scores[background, j], s=1, color="0.88", linewidths=0, rasterized=True)
    sc = ax.scatter(scores[rows, i], scores[rows, j], s=2, c=p_right[rows], cmap="coolwarm", vmin=0, vmax=1,
                    linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    return sc


def figure(scores, codes, p_right, background, depth, pcs, path, rng):
    i, j = pcs
    if depth == 2:
        n_rows = n_cols = 4
        cells = [((a, b), (a, b)) for a in range(4) for b in range(4)]  # (history, (row, col))
        size = (17, 16)
    else:
        n_rows = n_cols = 8
        cells = [((a, b, c), ((a // 2) * 4 + b, (a % 2) * 4 + c)) for a in range(4) for b in range(4) for c in range(4)]
        size = (30, 29)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=size, sharex=True, sharey=True)
    counts = {}
    for hist, (r, c) in cells:
        match = np.flatnonzero((codes[:, :depth] == np.array(hist)).all(axis=1))
        counts["/".join(INPUTS[h] for h in hist)] = len(match)
        rows = rng.choice(match, size=min(N_PER_PANEL, len(match)), replace=False) if len(match) else match
        title = "  ".join(f"t−{k + 1} {INPUTS[h]}" for k, h in enumerate(hist)) + f"\n(n={len(match):,})"
        sc = panel(axes[r, c], scores, background, rows, p_right, i, j, title)
    for ax in axes[-1]:
        ax.set_xlabel(f"PC{i + 1}", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"PC{j + 1}", fontsize=10)
    if depth == 3:
        fig.subplots_adjust(hspace=0.45, wspace=0.12, top=0.95)
        for a in range(4):
            r0, c0 = (a // 2) * 4, (a % 2) * 4
            p0, p1 = axes[r0, c0].get_position(), axes[r0 + 3, c0 + 3].get_position()
            fig.patches.append(plt.Rectangle((p0.x0 - 0.008, p1.y0 - 0.018), p1.x1 - p0.x0 + 0.016, p0.y1 - p1.y0 + 0.040,
                                             transform=fig.transFigure, fill=False, lw=1.5, ec="0.4"))
            fig.text((p0.x0 + p1.x1) / 2, p0.y1 + 0.026, f"t−1: {INPUT_LONG[INPUTS[a]]}", ha="center", va="bottom",
                     fontsize=16, weight="bold", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    else:
        fig.subplots_adjust(hspace=0.35, wspace=0.12)
    label = "previous three inputs" if depth == 3 else "previous two inputs"
    fig.suptitle(f"Hidden state (PC{i + 1} vs PC{j + 1}) by the {label}, coloured by P(right); + rewarded, − unrewarded",
                 fontsize=18, y=0.995 if depth == 3 else 0.94)
    fig.colorbar(sc, ax=axes, fraction=0.012, pad=0.03 if depth == 3 else 0.01, label="P(right)")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return counts


def main() -> None:
    s12.apply_presentation_style()
    trials = s12.load_trials(["ses_idx", "trial_idx", "action", "reward", "p_model1_right"])
    scores = s12.load_pca()["scores"]
    if len(scores) != len(trials):
        raise AssertionError("PCA scores and trial table are not row-aligned")
    codes = history_codes(trials, 3)
    p_right = trials["p_model1_right"].to_numpy()
    background = s12.subsample_rows(len(trials), N_BACKGROUND)
    for depth in (2, 3):
        for pcs, suffix in (((0, 1), ""), ((0, 3), "_pc1_pc4")):
            counts = figure(scores, codes, p_right, background, depth, pcs,
                            s12.figure_path(f"fig_B3c_prev_outcome_t{depth}{suffix}.png"), np.random.default_rng(SEED))
        print(depth, "smallest panels", sorted(counts.items(), key=lambda kv: kv[1])[:4])


if __name__ == "__main__":
    main()
