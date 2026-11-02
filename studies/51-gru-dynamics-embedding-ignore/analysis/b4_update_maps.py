"""B4: update maps — each readout on step t+1 against step t, for each input received at t.

    python analysis/b4_update_maps.py

Row t's readouts are computed before trial t's outcome and row t+1's after it, so each
consecutive pair of timesteps shows what one input did. Rows of the figure:
  1. C3 side readout ``logit_diff`` = log P(R | engaged) / P(L | engaged), five inputs
     (L+, L−, R+, R−, ignore);
  2. C (Study 50) ``logit_diff``, four inputs (C never receives an ignore: its step t+1 is
     the next engaged trial);
  3. C3 engage readout ``engage_logit`` = log P(engaged) / P(ignore), five inputs.
Both models on their shared eval sessions. Density of pairs plus the binned mean (pooled;
per task in the JSON). Writes ``fig_B4_update_maps.png`` and ``b4_update_maps.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk02 as s13  # noqa: E402

SIDE_EDGES = np.linspace(-4, 4, 33)
ENGAGE_EDGES = np.linspace(-3, 7, 41)
MIN_PER_BIN = 200
INPUTS = ["L+", "L−", "R+", "R−", "ignore"]


def input_label(action: np.ndarray, reward: np.ndarray) -> np.ndarray:
    side = np.where(action == 1, "R", "L").astype(object) + np.where(reward > 0, "+", "−")
    return np.where(action == s13.IGNORE, "ignore", side)


def pairs(d: pd.DataFrame, column: str) -> pd.DataFrame:
    nxt = d.groupby("ses_idx", sort=False)[column].shift(-1)
    out = d.assign(next=nxt).dropna(subset=["next"])
    return out.assign(input=input_label(out["action"].to_numpy(), out["reward"].to_numpy()))


def binned_mean(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    which = np.digitize(x, edges) - 1
    centres = (edges[:-1] + edges[1:]) / 2
    means, counts = np.full(len(centres), np.nan), np.zeros(len(centres), dtype=int)
    for b in range(len(centres)):
        sel = which == b
        counts[b] = sel.sum()
        if counts[b] >= MIN_PER_BIN:
            means[b] = y[sel].mean()
    return centres, means, counts


def main() -> None:
    s13.apply_presentation_style()
    drop = s13.unshared_sessions()
    cols = ["ses_idx", "session_id", "task", "action", "reward"]
    c3 = s13.load_trials(cols + ["logit_diff", "choice_logit_0", "choice_logit_1", "choice_logit_2"])
    c3 = c3[~c3["session_id"].isin(drop["C3"])]
    c = s13.load_twin(cols + ["logit_diff"])
    c = c[~c["session_id"].isin(drop["C"])]

    rows = [("C3 side: logit(R | engaged)", pairs(c3, "logit_diff"), "logit_diff", SIDE_EDGES),
            ("C side: logit(R)", pairs(c, "logit_diff"), "logit_diff", SIDE_EDGES),
            ("C3 engage: logit(engaged / ignore)", pairs(c3, "engage_logit"), "engage_logit", ENGAGE_EDGES)]
    fig, axes = plt.subplots(len(rows), len(INPUTS), figsize=(5 * len(INPUTS), 5.2 * len(rows)))
    summary = {}
    for r, (title, d, column, edges) in enumerate(rows):
        summary[title] = {}
        for col, value in enumerate(INPUTS):
            ax = axes[r, col]
            sel = d[d["input"] == value]
            if sel.empty:
                ax.axis("off")
                ax.text(0.5, 0.5, "C never receives\nan ignore", ha="center", va="center", fontsize=13, color="0.4",
                        transform=ax.transAxes)
                continue
            x, y = sel[column].to_numpy(float), sel["next"].to_numpy(float)
            ax.hist2d(x, y, bins=[edges, edges], cmap="Greys", norm=matplotlib.colors.LogNorm(), rasterized=True)
            centres, means, counts = binned_mean(x, y, edges)
            ax.plot(centres, means, color="k", lw=2.6, label="all tasks")
            entry = {"n_pairs": int(len(sel)), "bin_centres": centres, "mean_next": means, "counts": counts, "by_task": {}}
            for task in s13.TASKS:
                t_sel = sel[sel["task"] == task]
                _, t_means, t_counts = binned_mean(t_sel[column].to_numpy(float), t_sel["next"].to_numpy(float), edges)
                ax.plot(centres, t_means, color=s13.TASK_COLORS[task], lw=1.4, label=task)
                entry["by_task"][task] = {"n_pairs": int(len(t_sel)), "mean_next": t_means, "counts": t_counts}
            ax.plot(edges, edges, color="0.5", lw=1, ls=":")
            ax.axhline(0, color="0.8", lw=0.8)
            ax.axvline(0, color="0.8", lw=0.8)
            ax.set_xlim(edges[0], edges[-1])
            ax.set_ylim(edges[0], edges[-1])
            ax.set_title(f"input at t: {value} (n={len(sel):,})", fontsize=13)
            ax.set_xlabel(f"{column} at t")
            if col == 0:
                ax.set_ylabel(f"{title}\n{column} at t+1", fontsize=14)
            summary[title][value] = entry
    axes[0, 0].legend(fontsize=9, frameon=False, loc="upper left")
    fig.suptitle("Update maps on the shared eval sessions (grey: density of consecutive-step pairs; dotted: no change)",
                 fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(s13.figure_path("fig_B4_update_maps.png"))
    plt.close(fig)
    s13.write_json("b4_update_maps.json", {
        "side_bin_edges": SIDE_EDGES, "engage_bin_edges": ENGAGE_EDGES, "min_pairs_per_bin": MIN_PER_BIN,
        "sessions": "shared eval sessions of C and C3 (split_alignment.json)", "rows": summary,
    }, "analysis/b4_update_maps.py", extra_groups=s13.twin_wandb_groups())


if __name__ == "__main__":
    main()
