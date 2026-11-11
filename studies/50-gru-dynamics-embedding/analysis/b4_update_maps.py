"""B4: update maps — the model's logit on trial t+1 against trial t, for each outcome of trial t.

    python analysis/b4_update_maps.py

``logit_diff`` = log-odds of choosing right. Row t's logit is computed before trial t's
outcome, row t+1's after it, so each (t, t+1) pair within a session shows what one
choice x reward did to the model's choice tendency. Panels: left/right x rewarded/not;
density of all pairs plus the binned mean, pooled and per task. Writes
``fig_B4_update_maps.png`` and ``b4_update_maps.json`` (binned means).
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
import _pk01 as s12  # noqa: E402

EDGES = np.linspace(-4, 4, 33)
MIN_PER_BIN = 200
PANELS = [(0, 1, "left, rewarded"), (1, 1, "right, rewarded"), (0, 0, "left, unrewarded"), (1, 0, "right, unrewarded")]


def pairs() -> pd.DataFrame:
    d = s12.load_trials(["ses_idx", "trial_idx", "task", "action", "reward", "logit_diff"])
    nxt = d.groupby("ses_idx", sort=False)["logit_diff"].shift(-1)
    d = d.assign(logit_next=nxt).dropna(subset=["logit_next"])
    return d.assign(reward=(d["reward"] > 0).astype(int))


def binned_mean(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    which = np.digitize(x, EDGES) - 1
    centres = (EDGES[:-1] + EDGES[1:]) / 2
    means, counts = np.full(len(centres), np.nan), np.zeros(len(centres), dtype=int)
    for b in range(len(centres)):
        sel = which == b
        counts[b] = sel.sum()
        if counts[b] >= MIN_PER_BIN:
            means[b] = y[sel].mean()
    return centres, means, counts


def main() -> None:
    s12.apply_presentation_style()
    d = pairs()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 12), sharex=True, sharey=True)
    summary = {}
    for ax, (choice, reward, label) in zip(axes.flat, PANELS):
        sel = d[(d["action"] == choice) & (d["reward"] == reward)]
        x, y = sel["logit_diff"].to_numpy(), sel["logit_next"].to_numpy()
        ax.hist2d(x, y, bins=[EDGES, EDGES], cmap="Greys", norm=matplotlib.colors.LogNorm(), rasterized=True)
        centres, means, counts = binned_mean(x, y)
        ax.plot(centres, means, color="k", lw=2.6, label="all tasks")
        entry = {"n_pairs": int(len(sel)), "bin_centres": centres, "mean_next_logit": means, "counts": counts,
                 "by_task": {}}
        for task in s12.TASKS:
            t_sel = sel[sel["task"] == task]
            _, t_means, t_counts = binned_mean(t_sel["logit_diff"].to_numpy(), t_sel["logit_next"].to_numpy())
            ax.plot(centres, t_means, color=s12.TASK_COLORS[task], lw=1.6, label=task)
            entry["by_task"][task] = {"n_pairs": int(len(t_sel)), "mean_next_logit": t_means, "counts": t_counts}
        ax.plot(EDGES, EDGES, color="0.5", lw=1, ls=":")
        ax.axhline(0, color="0.8", lw=0.8)
        ax.axvline(0, color="0.8", lw=0.8)
        ax.set_title(f"trial t: {label} (n={len(sel):,})", fontsize=13)
        summary[label] = entry
    for ax in axes[-1]:
        ax.set_xlabel("logit(right) on trial t")
    for ax in axes[:, 0]:
        ax.set_ylabel("logit(right) on trial t+1")
    axes.flat[0].legend(fontsize=9, frameon=False, loc="upper left")
    fig.suptitle("Update maps: what one outcome does to the model's choice log-odds "
                 "(grey: density of trial pairs; dotted: no change)", fontsize=13)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_B4_update_maps.png"))
    plt.close(fig)
    s12.write_json("b4_update_maps.json", {"bin_edges": EDGES, "min_pairs_per_bin": MIN_PER_BIN,
                                           "panels": summary}, "analysis/b4_update_maps.py")


if __name__ == "__main__":
    main()
