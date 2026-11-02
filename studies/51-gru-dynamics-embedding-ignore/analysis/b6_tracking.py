"""B6: tracking — choice against the true reward probabilities, model vs animal, per task.

    python analysis/b6_tracking.py

Rows of the figure (one column per task), on the shared eval sessions of C and C3:
  1. C3: P(right | engaged) and the animal's choice on engaged trials vs
     P(reward | right) − P(reward | left);
  2. C (Study 50): P(right) and the animal's choice vs the same difference;
  3. C3: P(ignore) and the observed ignore rate vs the environment's richness
     P(reward | right) + P(reward | left).
As in Study 50, a bold set uses only trials at least STEADY trials after the last change of
either side's probability. Writes ``fig_B6_tracking.png`` and ``b6_tracking.json``.
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

DIFF_EDGES = np.linspace(-1, 1, 21)
SUM_EDGES = np.linspace(0, 1.2, 25)
MIN_PER_BIN = 200
STEADY = 10


def since_change(d: pd.DataFrame) -> np.ndarray:
    g = d.groupby("ses_idx", sort=False)
    changed = ((d["reward_probabilityL"] != g["reward_probabilityL"].shift())
               | (d["reward_probabilityR"] != g["reward_probabilityR"].shift()))
    return d.groupby(changed.cumsum()).cumcount().to_numpy()


def binned(x: np.ndarray, model: np.ndarray, animal: np.ndarray, edges: np.ndarray) -> dict:
    which = np.digitize(x, edges) - 1
    out = {"model": [], "animal": [], "n": []}
    for b in range(len(edges) - 1):
        sel = which == b
        n = int(sel.sum())
        out["n"].append(n)
        out["model"].append(float(model[sel].mean()) if n >= MIN_PER_BIN else np.nan)
        out["animal"].append(float(animal[sel].mean()) if n >= MIN_PER_BIN else np.nan)
    return out


def panel(ax, centres, rows, color, task) -> None:
    ax.plot(centres, rows["all"]["animal"], color="0.7", marker="x", ms=6, lw=0.8, label="animal, all trials")
    ax.plot(centres, rows["all"]["model"], color=color, marker="o", ms=3, lw=0.8, alpha=0.5, label="model, all trials")
    ax.plot(centres, rows["steady"]["animal"], color="0.3", marker="x", ms=8, lw=1.4,
            label=f"animal, ≥{STEADY} trials after a change")
    ax.plot(centres, rows["steady"]["model"], color=color, marker="o", ms=6, lw=2.4,
            label=f"model, ≥{STEADY} trials after a change")


def main() -> None:
    s13.apply_presentation_style()
    drop = s13.unshared_sessions()
    cols = ["ses_idx", "session_id", "task", "action", "reward_probabilityL", "reward_probabilityR"]
    c3 = s13.load_trials(cols + ["p_model1_left", "p_model1_right", "p_model1_ignore"])
    c3 = c3[~c3["session_id"].isin(drop["C3"])].reset_index(drop=True)
    c = s13.load_twin(cols + ["p_model1_right"])
    c = c[~c["session_id"].isin(drop["C"])].reset_index(drop=True)
    for d in (c3, c):
        d["since_change"] = since_change(d)
        d["dp"] = d["reward_probabilityR"] - d["reward_probabilityL"]
        d["richness"] = d["reward_probabilityR"] + d["reward_probabilityL"]

    diff_centres = (DIFF_EDGES[:-1] + DIFF_EDGES[1:]) / 2
    sum_centres = (SUM_EDGES[:-1] + SUM_EDGES[1:]) / 2
    fig, axes = plt.subplots(3, len(s13.TASKS), figsize=(5 * len(s13.TASKS), 15.5))
    summary = {}
    for col, task in enumerate(s13.TASKS):
        t3 = c3[c3["task"] == task]
        e3 = t3[t3["action"] != s13.IGNORE]
        t2 = c[c["task"] == task]
        entry = {}
        for row, (name, d, x, model, animal, edges) in enumerate((
                ("C3_side", e3, "dp", e3["p_right_engaged"], e3["action"] == 1, DIFF_EDGES),
                ("C_side", t2, "dp", t2["p_model1_right"], t2["action"] == 1, DIFF_EDGES),
                ("C3_engagement", t3, "richness", t3["p_model1_ignore"], t3["action"] == s13.IGNORE, SUM_EDGES))):
            steady = (d["since_change"] >= STEADY).to_numpy()
            xv, mv, av = d[x].to_numpy(), model.to_numpy(float), animal.to_numpy(float)
            rows = {"all": binned(xv, mv, av, edges), "steady": binned(xv[steady], mv[steady], av[steady], edges)}
            entry[name] = {"n": int(len(d)), **rows}
            ax = axes[row, col]
            panel(ax, diff_centres if x == "dp" else sum_centres, rows, s13.TASK_COLORS[task], task)
            if x == "dp":
                ax.axhline(0.5, color="0.85", lw=0.8)
                ax.axvline(0, color="0.85", lw=0.8)
                ax.set_xlabel("P(rew | R) − P(rew | L)")
            else:
                ax.set_xlabel("P(rew | R) + P(rew | L)")
            if row == 0:
                ax.set_title(f"{task}\n({len(t3):,} C3 steps, {len(t3) - len(e3):,} ignored)", fontsize=12)
        summary[task] = entry
    axes[0, 0].set_ylabel("C3: P(right | engaged)\n(engaged trials)")
    axes[1, 0].set_ylabel("C: P(right)")
    axes[2, 0].set_ylabel("C3: P(ignore)")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Tracking the true reward probabilities (shared eval sessions)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(s13.figure_path("fig_B6_tracking.png"))
    plt.close(fig)
    s13.write_json("b6_tracking.json", {
        "diff_bin_edges": DIFF_EDGES, "sum_bin_edges": SUM_EDGES, "min_per_bin": MIN_PER_BIN,
        "steady_after_trials": STEADY, "sessions": "shared eval sessions of C and C3 (split_alignment.json)",
        "tasks": summary,
    }, "analysis/b6_tracking.py", extra_groups=s13.twin_wandb_groups())


if __name__ == "__main__":
    main()
