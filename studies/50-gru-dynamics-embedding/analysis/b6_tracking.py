"""B6: tracking — choice against the true reward-probability difference, model vs animal, per task.

    python analysis/b6_tracking.py

x = P(reward | right) − P(reward | left) on the trial (the hidden task state); y = P(choose
right): the model's prediction and the animal's choice, binned. One panel per task, since the
difference means different things in coupled vs uncoupled and baited vs unbaited tasks.
Each task uses a few discrete probability levels, and each bin mixes trials just after a
probability change with settled ones, so a second, bold set uses only trials at least
STEADY trials after the last change of either side's probability.
Writes ``fig_B6_tracking.png`` and ``b6_tracking.json``.
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

# Bins centred on multiples of 0.1, and dp rounded first: dp values such as 0.4 − 0.1 sit on a
# 0.1 grid, and edges on that grid split them by floating-point error.
EDGES = np.arange(-1.05, 1.0501, 0.1)
MIN_PER_BIN = 200
STEADY = 10


def main() -> None:
    s12.apply_presentation_style()
    d = s12.load_trials(["ses_idx", "task", "action", "p_model1_right", "reward_probabilityL", "reward_probabilityR"])
    d["dp"] = d["reward_probabilityR"] - d["reward_probabilityL"]
    g = d.groupby("ses_idx", sort=False)
    changed = ((d["reward_probabilityL"] != g["reward_probabilityL"].shift())
               | (d["reward_probabilityR"] != g["reward_probabilityR"].shift()))
    d["since_change"] = d.groupby(changed.cumsum()).cumcount()
    centres = (EDGES[:-1] + EDGES[1:]) / 2
    fig, axes = plt.subplots(1, len(s12.TASKS), figsize=(5 * len(s12.TASKS), 5.4), sharey=True)
    summary = {}
    for ax, task in zip(axes, s12.TASKS):
        t = d[d["task"] == task]
        rows = {}
        for subset, mask in (("all", np.ones(len(t), bool)), ("steady", (t["since_change"] >= STEADY).to_numpy())):
            which = np.digitize(np.round(t["dp"].to_numpy()[mask], 6), EDGES) - 1
            model, animal = t["p_model1_right"].to_numpy()[mask], t["action"].to_numpy()[mask]
            r = {"model": [], "animal": [], "n": []}
            for b in range(len(centres)):
                sel = which == b
                n = int(sel.sum())
                r["n"].append(n)
                r["model"].append(float(model[sel].mean()) if n >= MIN_PER_BIN else np.nan)
                r["animal"].append(float(animal[sel].mean()) if n >= MIN_PER_BIN else np.nan)
            rows[subset] = r
        ax.plot(centres, rows["all"]["animal"], color="0.7", marker="x", ms=6, lw=0.8, label="animal, all trials")
        ax.plot(centres, rows["all"]["model"], color=s12.TASK_COLORS[task], marker="o", ms=3, lw=0.8, alpha=0.5,
                label="model, all trials")
        ax.plot(centres, rows["steady"]["animal"], color="0.3", marker="x", ms=8, lw=1.4,
                label=f"animal, ≥{STEADY} trials after a change")
        ax.plot(centres, rows["steady"]["model"], color=s12.TASK_COLORS[task], marker="o", ms=6, lw=2.4,
                label=f"model, ≥{STEADY} trials after a change")
        ax.axhline(0.5, color="0.85", lw=0.8)
        ax.axvline(0, color="0.85", lw=0.8)
        ax.set_title(f"{task}\n({len(t):,} trials)", fontsize=12)
        ax.set_xlabel("P(rew | R) − P(rew | L)")
        summary[task] = {"n_trials": int(len(t)), **rows}
    axes[0].set_ylabel("P(choose right)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_B6_tracking.png"))
    plt.close(fig)
    s12.write_json("b6_tracking.json", {"bin_edges": EDGES, "bin_centres": centres, "min_per_bin": MIN_PER_BIN,
                                        "steady_after_trials": STEADY,
                                        "tasks": summary}, "analysis/b6_tracking.py")


if __name__ == "__main__":
    main()
