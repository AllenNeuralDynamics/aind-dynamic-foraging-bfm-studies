"""B1: example sessions, one per task — choices, rewards, true reward probabilities, model P(right).

    python analysis/b1_example_sessions.py

Reads the A1 trial table; writes ``fig_B1_example_sessions.png`` and
``b1_example_sessions.json`` (the sessions shown and their per-session likelihood).
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

COLUMNS = ["subject_id", "ses_idx", "trial_idx", "task", "action", "reward",
           "reward_probabilityL", "reward_probabilityR", "p_model1_right", "p_model1_left"]
SMOOTH = 10


def main() -> None:
    s12.apply_presentation_style()
    trials = s12.load_trials(COLUMNS)
    chosen = s12.example_sessions(trials)

    fig, axes = plt.subplots(len(chosen), 1, figsize=(16, 3.3 * len(chosen)), sharey=True)
    shown = {}
    for ax, (task, ses) in zip(axes, chosen.items()):
        d = trials[trials["ses_idx"] == ses].sort_values("trial_idx")
        t = d["trial_idx"].to_numpy()
        choice = d["action"].to_numpy()
        rewarded = d["reward"].to_numpy() > 0
        p_chosen = np.where(choice == 1, d["p_model1_right"], d["p_model1_left"])

        ax.plot(t, d["reward_probabilityR"], color=s12.RIGHT_COLOR, lw=1.4, alpha=0.6, label="P(reward | right)")
        ax.plot(t, d["reward_probabilityL"], color=s12.LEFT_COLOR, lw=1.4, alpha=0.6, label="P(reward | left)")
        smoothed = pd.Series(choice).rolling(SMOOTH, center=True, min_periods=1).mean().to_numpy()  # no zero-padding at the ends
        ax.plot(t, smoothed, color="0.55", lw=1.6, ls="--", label=f"animal choice ({SMOOTH}-trial mean)")
        ax.plot(t, d["p_model1_right"], color=s12.MODEL_COLOR, lw=1.6, label="model P(right)")
        for side, y0, sign in ((1, 1.02, 1), (0, -0.02, -1)):
            for is_rewarded, height in ((True, 0.10), (False, 0.04)):
                mask = (choice == side) & (rewarded == is_rewarded)
                ax.vlines(t[mask], y0, y0 + sign * height,
                          color=s12.RIGHT_COLOR if side else s12.LEFT_COLOR, lw=0.8)
        ax.set_ylim(-0.15, 1.15)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylabel("P(right)")
        ax.set_title(f"{task} — mouse {d['subject_id'].iloc[0]}, {ses.split('__')[-1]} "
                     f"({len(d)} trials, model likelihood {np.exp(np.mean(np.log(p_chosen))):.3f})",
                     fontsize=13, loc="left")
        shown[task] = {"ses_idx": ses, "subject_id": d["subject_id"].iloc[0], "n_trials": len(d),
                       "model_likelihood": float(np.exp(np.mean(np.log(p_chosen))))}
    axes[-1].set_xlabel("trial")
    axes[0].legend(loc="upper right", ncol=4, fontsize=10, frameon=False, bbox_to_anchor=(1, 1.32))
    fig.text(0.01, 0.005, "Ticks: right choices above, left below; tall = rewarded, short = unrewarded.",
             fontsize=10, color="0.35")
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_B1_example_sessions.png"))
    plt.close(fig)
    s12.write_json("b1_example_sessions.json", {"sessions": shown}, "analysis/b1_example_sessions.py")


if __name__ == "__main__":
    main()
