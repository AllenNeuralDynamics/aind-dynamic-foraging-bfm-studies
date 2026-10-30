"""B1: example sessions, one per task — C3 (3-way) above C (2-way, Study 50) on the same session.

    python analysis/b1_example_sessions.py

Uses Study 50's B1 sessions (all four are in both eval splits). The x axis is the session's
original trial number, so ignored trials (absent from C's table) line up across panels.
Writes ``fig_B1_example_sessions.png`` and ``b1_example_sessions.json`` (per-session
likelihoods: C3's L/R | engaged and engage-vs-ignore, C's L/R).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk02 as s13  # noqa: E402

COLUMNS = ["subject_id", "ses_idx", "trial", "task", "action", "reward", "reward_probabilityL",
           "reward_probabilityR", "p_model1_left", "p_model1_right", "p_model1_ignore"]
TWIN_COLUMNS = ["ses_idx", "trial", "action", "p_model1_left", "p_model1_right"]
SMOOTH = 10


def _geo_mean(p: np.ndarray) -> float:
    return float(np.exp(np.mean(np.log(p))))


def _ticks(ax, t, choice, rewarded) -> None:
    """Right choices above, left below (tall = rewarded); ignored trials as grey ticks at the top."""
    for side, y0, sign in ((1, 1.02, 1), (0, -0.02, -1)):
        for is_rewarded, height in ((True, 0.10), (False, 0.04)):
            mask = (choice == side) & (rewarded == is_rewarded)
            ax.vlines(t[mask], y0, y0 + sign * height, color=s13.RIGHT_COLOR if side else s13.LEFT_COLOR, lw=0.8)


def main() -> None:
    s13.apply_presentation_style()
    chosen = s13.example_sessions()
    trials = s13.load_trials(COLUMNS, sessions=list(chosen.values()))
    twin = s13.load_twin(TWIN_COLUMNS, sessions=list(chosen.values()))

    fig, axes = plt.subplots(2 * len(chosen), 1, figsize=(16, 2.9 * 2 * len(chosen)))
    shown = {}
    for k, (task, ses) in enumerate(chosen.items()):
        ax3, ax2 = axes[2 * k], axes[2 * k + 1]
        d = trials[trials["ses_idx"] == ses].sort_values("trial")
        c = twin[twin["ses_idx"] == ses].sort_values("trial")
        t, choice = d["trial"].to_numpy(), d["action"].to_numpy()
        rewarded = d["reward"].to_numpy() > 0
        engaged = choice != s13.IGNORE
        p_r = d["p_right_engaged"].to_numpy()

        # C3: P(R | engaged), P(ignore), the animal's choices and ignores.
        for ax in (ax3, ax2):
            ax.plot(t, d["reward_probabilityR"], color=s13.RIGHT_COLOR, lw=1.3, alpha=0.5, label="P(reward | right)")
            ax.plot(t, d["reward_probabilityL"], color=s13.LEFT_COLOR, lw=1.3, alpha=0.5, label="P(reward | left)")
            ax.vlines(t[~engaged], -0.15, 1.15, color=s13.IGNORE_COLOR, lw=0.6, alpha=0.35, zorder=0)
            _ticks(ax, t, choice, rewarded)
            ax.set_ylim(-0.15, 1.15)
            ax.set_xlim(t[0], t[-1])
        ignored_mean = np.convolve((~engaged).astype(float), np.ones(SMOOTH) / SMOOTH, mode="same")
        ax3.plot(t, ignored_mean, color=s13.IGNORE_COLOR, lw=1.5, ls="--", label=f"animal ignore rate ({SMOOTH}-trial mean)")
        ax3.plot(t, d["p_model1_ignore"], color=s13.ENGAGE_COLOR, lw=1.5, label="C3 P(ignore)")
        ax3.plot(t, p_r, color=s13.MODEL_COLOR, lw=1.6, label="C3 P(right | engaged)")
        ax2.plot(c["trial"], c["p_model1_right"], color=s13.MODEL_COLOR, lw=1.6, label="C P(right)")
        ax3.set_xticklabels([])

        side = np.where(choice[engaged] == 1, p_r[engaged], 1 - p_r[engaged])
        p_engage = np.where(engaged, 1 - d["p_model1_ignore"].to_numpy(), d["p_model1_ignore"].to_numpy())
        c_side = np.where(c["action"] == 1, c["p_model1_right"], c["p_model1_left"])
        row = {
            "ses_idx": ses, "subject_id": str(d["subject_id"].iloc[0]), "n_trials": int(len(d)),
            "n_ignored": int((~engaged).sum()),
            "C3_LR_engaged_likelihood": _geo_mean(side), "C3_engage_likelihood": _geo_mean(p_engage),
            "C_LR_likelihood": _geo_mean(c_side), "C_n_trials": int(len(c)),
        }
        if row["C_n_trials"] != int(engaged.sum()):
            raise AssertionError(f"{ses}: C has {len(c)} trials, C3 has {engaged.sum()} engaged")
        shown[task] = row
        ax3.set_ylabel("C3 (3-way)")
        ax2.set_ylabel("C (2-way)")
        ax3.set_title(
            f"{task} — mouse {row['subject_id']}, {ses.split('__')[-1]} ({row['n_trials']} trials, "
            f"{row['n_ignored']} ignored) — L/R likelihood on engaged trials: C3 {row['C3_LR_engaged_likelihood']:.3f}, "
            f"C {row['C_LR_likelihood']:.3f}; C3 engage {row['C3_engage_likelihood']:.3f}",
            fontsize=12, loc="left")
    axes[-1].set_xlabel("trial (original numbering; ignored trials shaded grey)")
    handles = {}
    for ax in axes[:2]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=4, fontsize=11, frameon=False)
    fig.text(0.01, 0.003, "Ticks: right choices above, left below; tall = rewarded. Both panels of a task show the same session.",
             fontsize=10, color="0.35")
    fig.tight_layout(rect=(0, 0.01, 1, 0.97))
    fig.savefig(s13.figure_path("fig_B1_example_sessions.png"))
    plt.close(fig)
    s13.write_json("b1_example_sessions.json", {"sessions": shown}, "analysis/b1_example_sessions.py",
                   extra_groups=s13.twin_wandb_groups())


if __name__ == "__main__":
    main()
