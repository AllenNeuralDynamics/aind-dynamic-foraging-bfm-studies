"""B8: does C3 predict when the animals disengage — and when they re-engage?

    python analysis/b8_engagement.py

Every panel compares the model's predicted P(ignore) (mean over timesteps) with the
observed ignore rate on the same timesteps. The model is teacher-forced here (it sees the
animal's actual history), so these are calibration checks, not generative ones; Study 02
r3's free-running rollouts found generated ignore bouts too long.

Panels (``fig_B8_engagement.png``), each split by task where noted:
  a. calibration: observed ignore rate per bin of predicted P(ignore);
  b. bout exit: P(ignore) given the previous k timesteps were all ignored (k = 0 means the
     previous trial was engaged) — the hazard of staying disengaged;
  c. bout onset: P(ignore) around the first trial of a bout (bouts ≥ 2 trials, not cut by
     the session start), and d. bout end, around the first engaged trial after a bout;
  e. trial position within the session; f. trials since the last reward.
Writes ``b8_engagement.json`` with every curve plus run-length statistics.
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

COLUMNS = ["ses_idx", "trial_idx", "task", "action", "reward", "p_model1_ignore"]
CAL_BINS = np.r_[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MAX_RUN = 15
WINDOW = 10
POSITION_BIN = 25
MAX_POSITION = 1000
CAP = s13.CAP


def _curve(key: np.ndarray, observed: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"key": key, "observed": observed, "predicted": predicted})
    out = df.groupby("key").agg(n=("observed", "size"), observed=("observed", "mean"), predicted=("predicted", "mean"))
    out["observed_sem"] = np.sqrt(out["observed"] * (1 - out["observed"]) / out["n"])
    return out


def consecutive_ignores_before(action: np.ndarray, new_session: np.ndarray) -> np.ndarray:
    """For each timestep, how many immediately preceding timesteps in its session were ignored."""
    count = np.zeros(len(action), dtype=np.int64)
    for t in range(1, len(action)):
        if not new_session[t] and action[t - 1] == s13.IGNORE:
            count[t] = count[t - 1] + 1
    return count


def aligned(events: np.ndarray, values: np.ndarray, ses: np.ndarray) -> np.ndarray:
    """values[event + offset] for offset in [-WINDOW, WINDOW], NaN outside the event's session."""
    offsets = np.arange(-WINDOW, WINDOW + 1)
    idx = events[:, None] + offsets[None, :]
    valid = (idx >= 0) & (idx < len(values))
    idx_c = np.clip(idx, 0, len(values) - 1)
    valid &= ses[idx_c] == ses[events][:, None]
    return np.where(valid, values[idx_c], np.nan)


def main() -> None:
    s13.apply_presentation_style()
    trials = s13.load_trials(COLUMNS)
    action = trials["action"].to_numpy()
    ses = trials["ses_idx"].to_numpy()
    observed = (action == s13.IGNORE).astype(float)
    predicted = trials["p_model1_ignore"].to_numpy(np.float64)
    task = trials["task"].fillna("other").to_numpy()
    new_session = np.r_[True, ses[1:] != ses[:-1]]

    # a. calibration
    cal_bin = np.digitize(predicted, CAL_BINS[1:-1])
    calibration = _curve(cal_bin, observed, predicted)

    # b. hazard of staying disengaged
    k = np.minimum(consecutive_ignores_before(action, new_session), MAX_RUN)
    hazard = _curve(k, observed, predicted)
    hazard_by_task = {t: _curve(k[task == t], observed[task == t], predicted[task == t]) for t in s13.TASKS}

    # c, d. bout onset / end
    runs = s13.ignore_runs(action, ses)
    long_runs = runs[(runs["length"] >= 2) & ~runs["at_session_start"]]
    ended = runs[~runs["at_session_end"]]
    onset = {name: aligned(long_runs["start"].to_numpy(), v, ses) for name, v in (("observed", observed), ("predicted", predicted))}
    end = {name: aligned(ended["stop"].to_numpy(), v, ses) for name, v in (("observed", observed), ("predicted", predicted))}

    # e. trial position, f. trials since reward (timestep-based, ignored trials count as unrewarded)
    position = np.minimum(trials["trial_idx"].to_numpy() // POSITION_BIN * POSITION_BIN, MAX_POSITION)
    by_position = _curve(position, observed, predicted)
    rewarded = trials["reward"].to_numpy() > 0
    since = np.zeros(len(action), dtype=np.int64)
    for t in range(1, len(action)):
        since[t] = 0 if new_session[t] else (0 if rewarded[t - 1] else since[t - 1] + 1)
    since_reward = _curve(np.minimum(since, CAP), observed, predicted)

    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    ax = axes[0, 0]
    ax.plot([0, 1], [0, 1], color="0.7", lw=1, ls=":")
    ax.errorbar(calibration["predicted"], calibration["observed"], yerr=1.96 * calibration["observed_sem"],
                color=s13.ENGAGE_COLOR, marker="o", ms=5, lw=1.5)
    ax.set(xlabel="predicted P(ignore) (bin mean)", ylabel="observed ignore rate", xlim=(0, 1), ylim=(0, 1),
           title="a. calibration (all timesteps)")

    ax = axes[0, 1]
    for t in s13.TASKS:
        h = hazard_by_task[t]
        ax.plot(h.index, h["observed"], color=s13.TASK_COLORS[t], lw=1.6, marker="o", ms=4, label=f"{t} observed")
        ax.plot(h.index, h["predicted"], color=s13.TASK_COLORS[t], lw=1.6, ls="--")
    ax.plot(hazard.index, hazard["observed"], color="k", lw=2.2, marker="o", ms=5, label="all tasks observed")
    ax.plot(hazard.index, hazard["predicted"], color="k", lw=2.2, ls="--", label="model (dashed)")
    ax.set(xlabel=f"previous consecutive ignores k (capped at {MAX_RUN})", ylabel="P(ignore at this trial)",
           ylim=(0, 1), title="b. staying disengaged: P(ignore | k ignores just before)")
    ax.legend(fontsize=9, frameon=False, loc="lower right")

    offsets = np.arange(-WINDOW, WINDOW + 1)
    for ax, data, n, label, title in (
        (axes[0, 2], onset, len(long_runs), "first ignored trial of the bout", f"c. bout onset (bouts ≥ 2, n={len(long_runs):,})"),
        (axes[1, 0], end, len(ended), "first engaged trial after the bout", f"d. bout end (n={len(ended):,})"),
    ):
        ax.plot(offsets, np.nanmean(data["observed"], axis=0), color="k", marker="o", ms=4, label="observed")
        ax.plot(offsets, np.nanmean(data["predicted"], axis=0), color=s13.ENGAGE_COLOR, ls="--", lw=2, label="model")
        ax.axvline(0, color="0.7", lw=1)
        ax.set(xlabel=f"trials from the {label}", ylabel="P(ignore)", ylim=(0, 1), title=title)
        ax.legend(frameon=False)

    for ax, curve, xlabel, title in (
        (axes[1, 1], by_position, f"trial position in session ({POSITION_BIN}-trial bins)", "e. within the session"),
        (axes[1, 2], since_reward, f"trials since last reward (capped at {CAP})", "f. reward history"),
    ):
        ax.plot(curve.index, curve["observed"], color="k", marker="o", ms=4, label="observed")
        ax.plot(curve.index, curve["predicted"], color=s13.ENGAGE_COLOR, ls="--", lw=2, label="model")
        ax.set(xlabel=xlabel, ylabel="P(ignore)", title=title)
        ax.set_ylim(0, max(curve["observed"].max(), curve["predicted"].max()) * 1.05)
        ax.legend(frameon=False)
    fig.suptitle("C3 engagement, teacher-forced: model P(ignore) vs observed ignore rate (eval trials, 614 mice)", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(s13.figure_path("fig_B8_engagement.png"))
    plt.close(fig)

    def table(curve: pd.DataFrame) -> dict:
        return {str(i): {"n": int(r.n), "observed": round(float(r.observed), 5), "predicted": round(float(r.predicted), 5)}
                for i, r in curve.iterrows()}

    complete = runs[~runs["at_session_start"] & ~runs["at_session_end"]]
    s13.write_json("b8_engagement.json", {
        "n_timesteps": int(len(trials)),
        "ignore_rate": float(observed.mean()),
        "mean_predicted_p_ignore": float(predicted.mean()),
        "runs": {
            "n": int(len(runs)), "n_complete": int(len(complete)),
            "mean_length_complete": float(complete["length"].mean()),
            "share_length_1_complete": float((complete["length"] == 1).mean()),
            "length_quantiles_complete": complete["length"].quantile([0.5, 0.9, 0.99]).to_dict(),
        },
        "calibration_bins": CAL_BINS.tolist(),
        "calibration": table(calibration),
        "stay_disengaged_by_k": table(hazard),
        "stay_disengaged_by_k_by_task": {t: table(h) for t, h in hazard_by_task.items()},
        "bout_onset": {"n": int(len(long_runs)), "offsets": offsets.tolist(),
                       **{n: np.nanmean(v, axis=0).round(5).tolist() for n, v in onset.items()}},
        "bout_end": {"n": int(len(ended)), "offsets": offsets.tolist(),
                     **{n: np.nanmean(v, axis=0).round(5).tolist() for n, v in end.items()}},
        "by_trial_position": table(by_position),
        "by_trials_since_reward": table(since_reward),
    }, "analysis/b8_engagement.py")


if __name__ == "__main__":
    main()
