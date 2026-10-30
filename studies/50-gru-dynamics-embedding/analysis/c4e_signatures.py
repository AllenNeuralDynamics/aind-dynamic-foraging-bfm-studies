"""C4e: the two RL failure modes of C4c as behavioural curves — animals vs each model.

    python analysis/c4e_signatures.py

Row 1 — return after a switch. Trial t−1 switched side (its choice differs from t−2's); x =
the length of the run on the old side that ended at t−2 (capped at RUN_CAP); y = P(choice t
returns to the old side). Solid: the switch was unrewarded (a failed switch); dashed: it was
rewarded (control).
Row 2 — staying through losses. n = the number of consecutive unrewarded choices on one side
ending at t−1; y = P(choice t stays on that side). Solid: the streak began after a rewarded
choice on the same side; dashed: it began right after switching to that side. Streaks that
start a session are excluded.
Animals: observed rate ± binomial SEM (thick grey, in every panel); models: their mean predicted
probability on the same trials. Writes ``fig_C4e_signatures.png`` and ``c4e_signatures.json``.
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
from c4_behaviour import COLORS, MODELS  # noqa: E402

RUN_CAP = 10
STREAK_CAP = 8
MIN_N = 200
ANIMAL_COLOR = "#AAAAAA"


def curves(x: np.ndarray, hit: np.ndarray, p_hit: dict, rows: np.ndarray, values: range) -> dict:
    out = {"x": [], "n": [], "animal": [], "animal_sem": [], **{m: [] for m in p_hit}}
    for v in values:
        sel = rows[x[rows] == v]
        if len(sel) < MIN_N:
            continue
        rate = hit[sel].mean()
        out["x"].append(v)
        out["n"].append(len(sel))
        out["animal"].append(rate)
        out["animal_sem"].append(np.sqrt(rate * (1 - rate) / len(sel)))
        for m, ph in p_hit.items():
            out[m].append(ph[sel].mean())
    return {k: np.array(v) for k, v in out.items()}


def signature_curves(ses: np.ndarray, a: np.ndarray, r: np.ndarray, p: dict) -> tuple[dict, dict]:
    """Both signature curves for the animals and every model in ``p`` (P(right) per trial)."""
    n = len(a)
    first = np.r_[True, ses[1:] != ses[:-1]]
    idx = np.arange(n)
    # Position within the current run of same-side choices (1 at the run's first trial).
    new_run = first | np.r_[True, a[1:] != a[:-1]]
    run_pos = pd.Series(np.ones(n)).groupby(np.cumsum(new_run)).cumsum().to_numpy().astype(int)
    # Consecutive unrewarded same-side choices ending at each trial (0 on a rewarded trial).
    new_seg = first | np.r_[True, (a[1:] != a[:-1]) | r[:-1]]
    streak = np.where(r, 0, pd.Series(np.ones(n)).groupby(np.cumsum(new_seg)).cumsum().to_numpy()).astype(int)

    # Row 1: trial t−1 switched; rows t with two earlier trials in the same session.
    ok1 = np.zeros(n, bool)
    ok1[2:] = ~first[2:] & ~first[1:-1]
    switched = np.zeros(n, bool)
    switched[2:] = a[1:-1] != a[:-2]
    rows1 = idx[ok1 & switched]
    old = np.zeros(n, int)
    old[2:] = a[:-2]
    run_before = np.zeros(n, int)
    run_before[2:] = np.minimum(run_pos[:-2], RUN_CAP)
    ret = a == old
    p_ret = {m: np.where(old == 1, v, 1 - v) for m, v in p.items()}
    prev_rew = np.r_[False, r[:-1]]
    row1 = {"failed": curves(run_before, ret, p_ret, rows1[~prev_rew[rows1]], range(1, RUN_CAP + 1)),
            "rewarded": curves(run_before, ret, p_ret, rows1[prev_rew[rows1]], range(1, RUN_CAP + 1))}

    # Row 2: a loss streak ending at t−1.
    n_streak = np.zeros(n, int)
    n_streak[1:] = np.where(first[1:], 0, streak[:-1])
    has = n_streak > 0
    start = idx - n_streak  # first trial of the streak
    before = start - 1
    valid = has & (before >= 0)
    valid[valid] &= ses[before[valid]] == ses[idx[valid]]
    origin_switch = np.zeros(n, bool)
    origin_switch[valid] = a[before[valid]] != a[start[valid]]
    stay = a == np.r_[-1, a[:-1]]
    p_stay = {m: np.where(np.r_[0, a[:-1]] == 1, v, 1 - v) for m, v in p.items()}
    capped = np.minimum(n_streak, STREAK_CAP)
    rows2 = idx[valid]
    row2 = {"after a reward on that side": curves(capped, stay, p_stay, rows2[~origin_switch[rows2]], range(1, STREAK_CAP + 1)),
            "right after switching there": curves(capped, stay, p_stay, rows2[origin_switch[rows2]], range(1, STREAK_CAP + 1))}
    return row1, row2


def main() -> None:
    s12.apply_presentation_style()
    t = s12.load_trials(["ses_idx", "action", "reward", "p_model1_right"])
    a = t["action"].to_numpy().astype(int)
    r = (t["reward"].to_numpy() > 0)
    ses = t["ses_idx"].to_numpy()
    p = {"gru": t["p_model1_right"].to_numpy(dtype=np.float64)}
    for m in ("hattori", "bari", "ctt"):
        rl = pd.read_parquet(s12.CACHE / f"rl_trials_{m}_{s12.MODEL_KEY}.parquet")
        if not (rl["ses_idx"].to_numpy() == ses).all():
            raise AssertionError(f"{m} rollout rows are not aligned with the trial table")
        p[m] = rl["p_rl_right"].to_numpy(dtype=np.float64)
    row1, row2 = signature_curves(ses, a, r, p)

    fig, axes = plt.subplots(2, 4, figsize=(28, 13), sharey="row")
    for c, m in enumerate(MODELS):
        for ax, data, xlabel, ylabel, cap in (
                (axes[0, c], row1, "run length on the old side before the switch",
                 "P(return to the old side)", RUN_CAP),
                (axes[1, c], row2, "consecutive unrewarded choices on one side",
                 "P(stay on that side)", STREAK_CAP)):
            for (label, d), ls in zip(data.items(), ("-", "--")):
                ax.errorbar(d["x"], d["animal"], yerr=d["animal_sem"], color=ANIMAL_COLOR, ls=ls, marker="o", ms=7,
                            lw=3.5, alpha=0.8, capsize=2, label=f"animals: {label}")
                ax.plot(d["x"], d[m], color=COLORS[m], ls=ls, marker="s", ms=4, lw=2, label=f"{MODELS[m]}: {label}")
            ax.set_xticks(range(1, cap + 1), [str(v) if v < cap else f"{cap}+" for v in range(1, cap + 1)])
            ax.set_xlabel(xlabel)
            ax.legend(frameon=False, fontsize=9)
            if c == 0:
                ax.set_ylabel(ylabel)
        axes[0, c].set_title(MODELS[m], fontsize=15)
    fig.suptitle("Top: returning after a switch (solid: the switch was unrewarded; dashed: rewarded).   "
                 "Bottom: staying through losses (solid: the streak began after a reward on that side; dashed: right "
                 "after switching there).\nThick grey: animals (± SEM); colour: each model's mean prediction on the same "
                 "trials", fontsize=15)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_C4e_signatures.png"))
    plt.close(fig)

    def err(data):
        return {label: {m: float(np.average(np.abs(d[m] - d["animal"]), weights=d["n"])) for m in MODELS}
                for label, d in data.items()}

    s12.write_json("c4e_signatures.json", {
        "return_after_switch": {k: {kk: vv for kk, vv in v.items()} for k, v in row1.items()},
        "stay_through_losses": {k: {kk: vv for kk, vv in v.items()} for k, v in row2.items()},
        "mean_abs_error": {"return_after_switch": err(row1), "stay_through_losses": err(row2)},
        "run_cap": RUN_CAP, "streak_cap": STREAK_CAP, "min_n": MIN_N,
    }, "analysis/c4e_signatures.py", extra_groups=s12.rl_groups())
    for name, data in (("return", row1), ("losses", row2)):
        for label, d in data.items():
            print(name, label, "x", d["x"].tolist(), "n", d["n"].tolist())
            for k in ["animal", *MODELS]:
                print("   ", k, np.round(d[k], 3).tolist())


if __name__ == "__main__":
    main()
