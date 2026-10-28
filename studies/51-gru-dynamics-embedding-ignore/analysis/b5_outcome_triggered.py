"""B5: outcome-triggered averages — how long one input moves each readout, model vs animal.

    python analysis/b5_outcome_triggered.py

Steps k are GRU timesteps (an ignored trial is a step for C3; C's steps are engaged trials
only). Matched effects follow Study 50: the difference in Δreadout(t→t+k) between two input
groups, within bins of the readout at t, weighted by bin size — raw per-group curves are
dominated by regression to the mean. Panels:
  a. reward effect on the side readout (rewarded − unrewarded, same choice), C3;
  b. the same for C (Study 50), on the shared sessions;
  c. reward effect on the engage readout (rewarded − unrewarded, same choice), C3;
  d. P(ignore) on step t+k after each input at t, model vs animal;
  e. P(right | engaged) on engaged trials at t+k after each input, model vs animal;
  f. ignore effect on the engage readout: ignored − unrewarded engaged at t, matched on
     engage_logit(t).
Writes ``fig_B5_outcome_triggered.png`` and ``b5_outcome_triggered.json``.
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

K = 10
SIDE_EDGES = np.linspace(-4, 4, 17)
ENGAGE_EDGES = np.linspace(-3, 7, 21)
MIN_PER_CELL = 50
INPUTS = [("R+", s13.RIGHT_COLOR, "-"), ("R−", s13.RIGHT_COLOR, "--"), ("L+", s13.LEFT_COLOR, "-"),
          ("L−", s13.LEFT_COLOR, "--"), ("ignore", s13.IGNORE_COLOR, "-")]
KS = np.arange(1, K + 1)


def input_label(action: np.ndarray, reward: np.ndarray) -> np.ndarray:
    side = np.where(action == 1, "R", "L").astype(object) + np.where(reward > 0, "+", "−")
    return np.where(action == s13.IGNORE, "ignore", side)


def futures(d: pd.DataFrame, columns: list[str]) -> dict[int, dict[str, np.ndarray]]:
    g = d.groupby("ses_idx", sort=False)
    return {k: {c: g[c].shift(-k).to_numpy(dtype=float) for c in columns} for k in KS}


def matched_effect(value: np.ndarray, future: dict, column: str, edges: np.ndarray,
                   group_a: np.ndarray, group_b: np.ndarray, base: np.ndarray | None = None) -> np.ndarray:
    """Mean Δcolumn(t→t+k) for group a minus group b, within bins of column(t)."""
    bins = np.digitize(value, edges)
    out = []
    for k in KS:
        f = future[k][column]
        delta = f - value
        num, den = 0.0, 0
        for b in np.unique(bins[group_a | group_b]):
            sel = (bins == b) & np.isfinite(f) & (np.ones_like(f, bool) if base is None else base)
            a, z = sel & group_a, sel & group_b
            n = min(a.sum(), z.sum())
            if n >= MIN_PER_CELL:
                num += n * (delta[a].mean() - delta[z].mean())
                den += n
        out.append(num / den if den else np.nan)
    return np.asarray(out)


def side_reward_effect(d: pd.DataFrame, future: dict, label: np.ndarray, task: str | None = None) -> dict:
    base = None if task is None else (d["task"] == task).to_numpy()
    x = d["logit_diff"].to_numpy(float)
    return {side: matched_effect(x, future, "logit_diff", SIDE_EDGES, label == f"{side}+", label == f"{side}−", base)
            for side in ("R", "L")}


def main() -> None:
    s13.apply_presentation_style()
    drop = s13.unshared_sessions()
    cols = ["ses_idx", "session_id", "task", "action", "reward", "logit_diff"]
    c3 = s13.load_trials(cols + ["choice_logit_0", "choice_logit_1", "choice_logit_2",
                                 "p_model1_left", "p_model1_right", "p_model1_ignore"])
    c3 = c3[~c3["session_id"].isin(drop["C3"])].reset_index(drop=True)
    c = s13.load_twin(cols)
    c = c[~c["session_id"].isin(drop["C"])].reset_index(drop=True)

    label3 = input_label(c3["action"].to_numpy(), c3["reward"].to_numpy())
    label2 = input_label(c["action"].to_numpy(), c["reward"].to_numpy())
    c3["ignored"] = (c3["action"] == s13.IGNORE).astype(float)
    c3["chose_right"] = np.where(c3["action"] == s13.IGNORE, np.nan, (c3["action"] == 1).astype(float))
    c3["p_right_engaged_on_engaged"] = np.where(c3["action"] == s13.IGNORE, np.nan, c3["p_right_engaged"])
    f3 = futures(c3, ["logit_diff", "engage_logit", "ignored", "p_model1_ignore", "chose_right",
                      "p_right_engaged_on_engaged"])
    f2 = futures(c, ["logit_diff"])

    side3 = {"all": side_reward_effect(c3, f3, label3), "by_task": {t: side_reward_effect(c3, f3, label3, t) for t in s13.TASKS}}
    side2 = {"all": side_reward_effect(c, f2, label2), "by_task": {t: side_reward_effect(c, f2, label2, t) for t in s13.TASKS}}
    e = c3["engage_logit"].to_numpy()
    engage_reward = {side: matched_effect(e, f3, "engage_logit", ENGAGE_EDGES, label3 == f"{side}+", label3 == f"{side}−")
                     for side in ("R", "L")}
    unrewarded = np.isin(label3, ["R−", "L−"])
    ignore_effect = matched_effect(e, f3, "engage_logit", ENGAGE_EDGES, label3 == "ignore", unrewarded)

    after = {}
    for value, _, _ in INPUTS:
        mask = label3 == value
        after[value] = {name: [float(np.nanmean(f3[k][col][mask])) for k in KS] for name, col in (
            ("p_ignore_model", "p_model1_ignore"), ("ignore_rate_animal", "ignored"),
            ("p_right_engaged_model", "p_right_engaged_on_engaged"), ("p_right_animal", "chose_right"))}
        after[value]["n"] = int(mask.sum())

    fig, axes = plt.subplots(2, 3, figsize=(22, 12.5))
    side_lim = [min(np.nanmin(v) for s in (side3, side2) for v in s["all"].values()) - 0.1,
                max(np.nanmax(v) for s in (side3, side2) for v in s["all"].values()) + 0.1]
    for ax, effect, title in ((axes[0, 0], side3, "a. C3: reward effect on logit(R | engaged)"),
                              (axes[0, 1], side2, "b. C (Study 50): reward effect on logit(R)")):
        for side, color in (("R", s13.RIGHT_COLOR), ("L", s13.LEFT_COLOR)):
            ax.plot(KS, effect["all"][side], color=color, marker="o", ms=5, lw=2.6, label=f"chose {side}, all tasks")
            for task in s13.TASKS:
                ax.plot(KS, effect["by_task"][task][side], color=s13.TASK_COLORS[task], lw=1, alpha=0.8,
                        ls="-" if side == "R" else "--")
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set(title=title, xlabel="steps after t (k)", ylabel="rewarded − unrewarded, matched", ylim=side_lim)
        ax.legend(fontsize=9, frameon=False)
    ax = axes[0, 2]
    for side, color in (("R", s13.RIGHT_COLOR), ("L", s13.LEFT_COLOR)):
        ax.plot(KS, engage_reward[side], color=color, marker="o", ms=5, lw=2.4, label=f"chose {side}")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set(title="c. C3: reward effect on logit(engaged / ignore)", xlabel="steps after t (k)",
           ylabel="rewarded − unrewarded, matched")
    ax.legend(frameon=False)

    for ax, (model, animal, title, ylabel) in zip(axes[1, :2], (
            ("p_ignore_model", "ignore_rate_animal", "d. P(ignore) after each input", "P(ignore) on step t+k"),
            ("p_right_engaged_model", "p_right_animal", "e. P(right | engaged) after each input (engaged trials)",
             "P(right) on step t+k"))):
        for value, color, ls in INPUTS:
            ax.plot(KS, after[value][model], color=color, ls=ls, marker="o", ms=4, label=f"{value} — model")
            ax.plot(KS, after[value][animal], color=color, ls=ls, marker="x", ms=7, lw=0.8, alpha=0.7,
                    label=f"{value} — animal")
        ax.set(title=title, xlabel="steps after t (k)", ylabel=ylabel)
        ax.legend(fontsize=8, frameon=False, ncol=2)
    ax = axes[1, 2]
    ax.plot(KS, ignore_effect, color=s13.ENGAGE_COLOR, marker="o", ms=5, lw=2.4)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set(title="f. C3: ignore effect on logit(engaged / ignore)", xlabel="steps after t (k)",
           ylabel="ignored − unrewarded choice at t, matched")
    for a in axes.flat:
        a.set_xticks(KS)
    fig.suptitle("Outcome-triggered averages (shared eval sessions; k in GRU steps — C3 counts ignored trials, C does not)",
                 fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(s13.figure_path("fig_B5_outcome_triggered.png"))
    plt.close(fig)
    s13.write_json("b5_outcome_triggered.json", {
        "k": KS, "side_bin_edges": SIDE_EDGES, "engage_bin_edges": ENGAGE_EDGES, "min_per_cell": MIN_PER_CELL,
        "sessions": "shared eval sessions of C and C3 (split_alignment.json)",
        "side_reward_effect_C3": side3, "side_reward_effect_C": side2,
        "engage_reward_effect_C3": engage_reward, "engage_ignore_vs_unrewarded_C3": ignore_effect,
        "after_input_C3": after,
    }, "analysis/b5_outcome_triggered.py", extra_groups=s13.twin_wandb_groups())


if __name__ == "__main__":
    main()
