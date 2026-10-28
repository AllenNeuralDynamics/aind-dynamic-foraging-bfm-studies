"""B5: outcome-triggered averages — how long one trial's outcome moves the model and the animal.

    python analysis/b5_outcome_triggered.py

For every trial t, grouped by its choice x reward, over the next K trials:
  * top:    the reward effect — Δlogit(right) from t to t+k for rewarded minus unrewarded
            trials with the same choice, matched on logit(t) (difference within logit(t)
            bins, weighted by bin size). Raw Δlogit per group is dominated by regression to
            the mean (grouping by choice selects trials already leaning that way), so the
            raw curves are only kept in the JSON;
  * bottom: P(choose right) on trial t+k, model prediction vs animal choice.
Task-agnostic (no block structure needed). Writes ``fig_B5_outcome_triggered.png`` and
``b5_outcome_triggered.json``.
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

K = 10
LOGIT_EDGES = np.linspace(-4, 4, 17)
CONDITIONS = [(1, 1, "right, rewarded", s12.RIGHT_COLOR, "-"), (1, 0, "right, unrewarded", s12.RIGHT_COLOR, "--"),
              (0, 1, "left, rewarded", s12.LEFT_COLOR, "-"), (0, 0, "left, unrewarded", s12.LEFT_COLOR, "--")]


def main() -> None:
    s12.apply_presentation_style()
    d = s12.load_trials(["ses_idx", "trial_idx", "task", "action", "reward", "logit_diff", "p_model1_right"])
    d["reward"] = (d["reward"] > 0).astype(int)
    grouped = d.groupby("ses_idx", sort=False)
    ks = np.arange(1, K + 1)
    future = {k: {c: grouped[c].shift(-k).to_numpy() for c in ("logit_diff", "p_model1_right", "action")} for k in ks}
    logit_t = d["logit_diff"].to_numpy()

    def curves(mask: np.ndarray) -> dict[str, np.ndarray]:
        out = {"dlogit": [], "p_model": [], "p_animal": [], "n": []}
        for k in ks:
            f = future[k]
            valid = mask & np.isfinite(f["logit_diff"])
            out["dlogit"].append(np.mean(f["logit_diff"][valid] - logit_t[valid]))
            out["p_model"].append(np.mean(f["p_model1_right"][valid]))
            out["p_animal"].append(np.mean(f["action"][valid]))
            out["n"].append(int(valid.sum()))
        return {key: np.asarray(v) for key, v in out.items()}

    action, reward = d["action"].to_numpy(), d["reward"].to_numpy()
    logit_bin = np.digitize(logit_t, LOGIT_EDGES)

    def matched_reward_effect(choice: int, extra: np.ndarray | None = None) -> np.ndarray:
        """Rewarded − unrewarded Δlogit(t→t+k), same choice, within logit(t) bins."""
        base = action == choice if extra is None else (action == choice) & extra
        effect = []
        for k in ks:
            f = future[k]["logit_diff"]
            delta = f - logit_t
            num, den = 0.0, 0
            for b in np.unique(logit_bin[base]):
                sel = base & (logit_bin == b) & np.isfinite(f)
                rw, un = sel & (reward == 1), sel & (reward == 0)
                n = min(rw.sum(), un.sum())
                if n >= 50:
                    num += n * (delta[rw].mean() - delta[un].mean())
                    den += n
            effect.append(num / den if den else np.nan)
        return np.asarray(effect)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 11), sharex=True)
    summary = {}
    for choice, rewarded, label, color, ls in CONDITIONS:
        c = curves((action == choice) & (reward == rewarded))
        ax_bottom.plot(ks, c["p_model"], color=color, ls=ls, marker="o", ms=4, label=f"{label} — model")
        ax_bottom.plot(ks, c["p_animal"], color=color, ls=ls, marker="x", ms=7, lw=0.8, alpha=0.7,
                       label=f"{label} — animal")
        summary[label] = c
        summary[label]["by_task"] = {
            task: curves((action == choice) & (reward == rewarded) & (d["task"] == task).to_numpy())
            for task in s12.TASKS}
    reward_effect = {}
    for choice, label, color in ((1, "right", s12.RIGHT_COLOR), (0, "left", s12.LEFT_COLOR)):
        effect = matched_reward_effect(choice)
        ax_top.plot(ks, effect, color=color, marker="o", ms=5, lw=2.6, label=f"chose {label} (all tasks)")
        reward_effect[label] = {"all": effect, "by_task": {}}
        for task in s12.TASKS:
            task_effect = matched_reward_effect(choice, (d["task"] == task).to_numpy())
            ax_top.plot(ks, task_effect, color=s12.TASK_COLORS[task], lw=1, alpha=0.8,
                        ls="-" if choice else "--", label=f"{task}, {label}")
            reward_effect[label]["by_task"][task] = task_effect
    ax_top.axhline(0, color="0.7", lw=0.8)
    ax_top.set_ylabel("reward effect on logit(right)")
    ax_top.legend(fontsize=8, frameon=False, ncol=2)
    ax_top.set_title("Rewarded − unrewarded, same choice, matched on logit(t)", fontsize=13)
    ax_bottom.axhline(np.mean(action), color="0.7", lw=0.8)
    ax_bottom.set_ylabel("P(choose right) on t+k")
    ax_bottom.set_xlabel("trials after t (k)")
    ax_bottom.set_xticks(ks)
    ax_bottom.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_B5_outcome_triggered.png"))
    plt.close(fig)
    s12.write_json("b5_outcome_triggered.json", {"k": ks, "reward_effect_matched": reward_effect,
                                                 "logit_bin_edges": LOGIT_EDGES, "conditions_raw": summary,
                                                 "overall_p_right": float(np.mean(action))},
                   "analysis/b5_outcome_triggered.py")


if __name__ == "__main__":
    main()
