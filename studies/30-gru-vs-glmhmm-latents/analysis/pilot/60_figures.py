"""Stage 6: two scratch figures (SVG, editable text).

(a) per-state R^2 of the predictive state posterior by feature set: all trials and
    within-session (soft-label logistic probe; ridge-ALR shown as open markers).
(b) one example held-out-fold session: lab session plot (choices, rewards, block reward
    probabilities) on top; below, the GLM-HMM predictive posterior vs the latent probe's
    out-of-fold decoded posterior, and vs the history probe.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("/allen/aind/scratch/han.hou/pilots/study30-probe/out")
plt.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "lines.linewidth": 0.8})
MM = 1 / 25.4
STATE_COLORS = ["#4C72B0", "#DD8452", "#55A868"]
SETS = ["1_gru_latents", "2_gru_logits", "3_history", "4_latents+history", "5_untrained_latents", "6_trial_index"]
LABELS = ["GRU latents", "GRU logits", "History", "Latents+history", "Untrained GRU", "Trial index"]


def fig_r2(res, names):
    fig, axes = plt.subplots(1, 2, figsize=(175 * MM, 50 * MM), sharey=True)
    x = np.arange(len(SETS))
    w = 0.26
    for ax, key, title in [(axes[0], "r2", "All trials"), (axes[1], "r2_within", "Within session (demeaned)")]:
        for k in range(3):
            vals = [res[f"pred|{s}"]["logistic"][key][k] for s in SETS]
            rv = [res[f"pred|{s}"]["ridge_alr"][key][k] for s in SETS]
            ax.bar(x + (k - 1) * w, vals, w, color=STATE_COLORS[k], label=names[k])
            ax.scatter(x + (k - 1) * w, rv, s=6, facecolors="none", edgecolors="k", linewidths=0.4, zorder=3,
                       label="ridge-ALR" if k == 0 else None)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, rotation=30, ha="right")
        ax.set_title(title)
    axes[0].set_ylabel("R² of p(state | past choices)")
    axes[0].set_ylim(min(-0.1, axes[0].get_ylim()[0]), 1)
    axes[1].legend(frameon=False, fontsize=5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(OUT / "fig_probe_r2.svg")


def fig_example(j, oof, names):
    from aind_dynamic_foraging_basic_analysis.plot.plot_foraging_session import plot_foraging_session
    m = j.merge(oof, on=["subject_id", "ses_idx", "trial"], validate="one_to_one")
    P = m[[f"pred_{k}" for k in range(3)]].to_numpy()
    sd = pd.DataFrame(P).groupby(m["ses_idx"].to_numpy()).std().max(1)
    n = m.groupby("ses_idx").size()
    ok = sd[(n.reindex(sd.index) >= 300) & (n.reindex(sd.index) <= 700)]
    sid = ok.sort_values().index[-1]  # session with the most posterior movement
    g = m[m["ses_idx"] == sid].sort_values("trial")
    fig = plt.figure(figsize=(150 * MM, 105 * MM))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.3, 1, 1], hspace=0.55, top=0.85)
    ax0 = fig.add_subplot(gs[0])
    choice = np.where(g["choice"].to_numpy() == 2, np.nan, g["choice"].to_numpy().astype(float))
    _, sess_axes = plot_foraging_session(choice, g["reward"].to_numpy(),
                          [g["reward_probabilityL"].to_numpy(), g["reward_probabilityR"].to_numpy()],
                          fitted_data=g["choice_prob_1"].to_numpy() / (g["choice_prob_0"].to_numpy() + g["choice_prob_1"].to_numpy()),
                          ax=ax0)
    fig.text(0.01, 0.995, f"{sid} (mouse {g['subject_id'].iloc[0]}, stage {g['stage'].iloc[0]}); "
             "blue 'model' line = GRU p(right | respond)", fontsize=6, va="top")
    t = g["trial_in_session"].to_numpy()
    for row, (prefix, lab) in enumerate([("probe_latents_", "GRU-latent probe"), ("probe_history_", "History probe")], start=1):
        ax = fig.add_subplot(gs[row])
        for k in range(3):
            ax.plot(t, g[f"pred_{k}"], color=STATE_COLORS[k], lw=1.0, label=f"GLM-HMM {names[k]}")
            ax.plot(t, g[f"{prefix}{k}"], color=STATE_COLORS[k], lw=0.7, ls="--")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("p(state)")
        ax.set_title(f"solid: GLM-HMM predictive posterior; dashed: {lab} (out-of-fold)", fontsize=6)
        if row == 1:
            ax.legend(frameon=False, fontsize=5, ncol=3, loc="upper right")
    ax.set_xlabel("Trial in session")
    fig.savefig(OUT / "fig_example_session.svg")
    return sid


def main():
    pr = json.loads((OUT / "probe_results.json").read_text())
    names = json.loads((OUT / "state_names.json").read_text())["names"]
    fig_r2(pr["results"], names)
    j = pd.read_parquet(OUT / "joined_trials.parquet",
                        columns=["subject_id", "ses_idx", "trial", "trial_in_session", "choice", "reward", "stage",
                                 "reward_probabilityL", "reward_probabilityR", "choice_prob_0", "choice_prob_1",
                                 "pred_0", "pred_1", "pred_2"])
    oof = pd.read_parquet(OUT / "probe_oof_predictions.parquet")
    sid = fig_example(j, oof, names)
    (OUT / "figures.json").write_text(json.dumps({"example_session": sid}, indent=1))


if __name__ == "__main__":
    main()
