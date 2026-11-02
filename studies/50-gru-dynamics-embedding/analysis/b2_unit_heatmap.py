"""B2: all 128 hidden units over one session, sorted by correlation with the model's logit.

    python analysis/b2_unit_heatmap.py

Uses B1's Uncoupled Baiting example session. Writes ``fig_B2_unit_heatmap.png`` and
``b2_unit_heatmap.json`` (unit order, per-unit correlation, and activity summaries).
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

TASK = "Uncoupled Baiting"


def main() -> None:
    s12.apply_presentation_style()
    index = s12.load_trials(["ses_idx", "trial_idx", "task"])
    ses = s12.example_sessions(index)[TASK]
    d = s12.load_trials([*s12.STATE_COLUMNS, "subject_id", "ses_idx", "trial_idx", "action", "reward",
                         "logit_diff", "p_model1_right", "reward_probabilityL", "reward_probabilityR"],
                        sessions=[ses]).sort_values("trial_idx")
    states = d[s12.STATE_COLUMNS].to_numpy(dtype=float)  # trials x units
    logit = d["logit_diff"].to_numpy(dtype=float)

    std = states.std(axis=0)
    corr = np.array([np.corrcoef(states[:, u], logit)[0, 1] if std[u] > 1e-6 else 0.0
                     for u in range(s12.HIDDEN)])
    order = np.argsort(-corr)

    fig, (ax_top, ax_map) = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                                         gridspec_kw={"height_ratios": [1, 4]})
    t = d["trial_idx"].to_numpy()
    ax_top.plot(t, d["p_model1_right"], color=s12.MODEL_COLOR, lw=1.4, label="model P(right)")
    ax_top.plot(t, d["reward_probabilityR"], color=s12.RIGHT_COLOR, lw=1.2, alpha=0.6, label="P(reward | right)")
    ax_top.plot(t, d["reward_probabilityL"], color=s12.LEFT_COLOR, lw=1.2, alpha=0.6, label="P(reward | left)")
    ax_top.set_ylim(0, 1)
    ax_top.set_ylabel("P")
    ax_top.legend(loc="lower right", ncol=3, fontsize=10, frameon=False, bbox_to_anchor=(1, 1.0))
    ax_top.set_title(f"{TASK} — mouse {d['subject_id'].iloc[0]}, {ses.split('__')[-1]}",
                     fontsize=13, loc="left")
    image = ax_map.imshow(states[:, order].T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                          extent=(t[0] - 0.5, t[-1] + 0.5, s12.HIDDEN - 0.5, -0.5), interpolation="nearest")
    ax_map.set_ylabel("unit (sorted by corr. with logit, + → −)")
    ax_map.set_xlabel("trial")
    fig.colorbar(image, ax=ax_map, fraction=0.02, pad=0.01, label="hidden state")
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_B2_unit_heatmap.png"))
    plt.close(fig)

    s12.write_json("b2_unit_heatmap.json", {
        "ses_idx": ses,
        "unit_order": order.tolist(),
        "corr_with_logit": corr[order].round(4).tolist(),
        "n_units_abs_corr_gt_0.5": int(np.sum(np.abs(corr) > 0.5)),
        "n_units_std_lt_0.01": int(np.sum(std < 0.01)),
        "frac_values_abs_gt_0.95": float(np.mean(np.abs(states) > 0.95)),
    }, "analysis/b2_unit_heatmap.py")


if __name__ == "__main__":
    main()
