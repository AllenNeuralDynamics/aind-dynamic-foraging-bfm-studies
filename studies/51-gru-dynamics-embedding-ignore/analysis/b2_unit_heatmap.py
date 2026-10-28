"""B2: all 128 hidden units over one session, grouped by which readout each unit follows.

    python analysis/b2_unit_heatmap.py

Uses B1's Coupled Without Baiting session (77 ignored trials; the other example sessions
have 1–13). Each unit is assigned to the readout it correlates with more strongly over the
session — ``logit_diff`` (side) or ``engage_logit`` — and sorted within its group by that
correlation. The same counts are reported pooled over a random 200k timesteps, so the
grouping is not an artefact of one session. Writes ``fig_B2_unit_heatmap.png`` and
``b2_unit_heatmap.json``.
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

TASK = "Coupled Without Baiting"
N_POOLED = 200_000
READOUT_COLUMNS = ["choice_logit_0", "choice_logit_1", "choice_logit_2", "logit_diff"]


def unit_correlations(states: np.ndarray, readouts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    z = (states - states.mean(axis=0)) / np.where(states.std(axis=0) > 1e-6, states.std(axis=0), np.inf)
    out = {}
    for name, y in readouts.items():
        yz = (y - y.mean()) / y.std()
        out[name] = z.T @ yz / len(y)
    return out


def summary(corr: dict[str, np.ndarray]) -> dict:
    side, engage = np.abs(corr["side"]), np.abs(corr["engage"])
    return {
        "n_units_abs_corr_gt_0.5_side": int(np.sum(side > 0.5)),
        "n_units_abs_corr_gt_0.5_engage": int(np.sum(engage > 0.5)),
        "n_units_abs_corr_gt_0.5_both": int(np.sum((side > 0.5) & (engage > 0.5))),
        "n_units_follow_side_more": int(np.sum(side >= engage)),
        "n_units_follow_engage_more": int(np.sum(engage > side)),
    }


def main() -> None:
    s13.apply_presentation_style()
    ses = s13.example_sessions()[TASK]
    d = s13.load_trials([*s13.STATE_COLUMNS, *READOUT_COLUMNS, "subject_id", "ses_idx", "trial", "action",
                         "p_model1_left", "p_model1_right", "p_model1_ignore",
                         "reward_probabilityL", "reward_probabilityR"], sessions=[ses]).sort_values("trial")
    states = d[s13.STATE_COLUMNS].to_numpy(dtype=float)
    corr = unit_correlations(states, {"side": d["logit_diff"].to_numpy(float), "engage": d["engage_logit"].to_numpy()})

    follows_engage = np.abs(corr["engage"]) > np.abs(corr["side"])
    side_units = np.flatnonzero(~follows_engage)
    engage_units = np.flatnonzero(follows_engage)
    order = np.r_[side_units[np.argsort(-corr["side"][side_units])],
                  engage_units[np.argsort(-corr["engage"][engage_units])]]

    fig, (ax_top, ax_map) = plt.subplots(2, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [1, 4]})
    t = d["trial"].to_numpy()
    ignored = d["action"].to_numpy() == s13.IGNORE
    ax_top.vlines(t[ignored], 0, 1, color=s13.IGNORE_COLOR, lw=0.6, alpha=0.4)
    ax_top.plot(t, d["p_right_engaged"], color=s13.MODEL_COLOR, lw=1.4, label="P(right | engaged)")
    ax_top.plot(t, d["p_model1_ignore"], color=s13.ENGAGE_COLOR, lw=1.4, label="P(ignore)")
    ax_top.set_ylim(0, 1)
    ax_top.set_ylabel("C3")
    ax_top.legend(loc="lower right", ncol=2, fontsize=10, frameon=False, bbox_to_anchor=(1, 1.0))
    ax_top.set_title(f"{TASK} — mouse {d['subject_id'].iloc[0]}, {ses.split('__')[-1]} "
                     f"({len(d)} trials, {ignored.sum()} ignored, grey)", fontsize=13, loc="left")
    image = ax_map.imshow(states[:, order].T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                          extent=(t[0] - 0.5, t[-1] + 0.5, s13.HIDDEN - 0.5, -0.5), interpolation="nearest")
    ax_map.axhline(len(side_units) - 0.5, color="k", lw=1.5)
    ax_map.set_ylabel(f"unit: top {len(side_units)} follow side (by corr. with logit_diff),\n"
                      f"bottom {len(engage_units)} follow engagement (by corr. with engage_logit)", fontsize=12)
    ax_map.set_xlabel("trial")
    fig.colorbar(image, ax=ax_map, fraction=0.02, pad=0.01, label="hidden state")
    fig.tight_layout()
    fig.savefig(s13.figure_path("fig_B2_unit_heatmap.png"))
    plt.close(fig)

    pooled = s13.load_trials([*s13.STATE_COLUMNS, *READOUT_COLUMNS])
    sub = s13.subsample_rows(len(pooled), N_POOLED, seed=2)
    pooled_corr = unit_correlations(pooled[s13.STATE_COLUMNS].to_numpy(dtype=float)[sub],
                                    {"side": pooled["logit_diff"].to_numpy(float)[sub],
                                     "engage": pooled["engage_logit"].to_numpy()[sub]})
    s13.write_json("b2_unit_heatmap.json", {
        "ses_idx": ses,
        "unit_order": order.tolist(),
        "n_side_group": int(len(side_units)),
        "corr_with_side": corr["side"][order].round(4).tolist(),
        "corr_with_engage": corr["engage"][order].round(4).tolist(),
        "session": summary(corr),
        "pooled": {"n_timesteps": int(len(sub)), **summary(pooled_corr),
                   "corr_side_engage_readouts": float(np.corrcoef(pooled["logit_diff"].to_numpy(float)[sub],
                                                                  pooled["engage_logit"].to_numpy()[sub])[0, 1])},
        "frac_values_abs_gt_0.95": float(np.mean(np.abs(states) > 0.95)),
    }, "analysis/b2_unit_heatmap.py")


if __name__ == "__main__":
    main()
