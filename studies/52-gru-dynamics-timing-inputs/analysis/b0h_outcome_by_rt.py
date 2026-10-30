"""B0h: in behaviour, does an outcome count less after a very fast response?

    python analysis/b0h_outcome_by_rt.py

The behavioural counterpart of r3's F2 (in R, flipping the reward of a very fast trial moves
the next choice ~45% less). Per within-mouse decile of the RT on trial t−1, the reward effect
on switching at t is logit P(switch | t−1 unrewarded) − logit P(switch | t−1 rewarded), for
the mice and for each model's mean P(switch) (R-s0, R-s1, C) on the same trials. 95% CIs
from resampling mice. This is MARGINAL: trials with fast vs slow RTs also differ in the
state the animal is in. C never sees RT, so any RT dependence in C's curve comes from such
correlates; the R − C difference is the part attributable to RT.

RT-present sessions only. Writes ``fig_B0h_outcome_by_rt.png`` and ``b0h_outcome_by_rt.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pk03 import (  # noqa: E402
    N_BOOT, apply_presentation_style, figure_path, load_paired_trials, rt_bins, write_json,
)

PRODUCER = "analysis/b0h_outcome_by_rt.py"
SOURCES = {"Mice": "switch", "RT model (R)": "p_switch", "RT model, seed 1": "p_switch_seed",
           "Choice/reward model (C)": "p_switch_C"}


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main() -> None:
    apply_presentation_style()
    df = load_paired_trials(rt_present_only=True, with_seed=True)
    df["bin"] = rt_bins(df, "rt_lag1", within_mouse=True)
    df = df.dropna(subset=["bin", "reward_lag1", "switch"])
    cols = list(SOURCES.values())
    sums = df.groupby(["subject_id", "bin", "reward_lag1"])[cols].sum()
    counts = df.groupby(["subject_id", "bin", "reward_lag1"]).size()
    mice = df["subject_id"].unique()
    s = {c: sums[c].unstack(["bin", "reward_lag1"]).reindex(mice).fillna(0).to_numpy() for c in cols}  # [mice, 20]
    n = counts.unstack(["bin", "reward_lag1"]).reindex(mice).fillna(0).to_numpy()
    keys = list(counts.unstack(["bin", "reward_lag1"]).columns)
    rng = np.random.default_rng(0)
    w = np.stack([np.bincount(rng.integers(0, len(mice), len(mice)), minlength=len(mice)) for _ in range(N_BOOT)])
    bins = sorted({b for b, _ in keys})

    def effect(sum_vec, n_vec):
        p = dict(zip(keys, sum_vec / np.maximum(n_vec, 1)))
        return np.array([_logit(p[(b, 0.0)]) - _logit(p[(b, 1.0)]) for b in bins])

    result = {}
    fig, axes = plt.subplots(1, len(SOURCES), figsize=(19, 4.8), sharey=True)
    for ax, (label, col) in zip(axes, SOURCES.items()):
        point = effect(s[col].sum(0), n.sum(0))
        boot = np.stack([effect(wi @ s[col], wi @ n) for wi in w])
        lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
        x = np.array(bins) + 1
        ax.plot(x, point, color="#222222", lw=2, marker="o", ms=5)
        ax.fill_between(x, lo, hi, color="#222222", alpha=0.15, lw=0)
        ax.set_title(label)
        ax.set_xlabel("RT on t−1 (mouse decile)")
        ax.set_xticks([1, 5, 10])
        result[label] = {"effect": point, "ci_lo": lo, "ci_hi": hi}
    axes[0].set_ylabel("reward effect on switching (log-odds)\nunrewarded − rewarded on t−1")
    fig.suptitle("B0h  Does the previous outcome count less after a fast or slow response? (marginal)", y=1.03)
    fig.savefig(figure_path("fig_B0h_outcome_by_rt.png"), bbox_inches="tight")
    plt.close(fig)
    r_minus_c = result["RT model (R)"]["effect"] - result["Choice/reward model (C)"]["effect"]
    write_json("b0h_outcome_by_rt.json", {"deciles": [int(b) + 1 for b in bins], "by_source": result,
                                           "R_minus_C": r_minus_c}, PRODUCER)
    for label, r in result.items():
        print(label, np.round(r["effect"], 3))


if __name__ == "__main__":
    main()
