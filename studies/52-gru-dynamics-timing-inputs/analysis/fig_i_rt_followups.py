"""Figures for the tier-3 follow-ups, seeds side by side (offline; reads the stage-A npz files).

    python analysis/fig_i_rt_followups.py

  F1  gain vs push: a(v), b(v) of s_v = a(v) + b(v)·s_0, and Δs = s_v − s_0 binned by s_0.
  F2  reward-flip effect vs that step's RT input (same input vector).
  F3  reward-flip effect vs the previous step's RT input, with the current step's RT input
      held at 0 (the two RTs correlate, ρ ≈ 0.58, so leaving the current RT at its observed
      value would mix in F2's effect).

Rows are seeds of R (R-s0, R-s1). Writes ``fig_F1_gain_vs_push.png``,
``fig_F2F3_rt_and_outcome.png`` and ``i_rt_followups_summary.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pk03 import ANALYSIS, CACHE, REWARDED_COLOR, UNREWARDED_COLOR, apply_presentation_style, figure_path, write_json  # noqa: E402
import _pk03  # noqa: E402

PRODUCER = "analysis/fig_i_rt_followups.py"
KEYS = ["R-s0", "R-s1"]
OUTCOMES = ((1.0, REWARDED_COLOR, "-", "t−1 rewarded"), (0.0, UNREWARDED_COLOR, "--", "t−1 unrewarded"))
PROBE_V = (2.0, -1.5)


def _binned(x, y, n=10):
    bins = pd.qcut(x, n, labels=False, duplicates="drop")
    g = pd.DataFrame({"x": x, "y": y, "b": bins}).groupby("b")
    return g["x"].median().to_numpy(), g["y"].median().to_numpy(), g["y"].quantile(0.25).to_numpy(), g["y"].quantile(0.75).to_numpy()


def main() -> None:
    _pk03.use_keys(*KEYS)
    apply_presentation_style()
    data = {k: np.load(CACHE / f"i_rt_followups_{k}.npz") for k in KEYS}
    fits = {k: _pk03.read_json(ANALYSIS / f"i_rt_followups_{k}.json")["F1_fits_by_reward"] for k in KEYS}
    summary = {}

    fig, axes = plt.subplots(len(KEYS), 3, figsize=(20, 5 * len(KEYS)), sharex="col", sharey="col", gridspec_kw={"wspace": 0.3})
    for i, key in enumerate(KEYS):
        d = data[key]
        v = d["sweep"]
        summary[key] = {"F1": {}, "F2": {}, "F3": {}}
        for r, color, ls, label in OUTCOMES:
            f = fits[key][str(int(r))]
            axes[i, 0].plot(v, f["a"], color=color, ls=ls, lw=2, label=label)
            axes[i, 1].plot(v, f["b"], color=color, ls=ls, lw=2, label=label)
            m = d["reward"] == r
            for probe, marker in zip(PROBE_V, ("o", "s")):
                j = int(np.argmin(np.abs(v - probe)))
                x, y, lo, hi = _binned(d["s0"][m], d["s"][j, m] - d["s0"][m])
                axes[i, 2].plot(x, y, color=color, ls=ls, lw=2, marker=marker, ms=5,
                                label=f"{label}, RT {v[j]:+.1f} sd")
                summary[key]["F1"][f"{label} | v={v[j]:+.1f}"] = {"s0_bin_median": x, "delta_median": y}
        for ax in axes[i]:
            ax.axhline(0, color="#888888", lw=1)
        axes[i, 1].axhline(1, color="#888888", lw=1, ls=":")
        axes[i, 0].set_ylabel(f"{key}\nintercept a(v) (push)")
        axes[i, 1].set_ylabel("slope b(v) (gain; 1 = none)")
        axes[i, 2].set_ylabel("Δ switch-logit vs RT 0")
        axes[i, 2].axvline(0, color="#888888", lw=1, ls=":")
    for ax in axes[-1, :2]:
        ax.set_xlabel("RT input v (sd; 0 = population mean)")
    axes[-1, 2].set_xlabel("s₀: switch-logit with RT input 0")
    axes[0, 0].set_title("push: shift common to all states")
    axes[0, 1].set_title("gain: scaling of the current preference")
    axes[0, 2].set_title("raw change vs current preference (decile medians;\n98–99% of states favour staying, s₀ < 0)", fontsize=13)
    axes[0, 0].legend(frameon=False, fontsize=11)
    axes[0, 2].legend(frameon=False, fontsize=10)
    fig.suptitle("F1  Is RT a push toward switching or a gain on the current preference?  s_v = a(v) + b(v)·s₀", y=1.0)
    fig.savefig(figure_path("fig_F1_gain_vs_push.png"), bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(len(KEYS), 2, figsize=(14, 5 * len(KEYS)), sharex="col", sharey=True)
    for i, key in enumerate(KEYS):
        d = data[key]
        v = d["sweep"]
        for r, color, ls, label in OUTCOMES:
            m = d["reward"] == r
            eff = d["s_flip"][:, m] - d["s"][:, m]
            q = np.quantile(eff, [0.25, 0.5, 0.75], axis=1)
            axes[i, 0].plot(v, q[1], color=color, ls=ls, lw=2, label=f"{label} → flipped")
            axes[i, 0].fill_between(v, q[0], q[2], color=color, alpha=0.15, lw=0)
            summary[key]["F2"][label] = {"v": v, "median": q[1]}
            j0 = int(np.argmin(np.abs(v)))  # current RT input fixed at 0
            flip_obs = d["s_flip"][j0, m] - d["s"][j0, m]
            x, y, lo, hi = _binned(d["prev_rt_input"][m], flip_obs)
            axes[i, 1].plot(x, y, color=color, ls=ls, lw=2, marker="o", ms=5, label=f"{label} → flipped")
            axes[i, 1].fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
            summary[key]["F3"][label] = {"prev_rt_bin_median": x, "median": y,
                                        "spearman": float(pd.Series(d["prev_rt_input"][m]).corr(pd.Series(flip_obs), method="spearman"))}
        for ax in axes[i]:
            ax.axhline(0, color="#888888", lw=1)
        axes[i, 0].set_ylabel(f"{key}\nreward-flip effect (Δ switch-logit)")
    axes[-1, 0].set_xlabel("RT input of the same trial (sd)")
    axes[-1, 1].set_xlabel("RT input of the trial before (sd); this trial's RT input = 0")
    axes[0, 0].set_title("F2  does this trial's RT change how its outcome counts?")
    axes[0, 1].set_title("F3  does the previous trial's RT change it?")
    axes[0, 0].legend(frameon=False, fontsize=11)
    fig.suptitle("Reward-flip effect (median, IQR over 50k states) — does RT gate learning from outcomes?", y=1.0)
    fig.savefig(figure_path("fig_F2F3_rt_and_outcome.png"), bbox_inches="tight")
    plt.close(fig)
    write_json("i_rt_followups_summary.json", summary, PRODUCER)


if __name__ == "__main__":
    main()
