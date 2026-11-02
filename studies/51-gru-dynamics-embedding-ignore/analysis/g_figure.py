"""G1 figure: the headline results for C3 seeds 0–2 side by side.

    python analysis/g_figure.py

Offline: reads ``g_seeds.json`` and ``g_dynamics.json``. Panels: a. C3 − C per-trial L/R
advantage by gap; b. P(repeat previous engaged choice) by gap, each seed vs the animals and C;
c. lag-1 kernel weight after k ignores (fraction of no gap); d. |side| after k ignores
(fraction of start). Writes ``fig_G1_seeds.png``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk02 as s13  # noqa: E402

SEED_COLORS = {"C3-s0": s13.ENGAGE_COLOR, "C3-s1": "#B07CC6", "C3-s2": "#5E2A78"}


def main() -> None:
    s13.apply_presentation_style()
    g = json.loads((s13.ANALYSIS / "g_seeds.json").read_text())["metrics"]
    dyn = json.loads((s13.ANALYSIS / "g_dynamics.json").read_text())["seeds"]
    fig, axes = plt.subplots(1, 4, figsize=(26, 6.2))
    for key, m in g.items():
        color = SEED_COLORS[key]
        labels = list(m["by_gap"])
        axes[0].plot(labels, [m["by_gap"][k]["C3_minus_C_per_trial"] for k in labels], color=color, marker="o", lw=2, label=key)
        axes[1].plot(labels, [m["stay_after_gap"][k]["C3"] for k in labels], color=color, marker="o", lw=2, label=key)
        axes[2].plot(list(m["kernel_lag1_ratio_to_gap0"]), [v["RewC"] for v in m["kernel_lag1_ratio_to_gap0"].values()],
                     color=color, marker="o", lw=2, label=f"{key} (rewarded)")
        axes[2].plot(list(m["kernel_lag1_ratio_to_gap0"]), [v["UnrC"] for v in m["kernel_lag1_ratio_to_gap0"].values()],
                     color=color, marker="s", lw=1.2, ls="--")
    first = next(iter(g.values()))
    labels = list(first["stay_after_gap"])
    axes[1].plot(labels, [first["stay_after_gap"][k]["animal"] for k in labels], color="k", marker="x", ms=9, lw=1.8, label="animal")
    axes[1].plot(labels, [first["stay_after_gap"][k]["C"] for k in labels], color=s13.MODEL_COLOR, ls=":", lw=2, label="C (seed 0)")
    for key, d in dyn.items():
        ks = [0, *map(int, d["relaxation_side_fraction_median"])]
        axes[3].plot(ks, [1.0, *d["relaxation_side_fraction_median"].values()], color=SEED_COLORS[key], marker="o", lw=2,
                     label=f"{key}: ignore attractor P(ignore) {d['attractors_mean_context']['ignore']['p_ignore']:.3f}")
    axes[0].axhline(0, color="0.7", lw=0.8)
    axes[0].set(xlabel="ignored trials just before", ylabel="C3 − C log-likelihood per trial (nats)", title="a. advantage over C by gap")
    axes[1].set(xlabel="ignored trials in between", ylabel="P(repeat previous engaged choice)", title="b. stay after a gap")
    axes[2].set(xlabel="ignored trials just before", ylabel="lag-1 weight / no-gap weight", title="c. kernel shrinkage (solid rewarded, dashed unrewarded)")
    axes[3].set(xlabel="consecutive ignore inputs", ylabel="|side logit| / start (median)", title="d. side relaxation under ignores", ylim=(0, 1.05))
    for ax in axes:
        ax.legend(frameon=False, fontsize=10)
    fig.suptitle("Robustness across C3 seeds (C3 vs C seed 0; shared engaged trials)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(s13.figure_path("fig_G1_seeds.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
