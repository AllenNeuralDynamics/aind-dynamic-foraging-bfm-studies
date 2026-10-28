"""C1 figure: history kernels (animal, GRU, RL) and the GRU's counterfactual reward kernel.

    python analysis/c1_figure.py

Offline formatter over ``c1_kernels.json`` and ``c1_impulse.json``. Writes
``fig_C1_kernels.png``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402

STYLE = {"animal": ("#222222", 3.0, "-", "animal"), "gru": ("#E69F00", 2.4, "-", "GRU"),
         "bari": ("#4C72B0", 1.4, "--", "Bari"), "ctt": ("#C44E52", 1.4, "--", "CTT"),
         "hattori": ("#55A868", 1.4, "--", "Hattori")}
TITLES = {"RewC": "rewarded choice", "UnrC": "unrewarded choice", "reward_effect": "reward effect (RewC − UnrC)"}


def main() -> None:
    s12.apply_presentation_style()
    kernels = json.loads((s12.ANALYSIS / "c1_kernels.json").read_text())
    impulse = json.loads((s12.ANALYSIS / "c1_impulse.json").read_text())
    lags = np.arange(1, kernels["n_back"] + 1)
    k_imp = np.asarray(impulse["k"])

    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    for ax, term in zip(axes[0], ("RewC", "UnrC", "reward_effect")):
        for source, (color, lw, ls, label) in STYLE.items():
            ax.plot(lags, kernels["pooled"][source][term], color=color, lw=lw, ls=ls, marker="o", ms=3, label=label)
        spread = kernels["per_mouse_summary"]["animal"][term]
        centre = np.asarray(kernels["pooled"]["animal"][term])
        ax.fill_between(lags, centre - 1.96 * np.asarray(spread["sem"]), centre + 1.96 * np.asarray(spread["sem"]),
                        color="#222222", alpha=0.15, lw=0)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_title(TITLES[term], fontsize=14)
        ax.set_xlabel("trials back")
    axes[0, 2].errorbar(k_imp, impulse["all"]["mean"], yerr=1.96 * np.asarray(impulse["all"]["sem"]), color="#E69F00",
                        lw=0, elinewidth=1.5, marker="s", ms=7, mfc="white", label="GRU counterfactual (flip)")
    axes[0, 0].set_ylabel("logistic weight on logit(right)\n(all tasks pooled)")
    axes[0, 2].legend(fontsize=10, frameon=False)
    long_lags = np.arange(1, kernels["n_back_long"] + 1)
    for source in ("animal", "gru"):
        color, lw, ls, label = STYLE[source]
        axes[0, 3].plot(long_lags, kernels["pooled_n_back_long"][source]["reward_effect"], color=color, lw=lw, ls=ls,
                        marker="o", ms=3, label=f"{label}, {kernels['n_back_long']} lags")
    axes[0, 3].plot(lags, kernels["pooled"]["gru"]["reward_effect"], color="#E69F00", lw=1, ls=":", label=f"GRU, {kernels['n_back']} lags")
    axes[0, 3].axhline(0, color="0.7", lw=0.8)
    axes[0, 3].set_title("reward effect, longer history", fontsize=14)
    axes[0, 3].set_xlabel("trials back")
    axes[0, 3].legend(fontsize=10, frameon=False)
    for ax, task in zip(axes[1], s12.TASKS):
        for source in ("animal", "gru"):
            color, lw, ls, label = STYLE[source]
            ax.plot(lags, kernels["by_task"][task][source]["reward_effect"], color=color, lw=lw, ls=ls, marker="o", ms=3,
                    label=f"{label} (logistic)")
        task_imp = impulse["by_task"][task]
        ax.errorbar(k_imp, task_imp["mean"], yerr=1.96 * np.asarray(task_imp["sem"]), color="#E69F00", lw=0,
                    elinewidth=1.5, marker="s", ms=6, mfc="white", label="GRU counterfactual")
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_title(f"{task}: reward effect", fontsize=13)
        ax.set_xlabel("trials back / later")
    axes[1, 0].set_ylabel("effect on logit(right)")
    axes[1, 0].legend(fontsize=10, frameon=False)
    fig.suptitle(f"History kernels ({kernels['design']} design, {kernels['n_back']} trials back; shading: animal pooled "
                 "fit ± 1.96 SEM across mice). Counterfactual: flip one trial's reward, all later inputs as observed.",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_C1_kernels.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
