"""Figure for D-RT: each input's stable fixed point vs the RT input, seeds side by side.

    python analysis/fig_d_rt_fixed_points.py

Reads ``d_rt_fixed_points_R-s0.json`` / ``_R-s1.json``. Columns: the attractor's switch-logit
(logit toward the side not in the input; more negative = deeper "stay"), and the moduli of the
two leading Jacobian eigenvalues there (1 = no decay; τ = −1/ln|λ| trials). Every stable fixed
point is drawn; a second marker at the same RT means the input has more than one attractor.
Writes ``fig_D_RT_fixed_points.png`` and ``d_rt_fixed_points_summary.json``.
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
from _pk03 import ANALYSIS, apply_presentation_style, figure_path, write_json  # noqa: E402

PRODUCER = "analysis/fig_d_rt_fixed_points.py"
KEYS = ["R-s0", "R-s1"]
INPUT_COLORS = {"R rewarded": "#0072B2", "R unrewarded": "#56B4E9", "L rewarded": "#D55E00", "L unrewarded": "#E69F00"}


def main() -> None:
    apply_presentation_style()
    fig, axes = plt.subplots(len(KEYS), 3, figsize=(19, 5 * len(KEYS)), sharex=True, sharey="col",
                             gridspec_kw={"wspace": 0.3})
    summary = {}
    for i, key in enumerate(KEYS):
        d = json.loads((ANALYSIS / f"d_rt_fixed_points_{key}.json").read_text())["fixed_points"]
        summary[key] = {}
        for name, color in INPUT_COLORS.items():
            rows = [(e["rt"], p) for e in d.values() if e["input"] == name for p in e["stable"]]
            counts = {e["rt"]: len(e["stable"]) for e in d.values() if e["input"] == name}
            summary[key][name] = {"n_stable_by_rt": counts,
                                  "switch_logit": [(rt, p["switch_logit"]) for rt, p in rows],
                                  "lambda1": [(rt, p["eig_moduli_top3"][0]) for rt, p in rows],
                                  "lambda2": [(rt, p["eig_moduli_top3"][1]) for rt, p in rows]}
            single = all(n == 1 for n in counts.values())
            for col, field in enumerate(("switch_logit", "lambda1", "lambda2")):
                pts = sorted(summary[key][name][field])
                x, y = np.array(pts).T if pts else ([], [])
                axes[i, col].plot(x, y, color=color, lw=2 if single else 0, marker="o", ms=5, label=f"input: {name}")
        axes[i, 0].set_ylabel(f"{key}\nattractor switch-logit")
        axes[i, 1].set_ylabel("|λ₁| (slowest mode)")
        axes[i, 2].set_ylabel("|λ₂|")
    for ax in axes[-1]:
        ax.set_xlabel("constant RT input (sd)")
    axes[0, 0].set_title("where the state settles (choice readout)")
    axes[0, 1].set_title("slowest relaxation at the attractor")
    axes[0, 2].set_title("second-slowest relaxation")
    axes[0, 0].legend(frameon=False, fontsize=11)
    fig.suptitle("D-RT  Attractor of each constant input as a function of RT (mean context)", y=1.0)
    fig.savefig(figure_path("fig_D_RT_fixed_points.png"), bbox_inches="tight")
    plt.close(fig)
    write_json("d_rt_fixed_points_summary.json", summary, PRODUCER)


if __name__ == "__main__":
    main()
