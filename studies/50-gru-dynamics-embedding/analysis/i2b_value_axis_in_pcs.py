"""I2b: where the value coordinate lives in PC space.

    python analysis/i2b_value_axis_in_pcs.py

States (150,000 trials) in PC1–PC2, PC1–PC4 and PC2–PC4, coloured by v (I2's value
coordinate). Over each cloud, the projections of the relevant unit axes onto that plane, drawn
from the cloud centre at a common scale (an axis lying fully in the plane has length L): the
shared value axis e_v, the two lobe value axes (β_v ± β_vc), the readout direction r = w_R −
w_L, and I2's choice and reward axes. The share of each axis inside the plane is printed in
the legend. Right panel: each axis's loadings on PCs 1–8. Writes ``fig_I2b_value_axis_in_pcs.png``.
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
from i4_distilled_model import value_coordinate  # noqa: E402

N = 150_000
L = 3.0
PLANES = [(0, 1), (0, 3), (1, 3)]


def main() -> None:
    s12.apply_presentation_style()
    pca = s12.load_pca()
    comp, scores = pca["components"], pca["scores"]
    v = value_coordinate()
    tdr = np.load(s12.keyed_cache("i2_tdr_axes.npz"))
    terms = [str(x) for x in tdr["terms"]]
    q = tdr["basis"]
    w = np.asarray(s12.best_params()["multisubject_gru/~/readout"]["w"], dtype=np.float64)
    unit = lambda x: x / np.linalg.norm(x)  # noqa: E731
    axes_ = {"value axis e_v (shared)": (unit(q[:, terms.index("value")]), "#000000", "-"),
             "R-lobe value axis": (unit(tdr["lobe_r"]), "#0072B2", "--"),
             "L-lobe value axis": (unit(tdr["lobe_l"]), "#D55E00", "--"),
             "readout direction r": (unit(w[:, 1] - w[:, 0]), "#CC79A7", "-"),
             "choice axis": (unit(q[:, terms.index("choice")]), "#009E73", ":"),
             "reward axis": (unit(q[:, terms.index("reward")]), "#E69F00", ":")}
    load = {k: comp[:8] @ a for k, (a, _, _) in axes_.items()}  # loading of each unit axis on each PC
    rng = np.random.default_rng(0)
    idx = rng.choice(len(v), size=N, replace=False)
    lim = np.nanpercentile(np.abs(v[idx]), 99)

    fig, axs = plt.subplots(1, 4, figsize=(33, 8), gridspec_kw={"width_ratios": [1, 1, 1, 1.05]}, layout="constrained")
    for ax, (i, j) in zip(axs[:3], PLANES):
        sc = ax.scatter(scores[idx, i], scores[idx, j], c=v[idx], cmap="RdBu_r", vmin=-lim, vmax=lim, s=1.2, alpha=0.5,
                        linewidths=0, rasterized=True)
        c0 = np.array([np.median(scores[idx, i]), np.median(scores[idx, j])])
        for name, (a, color, ls) in axes_.items():
            d = np.array([load[name][i], load[name][j]])
            share = float(np.sum(d ** 2))
            ax.annotate("", xy=c0 + L * d, xytext=c0, arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.6, "ls": ls,
                                                                  "mutation_scale": 18})
            ax.plot([], [], color=color, ls=ls, lw=2.6, label=f"{name} ({100 * share:.0f}% in plane)")
        ax.set_xlabel(f"PC{i + 1}")
        ax.set_ylabel(f"PC{j + 1}")
        ax.set_aspect("equal")
        ax.set_title(f"PC{i + 1}–PC{j + 1}, coloured by v", fontsize=13)
        ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-lim, lim), cmap="RdBu_r"), ax=axs[2], fraction=0.05, pad=0.02)
    cb.set_label("v (value coordinate; > 0 favours right)")
    ax = axs[3]
    x = np.arange(1, 9)
    width = 0.13
    for k, (name, (_, color, _)) in enumerate(axes_.items()):
        ax.bar(x + (k - 2.5) * width, load[name], width, color=color, label=name)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(x, [f"PC{k}" for k in x])
    ax.set_ylabel("loading (cosine with the PC)")
    ax.set_title("Where each axis points: loadings on PCs 1–8", fontsize=13)
    ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f"The value coordinate in PC space: arrows are each unit axis projected onto the plane (length {L:g} = fully "
                 "in the plane), from the cloud centre", fontsize=14)
    fig.savefig(s12.figure_path("fig_I2b_value_axis_in_pcs.png"), bbox_inches="tight")
    plt.close(fig)
    for name in axes_:
        print(f"{name:26s}", np.round(load[name], 2), "share in PC1–4:", round(float(np.sum(load[name][:4] ** 2)), 2))


if __name__ == "__main__":
    main()
