"""D1f: the four inputs' fixed points together, then each input's flow field, in B3h's input colours.

    PYTHONPATH=<wrapper>/code python analysis/d1f_fixed_points_flow.py

All-sessions mean context, PC1–PC2. Only behaviourally relevant fixed points are drawn: those
that real states reach — N_STARTS real states iterated N_ITER steps under each constant input
(float64, as D1b), each end state assigned to its nearest D fixed point; a fixed point is kept
if any start ends on it (unstable points and the deep "L rewarded" basin are never reached).
Panel 1: the kept fixed points of every input, coloured by input, over the state cloud.
Panels 2–5: one input each — D1's full-length arrows (one-step Δh from N_SAMPLE real states,
mean per PC1–PC2 bin) and that input's kept fixed points, in the input's colour.
Writes ``d1f_fixed_points_flow.json`` (which points were reached),
``fig_D1f_fixed_points_flow.png`` and ``fig_D1f_flow_2x2.png`` (panels 2–5 alone, rows = side,
columns = outcome).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402
from b3h_pc_overview import COLORS  # noqa: E402
from d1c_flow_two_scales import INPUTS, N_SAMPLE, binned  # noqa: E402

XLIM, YLIM = (-4, 4), (-4.1, 3)
N_STARTS, N_ITER = 300, 600  # as D1b


def draw_fixed_points(ax, fps: list[dict], colour: str, size: float = 420) -> None:
    for fp in fps:
        ax.scatter(fp["pc"][0], fp["pc"][1], s=size, marker="*" if fp["stable"] else "X", color=colour,
                   edgecolors="k", linewidths=1.4, zorder=5)


def main() -> None:
    import jax  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp  # noqa: PLC0415

    from post_training_analysis.gru_step import GruStep  # noqa: PLC0415

    s12.apply_presentation_style()
    step = GruStep.from_params({k: {n: np.asarray(a, np.float64) for n, a in v.items()}
                                for k, v in s12.best_params().items() if isinstance(v, dict)})
    c = jnp.asarray(np.load(s12.CACHE / f"contexts_{s12.MODEL_KEY}.npz")["contexts"].mean(0), jnp.float64)
    pca = s12.load_pca()
    mean, comp = pca["mean"], pca["components"][:2]
    fps = json.loads((s12.ANALYSIS / "d_dynamics.json").read_text())["contexts"]["all sessions"]["inputs"]
    states = s12.load_trials(s12.STATE_COLUMNS).to_numpy(np.float32)
    rng = np.random.default_rng(0)  # as D1d
    sample = jnp.asarray(states[rng.choice(len(states), N_SAMPLE, replace=False)], jnp.float64)
    cloud = (states[rng.choice(len(states), 60_000, replace=False)] - mean) @ comp.T
    del states
    one = jax.jit(lambda h, u: step.step(h, u, c))

    # Which fixed points do real states reach under each constant input?
    starts = sample[:N_STARTS]
    kept, reached = {}, {}
    for name, u in INPUTS.items():
        h = starts
        for _ in range(N_ITER):
            h = one(h, jnp.asarray(u))
        end = (np.asarray(h) - mean) @ pca["components"][:6].T
        pts = np.array([fp["pc"] for fp in fps[name]["fixed_points"]])
        nearest = np.argmin(np.linalg.norm(end[:, None, :] - pts[None], axis=2), axis=1)
        counts = np.bincount(nearest, minlength=len(pts))
        reached[name] = [{"pc12": fp["pc"][:2], "stable": fp["stable"], "logit_right": fp["logit_right"],
                          "n_starts_reaching": int(n)} for fp, n in zip(fps[name]["fixed_points"], counts)]
        kept[name] = [fp for fp, n in zip(fps[name]["fixed_points"], counts) if n > 0]
        print(name, [(round(fp["logit_right"], 2), fp["stable"], int(n)) for fp, n in zip(fps[name]["fixed_points"], counts)], flush=True)
    ex, ey = np.linspace(-4, 4, 25), np.linspace(-4.5, 3, 23)

    fig, axes = plt.subplots(1, 5, figsize=(36, 8.4), sharex=True, sharey=True, layout="constrained")
    for ax in axes:
        ax.scatter(cloud[:, 0], cloud[:, 1], s=1, color="0.88", linewidths=0, rasterized=True)
    for name in INPUTS:
        draw_fixed_points(axes[0], kept[name], COLORS[name])
    axes[0].set_title("fixed points", fontsize=26)
    handles = [Line2D([], [], marker="o", ls="none", ms=14, color=COLORS[n], label=n) for n in INPUTS]
    axes[0].legend(handles=handles, loc="lower left", frameon=False, fontsize=17)
    flows = {}
    for name, u in INPUTS.items():
        dh = np.asarray(one(sample, jnp.asarray(u)) - sample)
        xy, dxy, speed = (np.asarray(sample) - mean) @ comp.T, dh @ comp.T, np.linalg.norm(dh, axis=1)
        flows[name] = tuple(binned(xy, dxy, speed, ex, ey))  # binned returns a one-shot map

    def draw_flow(ax, name: str, width: float) -> None:
        gx, gy, gu, gv, _ = flows[name]
        ax.quiver(gx, gy, gu, gv, angles="xy", scale_units="xy", scale=1.0, width=width, color=COLORS[name])
        draw_fixed_points(ax, kept[name], COLORS[name])
        ax.set_title(f"input: {name}", fontsize=26, color=COLORS[name])

    for ax, name in zip(axes[1:], INPUTS):
        draw_flow(ax, name, 0.004)
    for ax in axes:
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_xlabel("PC1", fontsize=23)
        ax.tick_params(labelsize=19)
    axes[0].set_ylabel("PC2", fontsize=23)
    fig.savefig(s12.figure_path("fig_D1f_fixed_points_flow.png"), bbox_inches="tight")
    plt.close(fig)

    # The four flow panels alone, 2 × 2: rows = side (R, L), columns = outcome (rewarded, unrewarded).
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), sharex=True, sharey=True, layout="constrained")
    for ax, name in zip(axes.ravel(), INPUTS):
        ax.scatter(cloud[:, 0], cloud[:, 1], s=1, color="0.88", linewidths=0, rasterized=True)
        draw_flow(ax, name, 0.0035)
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.tick_params(labelsize=19)
    for ax in axes[1]:
        ax.set_xlabel("PC1", fontsize=23)
    for ax in axes[:, 0]:
        ax.set_ylabel("PC2", fontsize=23)
    fig.savefig(s12.figure_path("fig_D1f_flow_2x2.png"), bbox_inches="tight")
    plt.close(fig)
    s12.write_json("d1f_fixed_points_flow.json", {"n_starts": N_STARTS, "n_iter": N_ITER, "context": "all sessions",
                                                   "fixed_points": reached}, "analysis/d1f_fixed_points_flow.py")


if __name__ == "__main__":
    main()
