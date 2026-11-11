"""D1b: where repeated constant inputs actually take real states, next to D1's one-step flow.

    PYTHONPATH=<wrapper>/code python analysis/d1b_convergence.py

D1 draws the mean one-step Δh of real states (binned in PC1–PC2). Arrows drawn at full length
end where a *single* step lands, which is not where the dynamics converge: after the first few
steps the state keeps drifting slowly (tens to hundreds of trials) toward the attractor. Here
300 real states are iterated under each constant input (all-sessions mean context, float64)
for 600 steps; their paths are drawn over the data cloud, with the endpoints after 1, 3, 10,
30, 100 and 600 steps and D's fixed points. Writes ``fig_D1b_convergence.png``.
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

N_STARTS, N_STEPS = 300, 600
MARKS = [1, 3, 10, 30, 100, 600]
INPUTS = {"R rewarded": (1.0, 1.0), "R unrewarded": (1.0, 0.0), "L rewarded": (0.0, 1.0), "L unrewarded": (0.0, 0.0)}


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
    cloud = (states[np.random.default_rng(1).choice(len(states), 60_000, replace=False)] - mean) @ comp.T
    h0 = jnp.asarray(states[np.random.default_rng(0).choice(len(states), N_STARTS, replace=False)], jnp.float64)
    del states
    run = jax.jit(lambda h, u: jax.lax.scan(lambda x, _: (step.step(x, u, c), step.step(x, u, c)), h, None, length=N_STEPS)[1])
    fig, axes = plt.subplots(1, 4, figsize=(28, 7.5), sharex=True, sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, (name, u) in zip(axes, INPUTS.items()):
        traj = np.asarray(run(h0, jnp.asarray(u)))  # [steps, starts, H]
        p = (np.concatenate([np.asarray(h0)[None], traj]) - mean) @ comp.T  # [steps+1, starts, 2]
        ax.scatter(cloud[:, 0], cloud[:, 1], s=1, color="0.88", linewidths=0, rasterized=True)
        for k in range(0, N_STARTS, 3):
            ax.plot(p[:, k, 0], p[:, k, 1], color="0.35", lw=0.4, alpha=0.5)
        for j, m in enumerate(MARKS):
            ax.scatter(p[m, :, 0], p[m, :, 1], s=10, color=cmap(j / (len(MARKS) - 1)), zorder=3,
                       label=f"after {m} step{'s' if m > 1 else ''}")
        for fp in fps[name]["fixed_points"]:
            ax.scatter(fp["pc"][0], fp["pc"][1], s=260, marker="*" if fp["stable"] else "X",
                       color="#D55E00" if fp["stable"] else "#0072B2", edgecolors="k", zorder=5)
        ax.set_title(f"constant input: {name}", fontsize=13)
        ax.set_xlabel("PC1")
        ax.legend(frameon=False, fontsize=9, loc="lower left" if name.startswith("R") else "lower right", markerscale=2)
    axes[0].set_ylabel("PC2")
    fig.suptitle("Repeating one input from 300 real states (grey paths; colour = number of steps); ★ D's stable fixed "
                 "points, ✕ unstable. One step covers most of the distance; the rest is a slow drift.", fontsize=13)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_D1b_convergence.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
