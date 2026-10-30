"""D1d: D1's flow field with arrow length scaled to the step size — linearly, and on a log scale.

    PYTHONPATH=<wrapper>/code python analysis/d1d_flow_scaled.py

Same one-step Δh as D1 (20,000 real states, all-sessions mean context), binned in PC1–PC2.
Top row: length ∝ median |Δh| of the bin (linear, shrunk so no arrow leaves its bin), so arrows
near the attractors are as short as the dynamics there. Bottom row: length ∝ log10 of the
median step size, compressing the ~1000× range so the slow drift near the attractors stays
visible. Directions are the bin's mean PC1–PC2 displacement. Writes ``fig_D1d_flow_scaled.png``.
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
from d1c_flow_two_scales import INPUTS, N_SAMPLE, binned  # noqa: E402

LOG_MIN, LOG_MAX = -3.0, 0.6  # log10 step-size range mapped onto arrow length


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
    rng = np.random.default_rng(0)
    sample = jnp.asarray(states[rng.choice(len(states), N_SAMPLE, replace=False)], jnp.float64)
    cloud = (states[rng.choice(len(states), 60_000, replace=False)] - mean) @ comp.T
    del states
    one = jax.jit(lambda h, u: step.step(h, u, c))
    ex, ey = np.linspace(-4, 4, 25), np.linspace(-4.5, 3, 23)
    cell = min(ex[1] - ex[0], ey[1] - ey[0])
    fig, axes = plt.subplots(2, 4, figsize=(28, 13), sharex=True, sharey=True)
    for k, (name, u) in enumerate(INPUTS.items()):
        dh = np.asarray(one(sample, jnp.asarray(u)) - sample)
        xy, dxy, speed = (np.asarray(sample) - mean) @ comp.T, dh @ comp.T, np.linalg.norm(dh, axis=1)
        gx, gy, gu, gv, gs = binned(xy, dxy, speed, ex, ey)
        nrm = np.hypot(gu, gv) + 1e-12
        lin = 0.9 * cell * gs / gs.max()
        logl = 0.9 * cell * np.clip((np.log10(gs) - LOG_MIN) / (LOG_MAX - LOG_MIN), 0.03, 1)
        for row, (length, label) in enumerate(((lin, "length ∝ step size (linear)"), (logl, "length ∝ log10 step size"))):
            ax = axes[row, k]
            ax.scatter(cloud[:, 0], cloud[:, 1], s=1, color="0.88", linewidths=0, rasterized=True)
            ax.quiver(gx, gy, length * gu / nrm, length * gv / nrm, angles="xy", scale_units="xy", scale=1,
                      width=0.004, headwidth=4, headlength=4, headaxislength=3.5, color="#222222")
            for fp in fps[name]["fixed_points"]:
                ax.scatter(fp["pc"][0], fp["pc"][1], s=220, marker="*" if fp["stable"] else "X",
                           color="#D55E00" if fp["stable"] else "#0072B2", edgecolors="k", zorder=5)
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4.6, 3)
            ax.set_title(f"{name}: {label}", fontsize=12)
            if row == 1:
                ax.set_xlabel("PC1")
            if k == 0:
                ax.set_ylabel("PC2")
    fig.suptitle("One-step flow (all-sessions mean context) with arrow length scaled to the step size: linear (top) "
                 f"and log10, {10 ** LOG_MIN:g}–{10 ** LOG_MAX:.0f} (bottom). ★ stable, ✕ unstable fixed points", fontsize=14)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_D1d_flow_scaled.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
