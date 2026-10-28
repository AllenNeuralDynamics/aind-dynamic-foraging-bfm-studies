"""D1c: the flow field at two scales, so the arrows point at the attractor.

    PYTHONPATH=<wrapper>/code python analysis/d1c_flow_two_scales.py

D1 drew one-step Δh at full length; those arrows meet where one step lands, not at the fixed
point. Here, per constant input (all-sessions mean context, float64):
  top row     the fast jump: Δh of one step from 20,000 real states, binned in PC1–PC2, drawn as
              unit-length directions coloured by step size, over a map of the median step size
              |Δh| (log scale); the fixed point sits where the speed map has its minimum.
  bottom row  the slow drift, zoomed onto the input's own region: states that have already had
              the input repeated ≥ 5 times (300 real starts, steps 5–600); the next step's Δh binned
              and drawn as a direction coloured by its size, the median path, and D's fixed points.
Writes ``fig_D1c_flow_two_scales.png``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402

INPUTS = {"R rewarded": (1.0, 1.0), "R unrewarded": (1.0, 0.0), "L rewarded": (0.0, 1.0), "L unrewarded": (0.0, 0.0)}
N_SAMPLE, N_STARTS, N_STEPS, FIRST = 20_000, 300, 600, 5
MIN_N = 15


def binned(xy: np.ndarray, d: np.ndarray, speed: np.ndarray, ex: np.ndarray, ey: np.ndarray):
    ix, iy = np.digitize(xy[:, 0], ex) - 1, np.digitize(xy[:, 1], ey) - 1
    gx, gy, u, v, s = [], [], [], [], []
    for i in range(len(ex) - 1):
        for j in range(len(ey) - 1):
            m = (ix == i) & (iy == j)
            if m.sum() >= MIN_N:
                gx.append((ex[i] + ex[i + 1]) / 2)
                gy.append((ey[j] + ey[j + 1]) / 2)
                u.append(d[m, 0].mean())
                v.append(d[m, 1].mean())
                s.append(np.median(speed[m]))
    return map(np.asarray, (gx, gy, u, v, s))


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
    starts = jnp.asarray(states[rng.choice(len(states), N_STARTS, replace=False)], jnp.float64)
    cloud = (states[rng.choice(len(states), 60_000, replace=False)] - mean) @ comp.T
    del states
    one = jax.jit(lambda h, u: step.step(h, u, c))
    run = jax.jit(lambda h, u: jax.lax.scan(lambda x, _: (step.step(x, u, c), step.step(x, u, c)), h, None, length=N_STEPS)[1])
    proj = lambda h: (np.asarray(h) - mean) @ comp.T  # noqa: E731

    fig, axes = plt.subplots(2, 4, figsize=(28, 13.5))
    ex, ey = np.linspace(-4, 4, 25), np.linspace(-4.5, 3, 23)
    for k, (name, u) in enumerate(INPUTS.items()):
        u = jnp.asarray(u)
        # Fast scale: one step from real states.
        dh = np.asarray(one(sample, u) - sample)
        xy, dxy, speed = proj(sample), dh @ comp.T, np.linalg.norm(dh, axis=1)
        gx, gy, gu, gv, gs = binned(xy, dxy, speed, ex, ey)
        ax = axes[0, k]
        ax.scatter(cloud[:, 0], cloud[:, 1], s=1, color="0.9", linewidths=0, rasterized=True)
        sc = ax.scatter(gx, gy, c=gs, s=170, marker="s", cmap="magma_r", norm=LogNorm(0.05, 4), alpha=0.55, linewidths=0)
        nrm = np.hypot(gu, gv) + 1e-12
        ax.quiver(gx, gy, gu / nrm, gv / nrm, angles="xy", scale_units="xy", scale=3.2, width=0.004, color="#222222")
        for fp in fps[name]["fixed_points"]:
            ax.scatter(fp["pc"][0], fp["pc"][1], s=260, marker="*" if fp["stable"] else "X",
                       color="#D55E00" if fp["stable"] else "#0072B2", edgecolors="k", zorder=5)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4.6, 3)
        ax.set_title(f"{name}: one step from real states\n(arrows = direction only; colour = median step size)", fontsize=12)
        ax.set_xlabel("PC1")
        if k == 0:
            ax.set_ylabel("PC2")
        # Slow scale: states already in the input's region.
        traj = np.asarray(run(starts, u))  # [steps, starts, H]
        late = traj[FIRST - 1:-1].reshape(-1, traj.shape[-1])
        nxt = traj[FIRST:].reshape(-1, traj.shape[-1])
        lxy, ldxy = proj(late), (nxt - late) @ comp.T
        lspeed = np.linalg.norm(nxt - late, axis=1)
        lo, hi = np.percentile(lxy, [1, 99], axis=0)
        pad = 0.15 * (hi - lo)
        zx, zy = np.linspace(lo[0] - pad[0], hi[0] + pad[0], 16), np.linspace(lo[1] - pad[1], hi[1] + pad[1], 16)
        gx, gy, gu, gv, gs = binned(lxy, ldxy, lspeed, zx, zy)
        ax = axes[1, k]
        ax.scatter(cloud[:, 0], cloud[:, 1], s=2, color="0.88", linewidths=0, rasterized=True)
        ax.scatter(gx, gy, c=gs, s=260, marker="s", cmap="magma_r", norm=LogNorm(0.0005, 0.5), alpha=0.5, linewidths=0)
        nrm = np.hypot(gu, gv) + 1e-12
        span = min(zx[-1] - zx[0], zy[-1] - zy[0]) / 22
        q = ax.quiver(gx, gy, span * gu / nrm, span * gv / nrm, angles="xy", scale_units="xy", scale=1, width=0.005,
                      color="#222222", label="direction of the next step")
        med = np.median(np.concatenate([proj(starts)[None], proj(traj.reshape(-1, traj.shape[-1])).reshape(N_STEPS, N_STARTS, 2)])[:, :, :], axis=1)
        ax.plot(med[FIRST:, 0], med[FIRST:, 1], color="#0072B2", lw=2.5, label=f"median path, steps {FIRST}–{N_STEPS}")
        for s_ in (FIRST, 10, 30, 100, N_STEPS):
            ax.annotate(str(s_), med[s_], textcoords="offset points", xytext=(5, 5), fontsize=10, color="#0072B2")
        for fp in fps[name]["fixed_points"]:
            ax.scatter(fp["pc"][0], fp["pc"][1], s=320, marker="*" if fp["stable"] else "X",
                       color="#D55E00" if fp["stable"] else "#0072B2", edgecolors="k", zorder=5)
        ax.set_xlim(zx[0], zx[-1])
        ax.set_ylim(zy[0], zy[-1])
        ax.set_title(f"{name}: zoomed, states after ≥ {FIRST} repeats\n(arrows = direction; colour = step size)", fontsize=12)
        ax.set_xlabel("PC1")
        if k == 0:
            ax.set_ylabel("PC2")
        ax.legend(frameon=False, fontsize=10, loc="best")
    fig.colorbar(sc, ax=axes[0, :], fraction=0.012, pad=0.01, label="median |Δh| of one step (log)")
    zsm = plt.cm.ScalarMappable(norm=LogNorm(0.0005, 0.5), cmap="magma_r")
    fig.colorbar(zsm, ax=axes[1, :], fraction=0.012, pad=0.01, label="median |Δh| of the next step (log)")
    fig.suptitle("Flow at two scales (all-sessions mean context): the fast jump into the input's region (top) and the slow drift "
                 "onto its fixed point (bottom, zoomed); ★ stable, ✕ unstable fixed points", fontsize=14)
    fig.savefig(s12.figure_path("fig_D1c_flow_two_scales.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
