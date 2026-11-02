"""I3: is the value coordinate carried when an input moves the state from one cluster to another?

    JAX_PLATFORMS=cpu python analysis/i3_tdr_dynamics.py

States at t and t+1 (within session) are projected onto I2's orthonormal TDR axes. For each
input received (choice × reward of trial t) and each source lobe (the state at t had
previous choice R or L, i.e. whether trial t repeated or switched side), regress the value
coordinate at t+1 on that at t: slope ≈ 1 and high R² = the value coordinate is carried
across the jump; slope ≈ 0 = it is reset. Also: the mean jump along each TDR axis, and the
stable fixed points of each constant input (forward iteration of the free-context step at the
all-sessions mean context) projected onto the same axes. Writes ``i3_tdr_dynamics.json`` and
``fig_I3_tdr_dynamics.png``.
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

N_TRANSITIONS = 1_000_000
SEED = 4
INPUTS = {"R rewarded": (1, 1), "R unrewarded": (1, 0), "L rewarded": (0, 1), "L unrewarded": (0, 0)}
N_FORWARD, N_STARTS, DEDUP = 600, 300, 1e-3


def attractors(basis: np.ndarray, mean: np.ndarray, starts: np.ndarray) -> dict:
    """Stable fixed points of each constant input (wrapper ``find_fixed_points``: forward
    iteration + damped Newton, float64) at the all-sessions mean context, in TDR coordinates."""
    import jax  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp  # noqa: PLC0415

    from post_training_analysis.gru_step import GruStep, find_fixed_points  # noqa: PLC0415

    step = GruStep.from_params({k: {n: np.asarray(a, dtype=np.float64) for n, a in v.items()}
                                for k, v in s12.best_params().items() if isinstance(v, dict)})
    c = jnp.asarray(np.load(s12.CACHE / f"contexts_{s12.MODEL_KEY}.npz")["contexts"].mean(axis=0), dtype=jnp.float64)
    out = {}
    for name, (ch, rw) in INPUTS.items():
        u = jnp.asarray([float(ch), float(rw)])
        found, q = find_fixed_points(step, starts.astype(np.float64), u, c, n_forward=N_FORWARD, dedup=DEDUP)
        stable = [f for f in found if np.max(np.abs(np.linalg.eigvals(np.asarray(step.jacobian(jnp.asarray(f), u, c))))) < 1]
        out[name] = {"tdr": [((f - mean) @ basis).tolist() for f in stable],
                     "logit_right": [float(step.logit_diff(jnp.asarray(f))) for f in stable],
                     "n_fixed_points": len(found), "n_stable": len(stable),
                     "frac_starts_converged": float(np.mean(q < 1e-12))}
    return out


def main() -> None:
    s12.apply_presentation_style()
    axes_file = np.load(s12.keyed_cache("i2_tdr_axes.npz"))
    basis, terms = axes_file["basis"], [str(t) for t in axes_file["terms"]]
    pca = s12.load_pca()
    t = s12.load_trials(["ses_idx", "action", "reward"])
    ses = t["ses_idx"].to_numpy()
    action, reward = t["action"].to_numpy(), (t["reward"].to_numpy() > 0).astype(int)
    # Transition t -> t+1 within a session, where the state at t itself had a previous trial.
    ok = np.flatnonzero((ses[1:-1] == ses[2:]) & (ses[1:-1] == ses[:-2])) + 1
    rows = np.sort(np.random.default_rng(SEED).choice(ok, size=min(N_TRANSITIONS, len(ok)), replace=False))
    states = s12.load_trials(s12.STATE_COLUMNS).to_numpy(dtype=np.float32)
    now = (states[rows].astype(np.float64) - pca["mean"]) @ basis
    nxt = (states[rows + 1].astype(np.float64) - pca["mean"]) @ basis
    source_lobe = np.where(action[rows - 1] == 1, "R", "L")  # the previous choice the state at t carries
    v = terms.index("value")

    results, curves = {}, {}
    for name, (ch, rw) in INPUTS.items():
        m_in = (action[rows] == ch) & (reward[rows] == rw)
        for lobe in ("R", "L"):
            m = m_in & (source_lobe == lobe)
            x, y = now[m, v], nxt[m, v]
            slope, intercept = np.polyfit(x, y, 1)
            r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
            kind = "stay" if (lobe == "R") == (ch == 1) else "switch"
            results[f"{name} | from {lobe} lobe ({kind})"] = {
                "n": int(m.sum()), "slope": float(slope), "intercept": float(intercept), "r2": r2,
                "mean_jump": dict(zip(terms, (nxt[m] - now[m]).mean(axis=0).round(4).tolist()))}
            curves[(name, lobe)] = (x, y)
    fixed = attractors(basis, pca["mean"], states[np.random.default_rng(0).choice(len(states), N_STARTS, replace=False)])

    fig, axes = plt.subplots(1, 4, figsize=(24, 6.2), sharex=True, sharey=True)
    lim = (-3.5, 4.5)
    for ax, (name, (ch, _)) in zip(axes, INPUTS.items()):
        for lobe, color in (("R", "#0072B2"), ("L", "#D55E00")):
            x, y = curves[(name, lobe)]
            sub = np.random.default_rng(1).permutation(len(x))[:25_000]
            kind = "stay" if (lobe == "R") == (ch == 1) else "switch"
            res = results[f"{name} | from {lobe} lobe ({kind})"]
            ax.scatter(x[sub], y[sub], s=1.5, alpha=0.25, color=color, linewidths=0, rasterized=True,
                       label=f"from {lobe} lobe ({kind}): slope {res['slope']:.2f}, R² {res['r2']:.2f}, n={res['n']:,}")
        for point in fixed[name]["tdr"]:
            ax.axhline(point[v], color="k", lw=1.2, ls="--")
        ax.plot(lim, lim, color="0.6", lw=1, ls=":")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(f"input: {name}", fontsize=13)
        ax.set_xlabel("value coordinate, trial t")
        ax.legend(frameon=False, fontsize=8.5, markerscale=6, loc="upper left")
    axes[0].set_ylabel("value coordinate, trial t+1")
    fig.suptitle("Is the value coordinate carried through the jump? (dashed: value coordinate of the input's attractor, "
                 "all-sessions mean context; dotted: carried unchanged)", fontsize=12)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_I3_tdr_dynamics.png"))
    plt.close(fig)

    s12.write_json("i3_tdr_dynamics.json", {
        "n_transitions": int(len(rows)), "seed": SEED, "axes": terms, "transitions": results,
        "attractors_all_sessions_context": fixed, "forward_iterations": N_FORWARD, "n_starts": N_STARTS,
    }, "analysis/i3_tdr_dynamics.py")
    for k, r in results.items():
        print(f"{k:42s} n={r['n']:7d} slope={r['slope']:+.2f} R2={r['r2']:.2f} jump={r['mean_jump']}")
    print({k: (np.round([p[v] for p in f['tdr']], 2).tolist(), np.round(f['logit_right'], 2).tolist()) for k, f in fixed.items()})


if __name__ == "__main__":
    main()
