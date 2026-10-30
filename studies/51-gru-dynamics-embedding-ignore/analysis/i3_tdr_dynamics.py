"""I3: is C3's TDR value coordinate carried through every input — including an ignore?

    python analysis/i3_tdr_dynamics.py

Study 50's I3 on C3. States at t and t+1 (within session, state at t has a previous step)
are projected onto I2's orthonormal TDR axes. For each input received at t (L+, L−, R+, R−,
ignore) and each source — the previous step of the state at t (R lobe, L lobe, or an
ignore, i.e. already in the post-ignore cluster) — regress the value coordinate at t+1 on
that at t: slope and R². Study 50's criterion for "carried": a tight linear map (C: slopes
0.66–0.80, R² 0.93–0.98 for all eight input × lobe cells). Also the mean jump along each
axis, and each constant input's stable fixed point (wrapper ``find_fixed_points``, float64,
all-sessions mean context) on the same axes.
Writes ``i3_tdr_dynamics.json`` and ``fig_I3_tdr_dynamics.png``.
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
import _pk02 as s13  # noqa: E402

N_TRANSITIONS = 1_000_000
SEED = 4
INPUTS = {"R+": (1.0, 1.0), "R−": (1.0, 0.0), "L+": (0.0, 1.0), "L−": (0.0, 0.0), "ignore": (2.0, 0.0)}
SOURCES = {"R": "R lobe", "L": "L lobe", "ignore": "post-ignore"}
N_FORWARD, N_STARTS, DEDUP = 600, 300, 1e-3


def attractors(basis: np.ndarray, mean: np.ndarray, starts: np.ndarray) -> dict:
    import jax  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    import jax.numpy as jnp  # noqa: PLC0415

    from post_training_analysis.gru_step import GruStep, find_fixed_points  # noqa: PLC0415

    step = GruStep.from_params({k: {n: np.asarray(a, dtype=np.float64) for n, a in v.items()}
                                for k, v in s13.best_params().items() if isinstance(v, dict)})
    c = jnp.asarray(np.load(s13.CACHE / f"contexts_{s13.MODEL_KEY}.npz")["contexts"].mean(axis=0), dtype=jnp.float64)
    out = {}
    for name, u in INPUTS.items():
        u = jnp.asarray(u)
        found, q = find_fixed_points(step, starts.astype(np.float64), u, c, n_forward=N_FORWARD, dedup=DEDUP)
        stable = [f for f in found if np.max(np.abs(np.linalg.eigvals(np.asarray(step.jacobian(jnp.asarray(f), u, c))))) < 1]
        out[name] = {"tdr": [((f - mean) @ basis).tolist() for f in stable], "n_fixed_points": len(found),
                     "n_stable": len(stable), "frac_starts_converged": float(np.mean(q < 1e-12))}
    return out


def main() -> None:
    s13.apply_presentation_style()
    axes_file = np.load(s13.CACHE / f"i2_tdr_axes_{s13.MODEL_KEY}.npz")
    basis, terms = axes_file["basis"], [str(t) for t in axes_file["terms"]]
    pca = s13.load_pca()
    t = s13.load_trials(["ses_idx", "action", "reward", "logit_diff"])
    ses = t["ses_idx"].to_numpy()
    action, reward = t["action"].to_numpy(), (t["reward"].to_numpy() > 0).astype(int)
    ok = np.flatnonzero((ses[1:-1] == ses[2:]) & (ses[1:-1] == ses[:-2])) + 1
    rows = np.sort(np.random.default_rng(SEED).choice(ok, size=min(N_TRANSITIONS, len(ok)), replace=False))
    states = s13.load_trials(s13.STATE_COLUMNS).to_numpy(dtype=np.float32)
    now = (states[rows].astype(np.float64) - pca["mean"]) @ basis
    nxt = (states[rows + 1].astype(np.float64) - pca["mean"]) @ basis
    prev = action[rows - 1]
    source = np.where(prev == s13.IGNORE, "ignore", np.where(prev == 1, "R", "L"))
    label = np.where(action[rows] == s13.IGNORE, "ignore",
                     np.where(action[rows] == 1, "R", "L").astype(object) + np.where(reward[rows] > 0, "+", "−"))
    v = terms.index("value")

    results, curves = {}, {}
    for name in INPUTS:
        for src in SOURCES:
            m = (label == name) & (source == src)
            if m.sum() < 500:
                continue
            x, y = now[m, v], nxt[m, v]
            slope, intercept = np.polyfit(x, y, 1)
            if name == "ignore":
                kind = "ignore"
            elif src == "ignore":
                kind = "after an ignore"
            else:
                kind = "stay" if (src == "R") == name.startswith("R") else "switch"
            results[f"{name} | from {SOURCES[src]} ({kind})"] = {
                "n": int(m.sum()), "slope": float(slope), "intercept": float(intercept),
                "r2": float(np.corrcoef(x, y)[0, 1] ** 2),
                "mean_jump": dict(zip(terms, (nxt[m] - now[m]).mean(axis=0).round(4).tolist()))}
            curves[(name, src)] = (x, y)
    # How the side readout maps onto the value coordinate in each source cluster (Study 50:
    # logit ≈ a·v + b[cluster]); the value of the readout-neutral point is −b/a.
    logit_now = t["logit_diff"].to_numpy(np.float64)[rows]
    cluster_label = np.where(source == "ignore", "ignore", source.astype(object)
                             + np.where(reward[rows - 1] > 0, "+", "−"))
    readout_map = {}
    for cl in ("R+", "R−", "L+", "L−", "ignore"):
        m = cluster_label == cl
        a, b = np.polyfit(now[m, v], logit_now[m], 1)
        readout_map[cl] = {"a": float(a), "b": float(b), "value_at_neutral_readout": float(-b / a),
                           "r2": float(np.corrcoef(now[m, v], logit_now[m])[0, 1] ** 2),
                           "value_mean": float(now[m, v].mean()), "n": int(m.sum())}
    fixed = attractors(basis, pca["mean"], states[np.random.default_rng(0).choice(len(states), N_STARTS, replace=False)])
    c12 = json.loads((s13.STUDY12 / "analysis" / "i3_tdr_dynamics.json").read_text())["transitions"]

    fig, axes = plt.subplots(1, len(INPUTS), figsize=(5.4 * len(INPUTS), 6.2), sharex=True, sharey=True)
    lim = (-4.5, 4.5)
    colors = {"R": "#0072B2", "L": "#D55E00", "ignore": s13.IGNORE_COLOR}
    for ax, name in zip(axes, INPUTS):
        for src, color in colors.items():
            if (name, src) not in curves:
                continue
            x, y = curves[(name, src)]
            key = next(k for k in results if k.startswith(f"{name} | from {SOURCES[src]}"))
            res = results[key]
            sub = np.random.default_rng(1).permutation(len(x))[:25_000]
            ax.scatter(x[sub], y[sub], s=1.5, alpha=0.25, color=color, linewidths=0, rasterized=True,
                       label=f"from {SOURCES[src]}: slope {res['slope']:.2f}, R² {res['r2']:.2f}, n={res['n']:,}")
        for point in fixed[name]["tdr"]:
            ax.axhline(point[v], color="k", lw=1.2, ls="--")
        ax.plot(lim, lim, color="0.6", lw=1, ls=":")
        ax.set(xlim=lim, ylim=lim, title=f"input: {name}", xlabel="value coordinate, step t")
        ax.legend(frameon=False, fontsize=8.5, markerscale=6, loc="upper left")
    axes[0].set_ylabel("value coordinate, step t+1")
    fig.suptitle("C3: is the TDR value coordinate carried through each input? (dashed: the input's attractor; "
                 "dotted: carried unchanged)", fontsize=13)
    fig.tight_layout()
    fig.savefig(s13.figure_path("fig_I3_tdr_dynamics.png"))
    plt.close(fig)

    s13.write_json("i3_tdr_dynamics.json", {
        "n_transitions": int(len(rows)), "seed": SEED, "axes": terms, "transitions": results,
        "attractors_all_sessions_context": fixed, "forward_iterations": N_FORWARD, "n_starts": N_STARTS,
        "readout_vs_value_by_cluster": readout_map,
        "C_study12_transitions": {k: {"slope": r["slope"], "r2": r["r2"], "n": r["n"]} for k, r in c12.items()},
    }, "analysis/i3_tdr_dynamics.py", extra_groups=s13.twin_wandb_groups())
    for k, r in results.items():
        print(f"{k:46s} n={r['n']:8,} slope={r['slope']:+.3f} R2={r['r2']:.3f} intercept={r['intercept']:+.2f}")
    print({k: [round(p[v], 2) for p in f["tdr"]] for k, f in fixed.items()})
    print({k: {a: round(b, 3) for a, b in r.items()} for k, r in readout_map.items()})


if __name__ == "__main__":
    main()
