"""D (stage A): flow fields, fixed points and linearised timescales of model C, per context.

    JAX_PLATFORMS=cpu python analysis/d_dynamics.py

Uses the wrapper's free-context step (``gru_step``): h' = F(h, obs, c) with obs = (choice,
reward) of the trial just completed and c the subject context. Steps:
  1. parity — stepping real sessions with the model's own contexts must reproduce A1's logits;
  2. the context of every eval session (cached: ``_cache/contexts_<key>.npz``);
  3. for five contexts (mean over all sessions; F1's low/high-gain and low/high-bias groups)
     and each constant input, fixed points from N_INIT real states: candidates from (a) Adam on
     q = ½‖F(h) − h‖² with a decaying learning rate and (b) forward iteration of F (attractors),
     each refined by damped Newton in float64 and accepted at q < FIXED_TOL; then their
     Jacobian spectra and time constants τ = −1/ln|λ|;
  4. flow fields: mean Δh in the PC1–PC2 plane over each group's own real states;
  5. slow modes vs the readout and PCs 1/2/5 (E1: choice/ΔQ, reward/ΣQ, trial position).
Writes ``d_dynamics.json``, ``fig_D1_flow_fields.png``, ``fig_D2_fixed_points.png``.
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

N_INIT = 2000
N_STEPS = 4000
LR = 1e-2               # Adam, decayed ×1e-3 over N_STEPS (a fixed rate stalls at q ~ 1e-5)
N_FORWARD = 500         # forward iterations for attractor candidates
N_NEWTON = 30
NEWTON_DAMPING = 1e-8
FIXED_TOL = 1e-12       # q = ½‖F(h) − h‖² (float64) below this counts as a fixed point
DEDUP = 1e-3            # fixed points closer than this are the same
INPUTS = {"R rewarded": (1.0, 1.0), "R unrewarded": (1.0, 0.0), "L rewarded": (0.0, 1.0), "L unrewarded": (0.0, 0.0)}
PARITY_SESSIONS = 200
PARITY_TOLERANCE_MEDIAN = 5e-3


def main() -> None:
    import jax  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp  # noqa: PLC0415
    import optax  # noqa: PLC0415

    from post_training_analysis.gru_counterfactual import load_gru_split  # noqa: PLC0415
    from post_training_analysis.gru_step import GruStep, split_inputs, subject_context  # noqa: PLC0415

    s12.apply_presentation_style()
    split = load_gru_split(s12.CACHE / "models" / s12.MODEL_KEY / "run")
    step = GruStep.from_params(split.params)
    subject_idx, session_idx, obs = split_inputs(split.xs, session_conditioning=split.run.session_conditioning_enabled)
    trials = s12.load_trials(["subject_id", "ses_idx", "trial_idx", "logit_diff", "p_model1_right"])
    col = trials["ses_idx"].map({s: i for i, s in enumerate(split.session_ids)}).to_numpy()
    tau = trials["trial_idx"].to_numpy() - 1

    # 2. one context per session (session index is constant within a session).
    contexts = subject_context(split.make_network, split.params, subject_idx[0], session_idx[0])
    np.savez(s12.CACHE / f"contexts_{s12.MODEL_KEY}.npz", contexts=contexts, session_ids=np.array(split.session_ids),
             subject_index=subject_idx[0], session_index=session_idx[0])

    # 1. parity on real sessions.
    cols = np.arange(PARITY_SESSIONS)
    states = np.asarray(step.run(jnp.asarray(obs[:, cols]), jnp.broadcast_to(contexts[cols], obs[:, cols].shape[:2] + contexts.shape[1:])))
    ours = np.asarray(step.logit_diff(states))
    sel = col < PARITY_SESSIONS
    parity = np.abs(ours[tau[sel], col[sel]] - trials["logit_diff"].to_numpy()[sel])
    if np.median(parity) > PARITY_TOLERANCE_MEDIAN:
        raise AssertionError(f"Step parity failed: median |Δlogit| {np.median(parity)}")

    # Context groups.
    f1 = json.loads((s12.ANALYSIS / "f1_embedding_extremes.json").read_text())
    session_subject = trials.drop_duplicates("ses_idx").set_index("ses_idx")["subject_id"].reindex(split.session_ids).to_numpy()
    groups = {"all sessions": np.ones(len(split.session_ids), bool)}
    for axis in ("gain", "bias"):
        for level in ("low", "high"):
            groups[f"{level} {axis}"] = np.isin(session_subject, f1["groups"][axis][level])
    pca = s12.load_pca()
    mean, components = pca["mean"], pca["components"]
    all_states = s12.load_trials(s12.STATE_COLUMNS).to_numpy(dtype=np.float32)
    readout = np.array(step.readout_w[:, 1] - step.readout_w[:, 0], dtype=np.float64)
    readout = readout / np.linalg.norm(readout)
    targets = {"readout": readout, "PC1 (choice/ΔQ)": components[0], "PC2 (reward/ΣQ)": components[1], "PC5 (time)": components[4]}
    rng = np.random.default_rng(0)

    def q_single(h, u, c):
        return 0.5 * jnp.sum((step.step(h, u, c) - h) ** 2)

    q_fn = jax.jit(jax.vmap(q_single, in_axes=(0, None, None)))
    grad_fn = jax.vmap(jax.grad(q_single), in_axes=(0, None, None))
    jac_fn = jax.jit(step.jacobian)
    opt = optax.adam(optax.exponential_decay(LR, N_STEPS, 1e-3))
    eye = jnp.eye(step.hidden_size)

    @jax.jit
    def adam_search(h0, u, c):
        def body(_, carry):
            h, state = carry
            updates, state = opt.update(grad_fn(h, u, c), state, h)
            return optax.apply_updates(h, updates), state
        return jax.lax.fori_loop(0, N_STEPS, body, (h0, opt.init(h0)))[0]

    @jax.jit
    def forward(h0, u, c):
        return jax.lax.fori_loop(0, N_FORWARD, lambda _, h: step.step(h, u, c), h0)

    @jax.jit
    def newton(h0, u, c):
        def one(h):
            a = step.jacobian(h, u, c) - eye
            r = step.step(h, u, c) - h
            delta = jnp.linalg.solve(a.T @ a + NEWTON_DAMPING * eye, -a.T @ r)
            return h + delta
        return jax.lax.fori_loop(0, N_NEWTON, lambda _, h: jax.vmap(one)(h), h0)

    def fixed_points(init, u, c):
        init = jnp.asarray(init, dtype=jnp.float64)
        candidates = {"adam": newton(adam_search(init, u, c), u, c), "forward": newton(forward(init, u, c), u, c)}
        found, q_all, source = [], [], []
        for method, h in candidates.items():
            h, q = np.asarray(h), np.asarray(q_fn(h, u, c))
            q_all.append(q)
            for i in np.argsort(q):
                if not np.isfinite(q[i]) or q[i] > FIXED_TOL:
                    break
                if all(np.linalg.norm(h[i] - f) > DEDUP for f in found):
                    found.append(h[i])
                    source.append(method)
        return found, np.concatenate(q_all), source

    def mode_overlap(vecs):
        """Max |projection| of each target axis onto the real span of an eigenvector."""
        basis, _ = np.linalg.qr(np.stack([vecs.real, vecs.imag], axis=1) if np.iscomplexobj(vecs) and np.abs(vecs.imag).max() > 1e-9
                                else vecs.real[:, None])
        return {k: float(np.linalg.norm(basis.T @ v)) for k, v in targets.items()}

    results = {"parity": {"median_abs_err": float(np.median(parity)), "max_abs_err": float(np.max(parity)),
                          "n_sessions": PARITY_SESSIONS}, "contexts": {}}
    flows = {}
    for gname, gmask in groups.items():
        c = jnp.asarray(contexts[gmask].mean(axis=0))
        rows = np.flatnonzero(np.isin(col, np.flatnonzero(gmask)))
        init = all_states[rng.choice(rows, size=N_INIT, replace=False)]
        entry = {"n_sessions": int(gmask.sum()), "context": np.asarray(c), "inputs": {}}
        for uname, u in INPUTS.items():
            u = jnp.asarray(u)
            found, q, source = fixed_points(init, u, c)
            fps = []
            for f, method in zip(found, source):
                eig, vec = np.linalg.eig(np.asarray(jac_fn(jnp.asarray(f), u, c)))
                order = np.argsort(-np.abs(eig))
                eig, vec = eig[order], vec[:, order]
                lam = np.abs(eig[:5])
                fps.append({"found_by": method, "pc": ((f - mean) @ components[:6].T).round(4),
                            "logit_right": float(step.logit_diff(jnp.asarray(f))),
                            "stable": bool(np.abs(eig[0]) < 1), "top_abs_eig": lam.round(5),
                            "tau_trials": [float(-1 / np.log(v)) if v < 1 else None for v in lam],
                            "slow_mode_overlap": [mode_overlap(vec[:, k]) for k in range(3)]})
            entry["inputs"][uname] = {"n_fixed_points": len(fps), "fixed_points": fps,
                                      "frac_candidates_converged": float(np.mean(q < FIXED_TOL)),
                                      "q_quantiles": np.quantile(q, [0.1, 0.5, 0.9]).tolist()}
            print(gname, uname, "fixed points:", len(fps),
                  [(round(fp["logit_right"], 2), fp["stable"], fp["top_abs_eig"][:2].tolist()) for fp in fps][:4])
        # 4. flow fields over the group's own states.
        sample = all_states[rng.choice(rows, size=min(20_000, len(rows)), replace=False)].astype(np.float64)
        flows[gname] = {uname: (np.asarray(step.step(jnp.asarray(sample), jnp.asarray(u), c)) - sample) for uname, u in INPUTS.items()}
        flows[gname]["_states"] = sample
        results["contexts"][gname] = entry

    # Figures: D1 flow fields (all-sessions context) and D2 fixed points across contexts.
    edges = [np.linspace(-4, 4, 21), np.linspace(-4, 3, 18)]
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True, sharey=True)
    base = flows["all sessions"]
    xy = (base["_states"] - mean) @ components[:2].T
    for ax, uname in zip(axes, INPUTS):
        dxy = base[uname] @ components[:2].T
        ix, iy = (np.digitize(xy[:, k], edges[k]) - 1 for k in range(2))
        grid_u, grid_v, gx, gy = [], [], [], []
        for i in range(len(edges[0]) - 1):
            for j in range(len(edges[1]) - 1):
                m = (ix == i) & (iy == j)
                if m.sum() >= 20:
                    gx.append((edges[0][i] + edges[0][i + 1]) / 2)
                    gy.append((edges[1][j] + edges[1][j + 1]) / 2)
                    grid_u.append(dxy[m, 0].mean())
                    grid_v.append(dxy[m, 1].mean())
        ax.scatter(xy[:, 0], xy[:, 1], s=1, color="0.88", linewidths=0, rasterized=True)
        ax.quiver(gx, gy, grid_u, grid_v, angles="xy", scale_units="xy", scale=1.0, width=0.004, color="#333333")
        for fp in results["contexts"]["all sessions"]["inputs"][uname]["fixed_points"]:
            ax.scatter(fp["pc"][0], fp["pc"][1], s=160, marker="*" if fp["stable"] else "X",
                       color="#D55E00" if fp["stable"] else "#0072B2", edgecolors="k", zorder=5)
        ax.set_title(f"input: {uname}", fontsize=13)
        ax.set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    fig.suptitle("Flow fields, all-sessions mean context: one step's Δh in PC1–PC2 (arrows, full length), "
                 "fixed points of the constant input (★ stable, ✕ unstable)", fontsize=13)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_D1_flow_fields.png"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
    colors = {"all sessions": "#222222", "low gain": "#56B4E9", "high gain": "#D55E00", "low bias": "#009E73", "high bias": "#CC79A7"}
    for gname, entry in results["contexts"].items():
        for k, (uname, rec) in enumerate(entry["inputs"].items()):
            for fp in rec["fixed_points"]:
                axes[0].scatter(fp["pc"][0], fp["pc"][1], s=90, marker="*" if fp["stable"] else "X", color=colors[gname],
                                edgecolors="k", label=gname if (k == 0 and fp is rec["fixed_points"][0]) else None)
                if fp["stable"]:
                    taus = [v for v in fp["tau_trials"][:3] if v is not None]
                    axes[1].scatter([k + 0.12 * list(colors).index(gname) - 0.24] * len(taus), taus,
                                    color=colors[gname], s=[60, 30, 15][:len(taus)])
    axes[0].scatter(xy[:, 0], xy[:, 1], s=1, color="0.9", linewidths=0, rasterized=True, zorder=0)
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(frameon=False, fontsize=10)
    axes[0].set_title("Fixed points of the four constant inputs, per context", fontsize=13)
    axes[1].set_xticks(range(4), list(INPUTS), fontsize=11)
    axes[1].set_ylabel("time constant τ (trials), three slowest modes")
    axes[1].set_yscale("log")
    axes[1].set_title("Timescales at the stable fixed points", fontsize=13)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_D2_fixed_points.png"))
    plt.close(fig)

    results.update({"n_init": N_INIT, "n_steps": N_STEPS, "lr": LR, "n_forward": N_FORWARD, "n_newton": N_NEWTON,
                    "newton_damping": NEWTON_DAMPING, "fixed_tol": FIXED_TOL, "dedup": DEDUP, "precision": "float64",
                    "inputs": INPUTS, "targets": list(targets)})
    s12.write_json("d_dynamics.json", results, "analysis/d_dynamics.py")
    print("parity", results["parity"])


if __name__ == "__main__":
    main()
