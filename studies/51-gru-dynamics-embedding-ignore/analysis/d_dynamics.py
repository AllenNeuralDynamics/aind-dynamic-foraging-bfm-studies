"""D (stage A): flow fields, fixed points and timescales of C3 — and what the ignore input does to them.

    python analysis/d_dynamics.py

Uses the wrapper's free-context step (``gru_step``): h' = F(h, obs, c), obs = (choice, reward)
of the step just completed (an ignored trial is (2, 0)), c the subject context. Steps:
  1. parity — stepping real sessions with their own contexts reproduces A1's side and engage
     readouts (ignored trials included);
  2. fixed points of each constant input (L+, L−, R+, R−, ignore) at the all-sessions mean
     context, from N_INIT real states: Adam on q = ½‖F(h) − h‖² and forward iteration, each
     refined by damped Newton in float64, accepted at q < FIXED_TOL; Jacobian spectra, time
     constants τ = −1/ln|λ|, and the overlap of the three slowest modes with the side and
     engage directions. Study 50's ``d_dynamics.json`` gives C's four inputs at its own mean
     context for comparison;
  3. flow fields: mean Δh per input in PC1–PC2 over real states;
  4. relaxation: from N_RELAX real engaged states, each with its own session context, feed
     the ignore input for K_RELAX steps and record the side and engage readouts; from the same
     states after K_RELAX ignores, feed unrewarded choices of the previously favoured side and
     record how engagement recovers.
Writes ``d_dynamics.json``, ``fig_D1_flow_fields.png``, ``fig_D2_fixed_points.png``,
``fig_D3_relaxation.png``.
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

N_INIT = 2000
N_STEPS = 4000
LR = 1e-2
N_FORWARD = 500
N_NEWTON = 30
NEWTON_DAMPING = 1e-8
FIXED_TOL = 1e-12
DEDUP = 1e-3
INPUTS = {"R+": (1.0, 1.0), "R−": (1.0, 0.0), "L+": (0.0, 1.0), "L−": (0.0, 0.0), "ignore": (2.0, 0.0)}
C_INPUT_NAMES = {"R+": "R rewarded", "R−": "R unrewarded", "L+": "L rewarded", "L−": "L unrewarded"}
PARITY_SESSIONS = 200
PARITY_TOLERANCE_MEDIAN = 5e-3
N_RELAX = 5000
K_RELAX = 30


def main() -> None:
    import jax  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    import jax.numpy as jnp  # noqa: PLC0415
    import optax  # noqa: PLC0415

    from post_training_analysis.gru_counterfactual import load_gru_split  # noqa: PLC0415
    from post_training_analysis.gru_step import GruStep, split_inputs, subject_context  # noqa: PLC0415

    s13.apply_presentation_style()
    split = load_gru_split(s13.CACHE / "models" / s13.MODEL_KEY / "run")
    step = GruStep.from_params(split.params)
    w = np.asarray(step.readout_w, dtype=np.float64)
    b = np.asarray(step.readout_b, dtype=np.float64)

    def readouts(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = np.asarray(h, dtype=np.float64) @ w + b
        return z[..., 1] - z[..., 0], np.logaddexp(z[..., 0], z[..., 1]) - z[..., 2]

    subject_idx, session_idx, obs = split_inputs(split.xs, session_conditioning=split.run.session_conditioning_enabled)
    trials = s13.load_trials(["subject_id", "ses_idx", "trial_idx", "action", "logit_diff",
                              "choice_logit_0", "choice_logit_1", "choice_logit_2"])
    col = trials["ses_idx"].map({s: i for i, s in enumerate(split.session_ids)}).to_numpy()
    tau = trials["trial_idx"].to_numpy() - 1
    contexts = subject_context(split.make_network, split.params, subject_idx[0], session_idx[0])
    np.savez(s13.CACHE / f"contexts_{s13.MODEL_KEY}.npz", contexts=contexts, session_ids=np.array(split.session_ids))

    # 1. parity on real sessions (ignored steps included).
    cols = np.arange(PARITY_SESSIONS)
    states = np.asarray(step.run(jnp.asarray(obs[:, cols]),
                                 jnp.broadcast_to(contexts[cols], obs[:, cols].shape[:2] + contexts.shape[1:])))
    side, engage = readouts(states)
    sel = col < PARITY_SESSIONS
    parity = {name: np.abs(ours[tau[sel], col[sel]] - trials[ref].to_numpy(float)[sel])
              for name, ours, ref in (("side", side, "logit_diff"), ("engage", engage, "engage_logit"))}
    for name, err in parity.items():
        if np.median(err) > PARITY_TOLERANCE_MEDIAN:
            raise AssertionError(f"Step parity failed for {name}: median {np.median(err)}")

    pca = s13.load_pca()
    mean, components = pca["mean"], pca["components"]
    all_states = s13.load_trials(s13.STATE_COLUMNS).to_numpy(dtype=np.float32)
    directions = {k: v / np.linalg.norm(v) for k, v in s13.readout_directions().items()}
    rng = np.random.default_rng(0)
    c_mean = jnp.asarray(contexts.mean(axis=0))

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
            return h + jnp.linalg.solve(a.T @ a + NEWTON_DAMPING * eye, -a.T @ r)
        return jax.lax.fori_loop(0, N_NEWTON, lambda _, h: jax.vmap(one)(h), h0)

    def fixed_points(init, u, c):
        init = jnp.asarray(init, dtype=jnp.float64)
        found, q_all, source = [], [], []
        for method, h in (("adam", newton(adam_search(init, u, c), u, c)), ("forward", newton(forward(init, u, c), u, c))):
            h, q = np.asarray(h), np.asarray(q_fn(h, u, c))
            q_all.append(q)
            for i in np.argsort(q):
                if not np.isfinite(q[i]) or q[i] > FIXED_TOL:
                    break
                if all(np.linalg.norm(h[i] - f) > DEDUP for f in found):
                    found.append(h[i])
                    source.append(method)
        return found, np.concatenate(q_all), source

    def mode_overlap(vec):
        basis, _ = np.linalg.qr(np.stack([vec.real, vec.imag], axis=1)
                                if np.iscomplexobj(vec) and np.abs(vec.imag).max() > 1e-9 else vec.real[:, None])
        return {k: float(np.linalg.norm(basis.T @ v)) for k, v in directions.items()}

    # 2. fixed points at the mean context.
    init = all_states[rng.choice(len(all_states), size=N_INIT, replace=False)]
    fixed = {}
    for uname, u in INPUTS.items():
        u = jnp.asarray(u)
        found, q, source = fixed_points(init, u, c_mean)
        fps = []
        for f, method in zip(found, source):
            eig, vec = np.linalg.eig(np.asarray(jac_fn(jnp.asarray(f), u, c_mean)))
            order = np.argsort(-np.abs(eig))
            eig, vec = eig[order], vec[:, order]
            lam = np.abs(eig[:5])
            s_logit, e_logit = readouts(f)
            fps.append({"found_by": method, "pc": ((f - mean) @ components[:6].T).round(4),
                        "side_logit": float(s_logit), "engage_logit": float(e_logit),
                        "p_ignore": float(1 / (1 + np.exp(e_logit))),
                        "stable": bool(np.abs(eig[0]) < 1), "top_abs_eig": lam.round(5),
                        "tau_steps": [float(-1 / np.log(v)) if v < 1 else None for v in lam],
                        "slow_mode_overlap": [mode_overlap(vec[:, k]) for k in range(3)]})
        fixed[uname] = {"n_fixed_points": len(fps), "fixed_points": fps,
                        "frac_candidates_converged": float(np.mean(q < FIXED_TOL))}
        print(uname, [(round(fp["side_logit"], 2), round(fp["p_ignore"], 3), fp["stable"],
                       round(fp["tau_steps"][0] or -1, 1)) for fp in fps])

    # 3. flow fields over real states at the mean context.
    sample = all_states[rng.choice(len(all_states), size=20_000, replace=False)].astype(np.float64)
    flows = {uname: np.asarray(step.step(jnp.asarray(sample), jnp.asarray(u), c_mean)) - sample for uname, u in INPUTS.items()}

    # 4. relaxation under repeated ignores, each state with its own session context.
    engaged_rows = np.flatnonzero(trials["action"].to_numpy() != s13.IGNORE)
    rows = rng.choice(engaged_rows, size=N_RELAX, replace=False)
    h0 = all_states[rows].astype(np.float64)
    c_rows = jnp.asarray(contexts[col[rows]])
    s0, e0 = readouts(h0)

    @jax.jit
    def unroll(h, u_seq):
        def body(h, u):
            h = step.step(h, u, c_rows)
            return h, h
        return jax.lax.scan(body, h, u_seq)

    ignore_seq = jnp.broadcast_to(jnp.asarray(INPUTS["ignore"]), (K_RELAX, N_RELAX, 2))
    h_end, path = unroll(jnp.asarray(h0), ignore_seq)
    side_path, engage_path = readouts(np.asarray(path))
    favoured = np.where(s0 >= 0, 1.0, 0.0)  # recovery input: unrewarded choice of the side favoured before
    recover_seq = jnp.broadcast_to(jnp.stack([jnp.asarray(favoured), jnp.zeros(N_RELAX)], axis=-1), (K_RELAX, N_RELAX, 2))
    _, rec_path = unroll(h_end, recover_seq)
    rec_side, rec_engage = readouts(np.asarray(rec_path))
    frac_side = np.abs(side_path) / np.abs(s0)[None, :]
    relax = {
        "k": np.arange(1, K_RELAX + 1),
        "side_abs_fraction_of_start_median": np.median(frac_side, axis=1),
        "side_abs_median": np.median(np.abs(side_path), axis=1),
        "p_ignore_mean": np.mean(1 / (1 + np.exp(engage_path)), axis=1),
        "start": {"side_abs_median": float(np.median(np.abs(s0))), "p_ignore_mean": float(np.mean(1 / (1 + np.exp(e0))))},
        "recovery_p_ignore_mean": np.mean(1 / (1 + np.exp(rec_engage)), axis=1),
        "recovery_side_toward_favoured_median": np.median(np.where(favoured == 1, rec_side, -rec_side), axis=1),
        "n_states": N_RELAX,
    }
    k_half = int(np.argmax(relax["side_abs_fraction_of_start_median"] <= 0.5)) + 1
    relax["k_to_half_side_preference"] = k_half

    # Figures. D1: flow fields of the five inputs.
    xy = (sample - mean) @ components[:2].T
    edges = [np.linspace(xy[:, 0].min(), xy[:, 0].max(), 22), np.linspace(xy[:, 1].min(), xy[:, 1].max(), 19)]
    fig, axes = plt.subplots(1, len(INPUTS), figsize=(5.4 * len(INPUTS), 5.8), sharex=True, sharey=True)
    for ax, uname in zip(axes, INPUTS):
        dxy = flows[uname] @ components[:2].T
        ix, iy = (np.digitize(xy[:, k], edges[k]) - 1 for k in range(2))
        gx, gy, gu, gv = [], [], [], []
        for i in range(len(edges[0]) - 1):
            for j in range(len(edges[1]) - 1):
                m = (ix == i) & (iy == j)
                if m.sum() >= 20:
                    gx.append((edges[0][i] + edges[0][i + 1]) / 2)
                    gy.append((edges[1][j] + edges[1][j + 1]) / 2)
                    gu.append(dxy[m, 0].mean())
                    gv.append(dxy[m, 1].mean())
        ax.scatter(xy[:, 0], xy[:, 1], s=1, color="0.88", linewidths=0, rasterized=True)
        ax.quiver(gx, gy, gu, gv, angles="xy", scale_units="xy", scale=1.0, width=0.004, color="#333333")
        for fp in fixed[uname]["fixed_points"]:
            ax.scatter(fp["pc"][0], fp["pc"][1], s=180, marker="*" if fp["stable"] else "X",
                       color=s13.ENGAGE_COLOR if uname == "ignore" else "#D55E00", edgecolors="k", zorder=5)
        ax.set_title(f"input: {uname}", fontsize=14)
        ax.set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    fig.suptitle("C3 flow fields at the mean context: one step's Δh in PC1–PC2 (full length); "
                 "fixed points of the constant input (★ stable, ✕ unstable)", fontsize=14)
    fig.tight_layout()
    fig.savefig(s13.figure_path("fig_D1_flow_fields.png"))
    plt.close(fig)

    # D2: fixed points — C3 vs C (Study 50), readouts and timescales.
    c_dyn = json.loads((s13.STUDY12 / "analysis" / "d_dynamics.json").read_text())["contexts"]["all sessions"]["inputs"]
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))
    names = list(INPUTS)
    for k, uname in enumerate(names):
        for fp in fixed[uname]["fixed_points"]:
            axes[0].scatter(k - 0.12, fp["side_logit"], s=140, marker="*" if fp["stable"] else "X",
                            color=s13.ENGAGE_COLOR, edgecolors="k", label="C3" if k == 0 else None)
            axes[1].scatter(k, fp["p_ignore"], s=140, marker="*" if fp["stable"] else "X", color=s13.ENGAGE_COLOR, edgecolors="k")
            if fp["stable"]:
                taus = [v for v in fp["tau_steps"][:3] if v is not None]
                axes[2].scatter([k - 0.12] * len(taus), taus, color=s13.ENGAGE_COLOR, s=[70, 35, 18][:len(taus)],
                                label="C3" if (k == 0) else None)
        if uname in C_INPUT_NAMES:
            for fp in c_dyn[C_INPUT_NAMES[uname]]["fixed_points"]:
                axes[0].scatter(k + 0.12, fp["logit_right"], s=140, marker="*" if fp["stable"] else "X",
                                color=s13.MODEL_COLOR, edgecolors="k", label="C (Study 50)" if k == 0 else None)
                if fp["stable"]:
                    taus = [v for v in fp["tau_trials"][:3] if v is not None]
                    axes[2].scatter([k + 0.12] * len(taus), taus, color=s13.MODEL_COLOR, s=[70, 35, 18][:len(taus)],
                                    label="C (Study 50)" if k == 0 else None)
    axes[0].axhline(0, color="0.8", lw=0.8)
    axes[0].set(xticks=range(len(names)), xticklabels=names, ylabel="side logit at the fixed point",
                title="a. where each constant input leads (side)")
    axes[1].set(xticks=range(len(names)), xticklabels=names, ylabel="P(ignore) at the fixed point", ylim=(0, 1),
                title="b. engagement at each fixed point (C3)")
    axes[2].set(xticks=range(len(names)), xticklabels=names, ylabel="τ (steps), three slowest modes", yscale="log",
                title="c. timescales at the stable fixed points")
    for ax in (axes[0], axes[2]):
        ax.legend(frameon=False)
    fig.suptitle("Fixed points of constant inputs at each model's mean context (★ stable, ✕ unstable)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(s13.figure_path("fig_D2_fixed_points.png"))
    plt.close(fig)

    # D3: relaxation under repeated ignores, then recovery.
    k = relax["k"]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    axes[0].plot(k, relax["side_abs_fraction_of_start_median"], color=s13.ENGAGE_COLOR, marker="o", lw=2)
    axes[0].axhline(0.5, color="0.8", lw=0.8)
    axes[0].set(xlabel="consecutive ignore inputs", ylabel="|side logit| / its starting value (median)", ylim=(0, 1.05),
                title=f"a. side preference under repeated ignores (half after {k_half})")
    axes[1].plot(np.r_[0, k], np.r_[relax["start"]["p_ignore_mean"], relax["p_ignore_mean"]], color=s13.ENGAGE_COLOR,
                 marker="o", lw=2)
    axes[1].set(xlabel="consecutive ignore inputs", ylabel="mean P(ignore)", ylim=(0, 1),
                title="b. engagement under repeated ignores")
    axes[2].plot(np.r_[0, k], np.r_[relax["p_ignore_mean"][-1], relax["recovery_p_ignore_mean"]], color=s13.IGNORE_COLOR,
                 marker="o", lw=2, label="P(ignore)")
    axes[2].plot(np.r_[0, k], np.r_[np.nan, relax["recovery_side_toward_favoured_median"]], color=s13.MODEL_COLOR,
                 marker="s", lw=1.5, label="side logit toward the inserted side (median)")
    axes[2].set(xlabel=f"unrewarded choices after {K_RELAX} ignores", title="c. recovery: unrewarded choices of the pre-bout side")
    axes[2].legend(frameon=False)
    fig.suptitle(f"C3 relaxation from {N_RELAX:,} real engaged states, each with its own session context", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(s13.figure_path("fig_D3_relaxation.png"))
    plt.close(fig)

    s13.write_json("d_dynamics.json", {
        "parity": {name: {"median_abs_err": float(np.median(e)), "max_abs_err": float(np.max(e))} for name, e in parity.items()},
        "parity_sessions": PARITY_SESSIONS,
        "context": "mean over all eval sessions (fixed points, flow); own session context (relaxation)",
        "fixed_points": fixed,
        "relaxation": relax,
        "C_fixed_points_from": "../50-gru-dynamics-embedding/analysis/d_dynamics.json (all sessions)",
        "n_init": N_INIT, "n_steps": N_STEPS, "lr": LR, "n_forward": N_FORWARD, "n_newton": N_NEWTON,
        "newton_damping": NEWTON_DAMPING, "fixed_tol": FIXED_TOL, "dedup": DEDUP, "precision": "float64, highest",
        "inputs": INPUTS,
    }, "analysis/d_dynamics.py")
    print("parity", {n: float(np.median(e)) for n, e in parity.items()}, "half-life", k_half)


if __name__ == "__main__":
    main()
