"""D1e / D1f: flow fields with each mouse's own context instead of the all-sessions mean.

    PYTHONPATH=<wrapper>/code python analysis/d1e_flow_own_context.py     # GPU (JAX)

D1–D1d applied the all-sessions mean context to every state. Here:
  D1e (data-faithful)  20,000 real states, each stepped under each constant input with its *own*
                       session's context; one-step Δh binned in PC1–PC2, arrows log-length (as D1d
                       bottom). Fixed points: every mouse's attractor per input — the end point of
                       repeating that input for N_ITER steps from N_PER_MOUSE of the mouse's own
                       states at its mean context (forward iteration converges onto F2's dominant
                       stable point; checked against F2's logits) — drawn as a cloud.
  D1f (prototypes)     F9's mice at the two ends of the three named strategy axes; each mouse's
                       own states and contexts, with its own attractors (★) and the mean-context
                       attractors (grey ☆) for reference. Rows = mice, columns = inputs.
Writes ``fig_D1e_flow_own_context.png``, ``fig_D1f_flow_prototypes.png`` and
``d1e_flow_own_context.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402
from d1c_flow_two_scales import INPUTS, binned  # noqa: E402
from d1d_flow_scaled import LOG_MAX, LOG_MIN  # noqa: E402

N_SAMPLE = 20_000
N_PER_MOUSE, N_ITER = 8, 800
N_PROTO_STATES = 4_000
SEED = 19
AXES = ["determinism / stickiness", "side bias", "slow learning / long choice memory"]
COLORS = {"R rewarded": "#0072B2", "R unrewarded": "#56B4E9", "L rewarded": "#D55E00", "L unrewarded": "#E69F00"}


def draw_flow(ax, xy, dxy, speed, ex, ey, cloud):
    gx, gy, gu, gv, gs = binned(xy, dxy, speed, ex, ey)
    cell = min(ex[1] - ex[0], ey[1] - ey[0])
    nrm = np.hypot(gu, gv) + 1e-12
    length = 0.9 * cell * np.clip((np.log10(gs) - LOG_MIN) / (LOG_MAX - LOG_MIN), 0.03, 1)
    ax.scatter(cloud[:, 0], cloud[:, 1], s=1, color="0.88", linewidths=0, rasterized=True)
    ax.quiver(gx, gy, length * gu / nrm, length * gv / nrm, angles="xy", scale_units="xy", scale=1,
              width=0.004, headwidth=4, headlength=4, headaxislength=3.5, color="#222222")


def main() -> None:
    import jax  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp  # noqa: PLC0415

    from post_training_analysis.gru_step import GruStep  # noqa: PLC0415

    s12.apply_presentation_style()
    step = GruStep.from_params({k: {n: np.asarray(a, np.float64) for n, a in v.items()}
                                for k, v in s12.best_params().items() if isinstance(v, dict)})
    pca = s12.load_pca()
    mean, comp = pca["mean"], pca["components"][:2]
    proj = lambda h: (np.asarray(h) - mean) @ comp.T  # noqa: E731
    t = s12.load_trials(["subject_id", "ses_idx"])
    ctx = np.load(s12.CACHE / f"contexts_{s12.MODEL_KEY}.npz")
    ctx_of = pd.Series(np.arange(len(ctx["session_ids"])), index=ctx["session_ids"])
    contexts = ctx["contexts"].astype(np.float64)
    row_ctx = contexts[ctx_of.reindex(t["ses_idx"].to_numpy()).to_numpy()]
    subj = t["subject_id"].to_numpy()
    mouse_ctx = pd.DataFrame(row_ctx).groupby(subj).mean()
    states = s12.load_trials(s12.STATE_COLUMNS).to_numpy(np.float32)
    rng = np.random.default_rng(SEED)
    cloud = proj(states[rng.choice(len(states), 60_000, replace=False)])
    one = jax.jit(lambda h, u, c: step.step(h, u, c))
    iterate = jax.jit(lambda h, u, c: jax.lax.fori_loop(0, N_ITER, lambda _, x: step.step(x, u, c), h))

    # Every mouse's attractor per input (forward iteration from its own states, its mean context).
    by_mouse = pd.Series(np.arange(len(t))).groupby(subj).indices
    mice = list(mouse_ctx.index)
    starts = np.concatenate([states[rng.choice(by_mouse[m], size=N_PER_MOUSE, replace=False)] for m in mice]).astype(np.float64)
    cstart = np.repeat(mouse_ctx.loc[mice].to_numpy(), N_PER_MOUSE, axis=0)
    att, att_logit = {}, {}
    for name, u in INPUTS.items():
        uu = jnp.broadcast_to(jnp.asarray(u), (len(starts), 2))
        end = np.asarray(iterate(jnp.asarray(starts), uu, jnp.asarray(cstart)))
        resid = np.linalg.norm(np.asarray(one(jnp.asarray(end), uu, jnp.asarray(cstart))) - end, axis=1)
        lg = np.asarray(step.logit_diff(jnp.asarray(end))).reshape(len(mice), N_PER_MOUSE)
        p = proj(end).reshape(len(mice), N_PER_MOUSE, 2)
        att[name] = pd.DataFrame({"pc1": np.median(p[:, :, 0], 1), "pc2": np.median(p[:, :, 1], 1)}, index=mice)
        att_logit[name] = pd.Series(np.median(lg, 1), index=mice)
        print(name, "median one-step residual at end points", float(np.median(resid)), flush=True)
    f2 = pd.read_parquet(s12.CACHE / f"f2_fixed_points_{s12.MODEL_KEY}.parquet").dropna(subset=["logit_right"])
    f2 = f2.query("context != 'all-sessions mean'").sort_values("n_hits", ascending=False).drop_duplicates(["context", "input"])
    f2_logit = f2.set_index(["context", "input"])["logit_right"]
    check = {}
    for name in INPUTS:
        key = {"R rewarded": "R+", "R unrewarded": "R−", "L rewarded": "L+", "L unrewarded": "L−"}[name]
        common = [m for m in mice if (m, key) in f2_logit.index]
        diff = np.abs(att_logit[name].loc[common].to_numpy() - f2_logit.loc[[(m, key) for m in common]].to_numpy())
        check[name] = {"median_abs_logit_diff_vs_F2": float(np.median(diff)), "share_within_0.05": float(np.mean(diff < 0.05))}
    print("check vs F2", check, flush=True)

    # D1e: each sampled state with its own context.
    sidx = rng.choice(len(states), N_SAMPLE, replace=False)
    sample, csample = jnp.asarray(states[sidx], jnp.float64), jnp.asarray(row_ctx[sidx])
    ex, ey = np.linspace(-4, 4, 25), np.linspace(-4.5, 3, 23)
    fig, axes = plt.subplots(1, 4, figsize=(28, 7.2), sharex=True, sharey=True)
    for ax, (name, u) in zip(axes, INPUTS.items()):
        dh = np.asarray(one(sample, jnp.broadcast_to(jnp.asarray(u), (N_SAMPLE, 2)), csample) - sample)
        draw_flow(ax, proj(sample), dh @ comp.T, np.linalg.norm(dh, axis=1), ex, ey, cloud)
        a = att[name]
        ax.scatter(a["pc1"], a["pc2"], s=9, color=COLORS[name], edgecolors="k", linewidths=0.2, zorder=5,
                   label=f"each mouse's attractor ({len(a)})")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4.6, 3)
        ax.set_title(f"{name}: own contexts", fontsize=13)
        ax.set_xlabel("PC1")
        ax.legend(frameon=False, fontsize=10, loc="lower left" if name.startswith("R") else "lower right")
    axes[0].set_ylabel("PC2")
    fig.suptitle("One-step flow with each state's own session context (arrow length ∝ log10 step size); "
                 "dots: every mouse's attractor at its own mean context", fontsize=14)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_D1e_flow_own_context.png"))
    plt.close(fig)

    # D1f: prototype mice.
    proto = json.loads((s12.ANALYSIS / "f9_strategy_cards.json").read_text())["prototypes"]
    rows = [(ax_name, end, proto[f"{ax_name} | {end}"]) for ax_name in AXES for end in ("low", "high")]
    mean_ctx = jnp.asarray(contexts.mean(0))
    ref = {}
    ref_starts = jnp.asarray(states[rng.choice(len(states), 64, replace=False)], jnp.float64)
    for name, u in INPUTS.items():
        uu = jnp.broadcast_to(jnp.asarray(u), (64, 2))
        e_ = np.asarray(iterate(ref_starts, uu, jnp.broadcast_to(mean_ctx, (64, contexts.shape[1]))))
        ref[name] = np.median(proj(e_), 0)
    fig, axs = plt.subplots(len(rows), 4, figsize=(28, 6.2 * len(rows)), sharex=True, sharey=True)
    for r, (ax_name, end, m) in enumerate(rows):
        own = by_mouse[m]
        pick = rng.choice(own, size=min(N_PROTO_STATES, len(own)), replace=False)
        h, c = jnp.asarray(states[pick], jnp.float64), jnp.asarray(row_ctx[pick])
        for k, (name, u) in enumerate(INPUTS.items()):
            ax = axs[r, k]
            dh = np.asarray(one(h, jnp.broadcast_to(jnp.asarray(u), (len(pick), 2)), c) - h)
            draw_flow(ax, proj(h), dh @ comp.T, np.linalg.norm(dh, axis=1), ex, ey, cloud)
            ax.scatter(*ref[name], s=220, marker="*", facecolor="none", edgecolor="0.4", lw=1.5, zorder=4)
            ax.scatter(att[name].at[m, "pc1"], att[name].at[m, "pc2"], s=260, marker="*", color=COLORS[name], edgecolor="k", zorder=5)
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4.6, 3)
            if r == 0:
                ax.set_title(f"input: {name}", fontsize=14)
            if k == 0:
                ax.set_ylabel(f"{ax_name}: {end}\nmouse {m}\nPC2", fontsize=12)
            if r == len(rows) - 1:
                ax.set_xlabel("PC1")
    fig.suptitle("Flow fields of prototype mice (each with its own states and contexts; arrow length ∝ log10 step size).\n"
                 "★ the mouse's attractor; ☆ the mean-context attractor", fontsize=15, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(s12.figure_path("fig_D1f_flow_prototypes.png"))
    plt.close(fig)

    spread = {name: {"pc1_10_90": np.percentile(a["pc1"], [10, 90]).round(3).tolist(),
                     "pc2_10_90": np.percentile(a["pc2"], [10, 90]).round(3).tolist(),
                     "logit_10_50_90": np.percentile(att_logit[name], [10, 50, 90]).round(3).tolist()} for name, a in att.items()}
    s12.write_json("d1e_flow_own_context.json", {
        "attractor_check_vs_F2": check, "attractor_spread": spread, "prototypes": {f"{a} | {e}": m for a, e, m in rows},
        "n_sample": N_SAMPLE, "n_per_mouse": N_PER_MOUSE, "n_iter": N_ITER, "seed": SEED,
    }, "analysis/d1e_flow_own_context.py")
    print(spread)


if __name__ == "__main__":
    main()
