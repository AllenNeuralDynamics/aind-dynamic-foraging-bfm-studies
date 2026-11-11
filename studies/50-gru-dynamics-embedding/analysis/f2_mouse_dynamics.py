"""F2 (stage A, GPU): attractors, timescales and v's retention for every mouse's context.

    PYTHONPATH=<wrapper>/code python analysis/f2_mouse_dynamics.py

D2 computed fixed points for five contexts. Here: every mouse's context (the mean of the
cached contexts of its eval sessions), each of the four constant inputs (previous choice ×
reward), fixed points of h ↦ F(h, input, c) reached from N_STARTS of the mouse's own states by
forward iteration plus damped Newton in float64 — the wrapper's ``find_fixed_points``
procedure, batched over mice × inputs × starts. Only stable points (spectral radius < 1) are
kept, deduplicated per mouse and input. Per attractor: logit(right), v (I2 value coordinate),
the three slowest |eigenvalues| (τ = −1/ln|λ|), the slowest mode's overlap with the readout,
and v's one-step retention e_vᵀ J e_v. Also the all-sessions mean context, checked against D2.

Per-mouse features (input names as I5: R+ = right rewarded, …):
  attractor logits; depth (mean |logit| over the four inputs); asymmetry (mean of the four
  logits, > 0 rightward); reward depth (rewarded − unrewarded |logit|); bistable (any input with
  two attractors); slowest τ; readout overlap of the slowest mode; v retention.
Related across mice to F3's GRU kernel features and I5's per-mouse dynamics fits (Spearman) and
decoded from the 4-d embedding (F3's rank-R² method). Writes ``f2_mouse_dynamics.json`` and
``fig_F2_mouse_dynamics.png``.
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
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402

N_STARTS = 96
N_FORWARD = 500
N_NEWTON = 30
DAMPING = 1e-8
TOL = 1e-12
DEDUP = 1e-3
BATCH = 8192
SEED = 9
INPUTS = {"R+": (1.0, 1.0), "R−": (1.0, 0.0), "L+": (0.0, 1.0), "L−": (0.0, 0.0)}
F3_FEATURES = ["bias", "reward gain (1 back)", "persistence (1 back)", "reward decay (2/1 back)"]
I5_FEATURES = ["λ stay", "λ switch", "reward push", "choice push", "target asymmetry", "switch target"]


def main() -> None:
    import jax  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp  # noqa: PLC0415

    from f3_embedding_features import decode, kernel_features  # noqa: PLC0415
    from i5_mouse_dynamics import dynamics_features  # noqa: PLC0415
    from post_training_analysis.gru_counterfactual import load_gru_split  # noqa: PLC0415
    from post_training_analysis.gru_step import GruStep  # noqa: PLC0415

    s12.apply_presentation_style()
    print("devices", jax.devices(), flush=True)
    split = load_gru_split(s12.CACHE / "models" / s12.MODEL_KEY / "run")
    step = GruStep.from_params(jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), split.params))
    ctx = np.load(s12.CACHE / f"contexts_{s12.MODEL_KEY}.npz")
    t = s12.load_trials(["subject_id", "ses_idx"])
    ses_subject = t.drop_duplicates("ses_idx").set_index("ses_idx")["subject_id"]
    ctx_df = pd.DataFrame(ctx["contexts"], index=ctx["session_ids"])
    ctx_df["subject_id"] = ses_subject.reindex(ctx_df.index).to_numpy()
    mouse_ctx = ctx_df.groupby("subject_id").mean()
    mice = list(mouse_ctx.index)
    contexts = np.vstack([mouse_ctx.to_numpy(), ctx["contexts"].mean(axis=0, keepdims=True)])  # last row: mean context
    labels = mice + ["all-sessions mean"]

    rng = np.random.default_rng(SEED)
    states = s12.load_trials(s12.STATE_COLUMNS).to_numpy(dtype=np.float32)
    rows_by_mouse = pd.Series(np.arange(len(t))).groupby(t["subject_id"].to_numpy()).indices
    starts = [states[rng.choice(rows_by_mouse[m], size=N_STARTS, replace=False)] for m in mice]
    starts.append(states[rng.choice(len(states), size=N_STARTS, replace=False)])
    starts = np.stack(starts).astype(np.float64)  # [M+1, S, H]
    del states
    n_ctx, hdim = len(contexts), starts.shape[-1]

    eye = jnp.eye(hdim, dtype=jnp.float64)

    @jax.jit
    def refine(h0, obs, c):
        h = jax.lax.fori_loop(0, N_FORWARD, lambda _, x: step.step(x, obs, c), h0)

        def newton(h, o, cc):
            a = step.jacobian(h, o, cc) - eye
            r = step.step(h, o, cc) - h
            return h + jnp.linalg.solve(a.T @ a + DAMPING * eye, -a.T @ r)

        h = jax.lax.fori_loop(0, N_NEWTON, lambda _, x: jax.vmap(newton)(x, obs, c), h)
        q = 0.5 * jnp.sum((step.step(h, obs, c) - h) ** 2, axis=-1)
        return h, q

    jac = jax.jit(jax.vmap(step.jacobian))
    tdr = np.load(s12.keyed_cache("i2_tdr_axes.npz"))
    e_v = tdr["basis"][:, [str(x) for x in tdr["terms"]].index("value")]
    pca_mean = s12.load_pca()["mean"]
    r = np.asarray(step.readout_w[:, 1] - step.readout_w[:, 0])
    r_unit = r / np.linalg.norm(r)

    cache = s12.CACHE / f"f2_fixed_points_{s12.MODEL_KEY}.parquet"
    records = pd.read_parquet(cache).to_dict("records") if cache.exists() else []
    for name, (ch, rw) in ({} if cache.exists() else INPUTS).items():  # reuse the cached search
        h0 = starts.reshape(-1, hdim)
        obs = np.tile(np.array([ch, rw]), (len(h0), 1))
        cc = np.repeat(contexts, N_STARTS, axis=0)
        hs, qs = [], []
        for s in range(0, len(h0), BATCH):
            h, q = refine(jnp.asarray(h0[s:s + BATCH]), jnp.asarray(obs[s:s + BATCH]), jnp.asarray(cc[s:s + BATCH]))
            hs.append(np.asarray(h))
            qs.append(np.asarray(q))
        h_all = np.concatenate(hs).reshape(n_ctx, N_STARTS, hdim)
        q_all = np.concatenate(qs).reshape(n_ctx, N_STARTS)
        print(name, "converged fraction", round(float(np.mean(q_all < TOL)), 3), flush=True)
        for i in range(n_ctx):
            found, counts = [], []
            for j in np.argsort(q_all[i]):
                if not np.isfinite(q_all[i, j]) or q_all[i, j] > TOL:
                    break
                d = [np.linalg.norm(h_all[i, j] - f) for f in found]
                if d and min(d) <= DEDUP:
                    counts[int(np.argmin(d))] += 1
                else:
                    found.append(h_all[i, j])
                    counts.append(1)
            if not found:
                records.append({"context": labels[i], "input": name, "n_converged": 0})
                continue
            fps = np.stack(found)
            jacs = np.asarray(jac(jnp.asarray(fps), jnp.asarray(np.tile([ch, rw], (len(fps), 1))),
                                  jnp.asarray(np.repeat(contexts[i:i + 1], len(fps), axis=0))))
            for f, n_hit, jm in zip(fps, counts, jacs):
                eig, vec = np.linalg.eig(jm)
                order = np.argsort(-np.abs(eig))
                eig, vec = eig[order], vec[:, order]
                if np.abs(eig[0]) >= 1:
                    continue  # saddle or unstable
                slow = np.real(vec[:, 0]) if np.isreal(eig[0]) else np.real(vec[:, 0]) / np.linalg.norm(np.real(vec[:, 0]))
                slow = slow / np.linalg.norm(slow)
                lam = np.abs(eig[:3])
                records.append({
                    "context": labels[i], "input": name, "n_converged": int(np.sum(q_all[i] < TOL)), "n_hits": n_hit,
                    "logit_right": float(f @ r + float(step.readout_b[1] - step.readout_b[0])),
                    "v": float((f - pca_mean) @ e_v), "abs_eig": lam.tolist(),
                    "tau_slowest": float(-1 / np.log(lam[0])), "slow_mode_readout_overlap": float(abs(slow @ r_unit)),
                    "v_retention": float(e_v @ jm @ e_v),
                })
    fp = pd.DataFrame(records)
    if not cache.exists():
        fp.to_parquet(cache)

    # D2 check: mean-context attractors.
    d2 = json.loads((s12.ANALYSIS / "d_dynamics.json").read_text())
    mean_rows = fp[fp["context"] == "all-sessions mean"].dropna(subset=["logit_right"])
    check = {name: sorted(mean_rows.loc[mean_rows["input"] == name, "logit_right"].round(2).tolist()) for name in INPUTS}
    print("mean-context attractor logits", check, flush=True)

    # Per-mouse features (dominant attractor = most hits per input).
    att = fp.dropna(subset=["logit_right"]).query("context != 'all-sessions mean'")
    dom = att.sort_values("n_hits", ascending=False).drop_duplicates(["context", "input"]).set_index(["context", "input"])
    n_att = att.groupby(["context", "input"]).size().unstack()
    logit = dom["logit_right"].unstack()[list(INPUTS)].dropna()  # mice with an attractor for every input
    feat = pd.DataFrame({
        "attractor depth": logit.abs().mean(axis=1),
        "attractor asymmetry": logit.mean(axis=1),
        "reward depth (rew − unrew)": (logit["R+"] - logit["R−"] - logit["L+"] + logit["L−"]) / 2,
        "bistable (any input)": (n_att.reindex(logit.index).fillna(0) > 1).any(axis=1).astype(float),
        "slowest τ (trials)": dom["tau_slowest"].unstack().reindex(logit.index).max(axis=1),
        "slow-mode readout overlap": dom["slow_mode_readout_overlap"].unstack().reindex(logit.index).mean(axis=1),
        "v retention (e_vᵀJe_v)": dom["v_retention"].unstack().reindex(logit.index).mean(axis=1),
    })
    for name in INPUTS:
        feat[f"attractor logit {name}"] = logit[name]
    feat.index.name = "subject_id"

    kernels = json.loads((s12.ANALYSIS / "c1_kernels.json").read_text())
    beh = kernel_features(kernels["per_mouse_coef"]["gru"], kernels["n_back"])[F3_FEATURES]
    p = np.load(s12.keyed_cache("i4_mouse_params.npz"), allow_pickle=False)
    i5 = dynamics_features(p["lam"], p["target"], p["a"], p["b"]).set_index(pd.Index([str(x) for x in p["subjects"]]))[I5_FEATURES]
    emb = s12.subject_embeddings()
    common = feat.index.intersection(beh.index).intersection(i5.index).intersection(emb.index)
    feat, beh, i5, x = feat.loc[common], beh.loc[common], i5.loc[common], emb.loc[common].to_numpy()
    main_feats = [c for c in feat.columns if not c.startswith("attractor logit")]
    rho_beh = pd.DataFrame({f: [spearmanr(feat[d], beh[f], nan_policy="omit")[0] for d in main_feats] for f in F3_FEATURES}, index=main_feats)
    rho_i5 = pd.DataFrame({f: [spearmanr(feat[d], i5[f], nan_policy="omit")[0] for d in main_feats] for f in I5_FEATURES}, index=main_feats)
    rng2 = np.random.default_rng(SEED)
    decoding = {d: decode(x, feat[d].to_numpy(dtype=float), rng2) for d in main_feats if feat[d].nunique() > 2}

    fig, axes = plt.subplots(2, 4, figsize=(32, 13))
    ax = axes[0, 0]
    parts = ax.violinplot([logit[n].dropna() for n in INPUTS], showmedians=True)
    for body, color in zip(parts["bodies"], ("#0072B2", "#56B4E9", "#D55E00", "#E69F00")):
        body.set_facecolor(color)
    ax.set_xticks(range(1, 5), [f"{n}\n{int((n_att[n] > 1).sum())} bistable" for n in INPUTS])
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_ylabel("attractor logit(right)")
    ax.set_title(f"Attractor of each constant input ({len(logit)} mice with all four)", fontsize=13)
    for ax, (fx, fy, title) in zip((axes[0, 1], axes[0, 2], axes[0, 3]), (
            ("attractor asymmetry", "bias", "Bias = shifted attractors?"),
            ("reward depth (rew − unrew)", "reward gain (1 back)", "Reward gain = deeper rewarded attractors?"),
            ("v retention (e_vᵀJe_v)", None, "v's retention at the attractor vs I5's fitted λ"))):
        if fy is None:
            ax.scatter(feat[fx], i5.loc[feat.index, "λ stay"], s=10, alpha=0.6, color="#4C72B0")
            rho = spearmanr(feat[fx], i5["λ stay"])[0]
            ax.set_ylabel("I5 λ stay (fitted on the GRU's v)")
        else:
            ax.scatter(feat[fx].rank(pct=True), beh[fy].rank(pct=True), s=10, alpha=0.6, color="#4C72B0")
            rho = rho_beh.at[fx, fy]
            ax.set_ylabel(f"GRU {fy} (rank)")
            fx = f"{fx} (rank)"
        ax.set_xlabel(fx)
        ax.set_title(f"{title}  ρ = {rho:+.2f}", fontsize=13)
    ax = axes[1, 0]
    ax.hist(feat["slowest τ (trials)"].clip(upper=1000), bins=40, color="#999999")
    ax.set_xlabel("slowest attractor timescale τ (trials, clipped at 1000)")
    ax.set_ylabel("mice")
    ax.set_title(f"Slow modes; median readout overlap {feat['slow-mode readout overlap'].median():.2f}", fontsize=13)
    for ax, rho, title in ((axes[1, 1], rho_beh, "Spearman: dynamics × GRU behaviour (F3)"),
                           (axes[1, 2], rho_i5, "Spearman: dynamics × I5 fitted dynamics")):
        im = ax.imshow(rho.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(rho.shape[1]), rho.columns, rotation=30, ha="right", fontsize=10)
        ax.set_yticks(range(rho.shape[0]), rho.index, fontsize=10)
        for a_ in range(rho.shape[0]):
            for b_ in range(rho.shape[1]):
                ax.text(b_, a_, f"{rho.iat[a_, b_]:+.2f}", ha="center", va="center", fontsize=12)
        ax.set_title(title, fontsize=13)
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.02)
    ax = axes[1, 3]
    names = list(decoding)
    ax.barh(range(len(names)), [decoding[n]["rank_r2"] for n in names], color="#E69F00")
    ax.scatter([decoding[n]["null_95"] for n in names], range(len(names)), marker="|", s=120, color="#C44E52",
               label="shuffle 95th pct")
    ax.set_yticks(range(len(names)), names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("rank R² from the 4-d embedding (5-fold CV)")
    ax.set_title("Is each dynamical feature set by the embedding?", fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_F2_mouse_dynamics.png"))
    plt.close(fig)

    conv = fp.groupby("input")["n_converged"].mean() / N_STARTS
    s12.write_json("f2_mouse_dynamics.json", {
        "n_mice": int(len(feat)), "n_starts": N_STARTS, "n_forward": N_FORWARD, "n_newton": N_NEWTON, "tol": TOL,
        "dedup": DEDUP, "precision": "float64", "converged_fraction_by_input": conv.to_dict(),
        "mean_context_attractor_logits": check, "d2_reference_logits": {k: [round(f["logit_right"], 2) for f in v["fixed_points"] if f["stable"]]
                                for k, v in d2["contexts"]["all sessions"]["inputs"].items()},
        "n_bistable_mice_by_input": {n: int((n_att[n] > 1).sum()) for n in INPUTS},
        "n_mice_with_all_four_attractors": int(len(logit)),
        "mice_without_a_converged_attractor": {n: sorted(set(mice) - set(att.loc[att["input"] == n, "context"])) for n in INPUTS},
        "feature_summary": {c: np.nanpercentile(feat[c], [10, 50, 90]).round(4).tolist() for c in feat.columns},
        "spearman_vs_f3": rho_beh.to_dict(), "spearman_vs_i5": rho_i5.to_dict(), "embedding_decoding": decoding,
    }, "analysis/f2_mouse_dynamics.py")
    print(rho_beh.round(2).to_string())
    print(rho_i5.round(2).to_string())
    print({d: round(v["rank_r2"], 3) for d, v in decoding.items()})
    print({c: np.nanpercentile(feat[c], [10, 50, 90]).round(3).tolist() for c in feat.columns})


if __name__ == "__main__":
    main()
