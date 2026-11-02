"""F8: embedding traversal — synthetic mice along each named axis (F7), behaviour measured causally.

    PYTHONPATH=<wrapper>/code python analysis/f8_embedding_traversal.py     # GPU (JAX)

F7 rotated the whitened embedding into four named axes. Here a synthetic mouse is placed at
α ∈ {−2, −1, 0, +1, +2} SDs along one axis from the mean mouse (embedding = mean + α·axis), and
its context = that embedding + the mean session delta of all real sessions (the delta is small,
F4c). The GRU is run with each synthetic context on the same N_SESSIONS real eval sessions
(the animals' choice / reward sequences as input, open loop), and every F7 feature is computed
from its predicted P(right) on those trials (soft labels, as F7's GRU source). The slope of each
feature against α (in between-mouse SDs of the real GRU feature) is the axis's causal effect,
compared with F7's correlational loading of that feature on the axis.
Writes ``f8_embedding_traversal.json`` and ``fig_F8_embedding_traversal.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402

SEED = 16
N_SESSIONS = 600
ALPHAS = [-2.0, -1.0, 0.0, 1.0, 2.0]
BATCH = 512


def main() -> None:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from c1_kernels import design, fit_rows  # noqa: PLC0415
    from f7_feature_library import FEATURES, N_BACK, kernel_derived, model_free  # noqa: PLC0415
    from post_training_analysis.gru_counterfactual import load_gru_split  # noqa: PLC0415
    from post_training_analysis.gru_step import GruStep, split_inputs  # noqa: PLC0415

    s12.apply_presentation_style()
    ax_npz = np.load(s12.CACHE / f"f7_named_axes_{s12.MODEL_KEY}.npz", allow_pickle=False)
    f7 = __import__("json").loads((s12.ANALYSIS / "f7_feature_library.json").read_text())
    names = f7["named_axes"]["names"]
    mean, whiten, rot = ax_npz["mean"], ax_npz["whiten"], ax_npz["rotation"]
    unwhiten = np.linalg.inv(whiten)  # whitened → centred embedding
    ctx = np.load(s12.CACHE / f"contexts_{s12.MODEL_KEY}.npz")
    t_all = s12.load_trials(["subject_id", "ses_idx"]).drop_duplicates("ses_idx").set_index("ses_idx")["subject_id"]
    emb = s12.subject_embeddings()
    delta = ctx["contexts"] - emb.loc[t_all.reindex(ctx["session_ids"]).to_numpy()].to_numpy()
    mean_delta = delta.mean(0)

    split = load_gru_split(s12.CACHE / "models" / s12.MODEL_KEY / "run")
    step = GruStep.from_params(split.params)
    _, _, obs = split_inputs(split.xs, session_conditioning=split.run.session_conditioning_enabled)
    rng = np.random.default_rng(SEED)
    cols = np.sort(rng.choice(obs.shape[1], size=N_SESSIONS, replace=False))
    sess = [split.session_ids[c] for c in cols]
    run = jax.jit(lambda o, cc: step.logit_diff(step.run(o, cc)))

    t = s12.load_trials(["subject_id", "ses_idx", "trial_idx", "action", "reward", "reward_probabilityL", "reward_probabilityR"])
    t = t[t["ses_idx"].isin(sess)].reset_index(drop=True)
    col_of = {s: i for i, s in enumerate(sess)}
    tc, tau = t["ses_idx"].map(col_of).to_numpy(), t["trial_idx"].to_numpy() - 1
    x, td = design(N_BACK)
    keep = td["ses_idx"].isin(sess).to_numpy()
    x, td = x[keep], td[keep].reset_index(drop=True)
    dc, dtau = td["ses_idx"].map(col_of).to_numpy(), td["trial_idx"].to_numpy() - 1

    def p_right(context: np.ndarray) -> np.ndarray:
        out = np.empty((obs.shape[0], N_SESSIONS), dtype=np.float32)
        for s in range(0, N_SESSIONS, BATCH):
            idx = np.arange(s, min(s + BATCH, N_SESSIONS))
            pad = np.r_[idx, np.full(BATCH - len(idx), idx[-1])]
            o = obs[:, cols[pad]]
            cc = np.broadcast_to(context[None, None].astype(np.float32), o.shape[:2] + (len(context),))
            out[:, idx] = np.asarray(run(jnp.asarray(o), jnp.asarray(cc)))[:, :len(idx)]
        return 1 / (1 + np.exp(-out.astype(np.float64)))

    conditions = [(k, a) for k in range(4) for a in ALPHAS]
    probs, contexts = {}, {}
    for k, a in conditions:
        e = mean + (a * rot[:, k]) @ unwhiten
        contexts[(k, a)] = e + mean_delta
        probs[(k, a)] = p_right(contexts[(k, a)])
    print("GRU runs done", flush=True)
    coefs = Parallel(n_jobs=20)(delayed(fit_rows)(x, probs[c][dtau, dc], None, True) for c in conditions)
    rows = []
    for (k, a), coef in zip(conditions, coefs):
        p = probs[(k, a)][tau, tc]
        rec = {"axis": k, "alpha": a, **model_free(t, p), **kernel_derived(coef, x)}
        rows.append(rec)
    res = pd.DataFrame(rows)

    real = pd.read_parquet(s12.CACHE / f"f7_mouse_features_{s12.MODEL_KEY}.parquet").query("part == 'all' and source == 'gru'")
    sd = real[FEATURES].std()
    slopes = np.array([[np.polyfit(ALPHAS, res.query("axis == @k")[f].to_numpy(dtype=float), 1)[0] / sd[f]
                        if res.query("axis == @k")[f].notna().all() else np.nan for k in range(4)] for f in FEATURES])
    load = np.array([f7["named_axes"]["loadings"][f] for f in FEATURES])
    agree = [float(np.corrcoef(np.nan_to_num(slopes[:, k]), load[:, k])[0, 1]) for k in range(4)]
    print("agreement causal vs correlational per axis", np.round(agree, 2), flush=True)

    fig, axes = plt.subplots(2, 4, figsize=(32, 13))
    palette = plt.get_cmap("tab20")
    for k in range(4):
        ax = axes[0, k]
        d = res.query("axis == @k")
        for i, f in enumerate(FEATURES):
            z = (d[f].to_numpy(dtype=float) - real[f].mean()) / sd[f]
            ax.plot(ALPHAS, z, marker="o", lw=2 if abs(load[i, k]) > 0.4 else 0.8, color=palette(i), label=f)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_xlabel("position along the axis (SDs of the embedding)")
        ax.set_ylabel("feature (z across real mice)")
        ax.set_title(f"axis {k + 1}: {names[k]}", fontsize=11)
        if k == 3:
            ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for ax, (mat, title) in zip(axes[1, :2], ((slopes, "Causal: feature change per axis SD (traversal)"),
                                                (load, "Correlational: F7 loadings"))):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_yticks(range(len(FEATURES)), FEATURES, fontsize=10)
        ax.set_xticks(range(4), [f"axis {k + 1}" for k in range(4)])
        for i in range(len(FEATURES)):
            for j in range(4):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=9)
        ax.set_title(title, fontsize=13)
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[1, 2]
    ax.bar(range(4), agree, color="#4C72B0")
    ax.set_xticks(range(4), [f"axis {k + 1}" for k in range(4)])
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_ylabel("correlation of causal slopes with F7 loadings")
    ax.set_title("Does moving along the axis do what the axis was named for?", fontsize=13)
    axes[1, 3].axis("off")
    axes[1, 3].text(0, 0.95, "\n".join(f"axis {k + 1}: {names[k]}" for k in range(4)), va="top", fontsize=12,
                    transform=axes[1, 3].transAxes)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_F8_embedding_traversal.png"))
    plt.close(fig)

    s12.write_json("f8_embedding_traversal.json", {
        "axes": names, "alphas": ALPHAS, "n_sessions": N_SESSIONS, "features": FEATURES,
        "context": "mean embedding + α·axis (F7 rotation, whitened units) + mean session delta",
        "features_by_condition": res.to_dict(orient="records"),
        "causal_slope_per_sd": {f: slopes[i].tolist() for i, f in enumerate(FEATURES)},
        "agreement_with_f7_loadings": agree,
    }, "analysis/f8_embedding_traversal.py")


if __name__ == "__main__":
    main()
