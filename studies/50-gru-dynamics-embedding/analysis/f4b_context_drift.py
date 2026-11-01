"""F4b (context half of F4): how each mouse's context moves with training, and what it does.

    PYTHONPATH=<wrapper>/code python analysis/f4b_context_drift.py     # GPU (JAX) for part 3

The GRU's context c = subject embedding + a session delta (an MLP of the embedding and the
session's training phase), so a mouse's context moves across its sessions. Using F4's split
(each mouse's eval sessions at the median training-session index; ``_cache/f4_session_halves``):
1. Size. Share of context variance within mice (across sessions) vs between mice.
2. Direction. Each mouse's drift Δc = mean late context − mean early context, expressed on
   the embedding's own PCs (F5's dictionary: PC2 attractor depth, PC3 learning rate / gain,
   PC4 bias). Population consistency: median component, share of mice moving the same way,
   Wilcoxon p.
3. Effect (counterfactual, GPU). Each mouse's late sessions are rerun with every late
   context replaced by the mouse's mean early context; C1's Su2022 logistic is refit on the
   late rows with the GRU's soft labels, actual vs counterfactual. Δ_context(f) = f(actual
   late) − f(late with early context) is the part of the GRU's early→late change in feature
   f caused by the context drift (input history held as observed). Compared per mouse with
   the GRU's total change Δ_GRU(f) and the animal's change Δ_animal(f) from F4
   (``_cache/f4_mouse_deltas``).
Writes ``f4b_context_drift.json`` and ``fig_F4b_context_drift.png``.
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
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402

BATCH = 512
SPLIT = "all sessions"


def main() -> None:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from c1_kernels import design, fit_rows  # noqa: PLC0415
    from f4_early_late import FEATURES, MIN_ROWS, features  # noqa: PLC0415
    from post_training_analysis.gru_counterfactual import load_gru_split  # noqa: PLC0415
    from post_training_analysis.gru_step import GruStep, split_inputs  # noqa: PLC0415

    s12.apply_presentation_style()
    ctx = np.load(s12.CACHE / f"contexts_{s12.MODEL_KEY}.npz")
    halves = pd.read_parquet(s12.CACHE / f"f4_session_halves_{s12.MODEL_KEY}.parquet").query("split == @SPLIT").set_index("ses_idx")["half"]
    t = s12.load_trials(["subject_id", "ses_idx"]).drop_duplicates("ses_idx").set_index("ses_idx")["subject_id"]
    c = pd.DataFrame(ctx["contexts"], index=pd.Index(ctx["session_ids"], name="ses_idx"))
    c["subject_id"] = t.reindex(c.index).to_numpy()
    c["half"] = halves.reindex(c.index).to_numpy()
    dims = [0, 1, 2, 3]

    # 1. Variance decomposition.
    mouse_mean = c.groupby("subject_id")[dims].transform("mean")
    within = float(((c[dims] - mouse_mean) ** 2).to_numpy().sum())
    total = float(((c[dims] - c[dims].mean()) ** 2).to_numpy().sum())
    share_within = within / total

    # 2. Drift on the embedding PCs.
    emb = s12.subject_embeddings()
    e = emb.to_numpy()
    evals, evecs = np.linalg.eigh(np.cov(e - e.mean(0), rowvar=False))
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    halfmean = c.dropna(subset=["half"]).groupby(["subject_id", "half"])[dims].mean().unstack("half")
    halfmean = halfmean.dropna()
    early = np.stack([halfmean[(d, "early")] for d in dims], axis=1)
    late = np.stack([halfmean[(d, "late")] for d in dims], axis=1)
    drift = late - early
    drift_pc = drift @ evecs
    spread_pc = np.sqrt(evals[order])  # between-mouse SD of the embedding along each PC
    drift_summary = {f"PC{k + 1}": {"median": float(np.median(drift_pc[:, k])), "median_in_between_mouse_sd": float(np.median(drift_pc[:, k]) / spread_pc[k]),
                                    "abs_median_in_between_mouse_sd": float(np.median(np.abs(drift_pc[:, k])) / spread_pc[k]),
                                    "share_positive": float(np.mean(drift_pc[:, k] > 0)), "wilcoxon_p": float(wilcoxon(drift_pc[:, k]).pvalue)}
                     for k in range(4)}
    print("share of context variance within mice", round(share_within, 3), drift_summary, flush=True)

    # 3. Counterfactual: late sessions with the early mean context.
    split = load_gru_split(s12.CACHE / "models" / s12.MODEL_KEY / "run")
    step = GruStep.from_params(split.params)
    _, _, obs = split_inputs(split.xs, session_conditioning=split.run.session_conditioning_enabled)
    if list(ctx["session_ids"]) != split.session_ids:
        raise AssertionError("Cached contexts are not in the split's session order")
    ctx_actual = ctx["contexts"].astype(np.float32)
    ctx_cf = ctx_actual.copy()
    early_mean = pd.DataFrame(early, index=halfmean.index)
    for i, sid in enumerate(split.session_ids):
        if c.at[sid, "half"] == "late" and c.at[sid, "subject_id"] in early_mean.index:
            ctx_cf[i] = early_mean.loc[c.at[sid, "subject_id"]].to_numpy(dtype=np.float32)
    run = jax.jit(lambda o, cc: step.logit_diff(step.run(o, cc)))

    def logits(contexts):
        out = np.empty(obs.shape[:2], dtype=np.float32)
        for start in range(0, obs.shape[1], BATCH):
            idx = np.arange(start, min(start + BATCH, obs.shape[1]))
            pad = np.r_[idx, np.full(BATCH - len(idx), idx[-1])]
            cc = np.broadcast_to(contexts[pad][None], obs[:, pad].shape[:2] + (contexts.shape[1],))
            out[:, idx] = np.asarray(run(jnp.asarray(obs[:, pad]), jnp.asarray(cc)))[:, :len(idx)]
        return out

    z_act, z_cf = logits(ctx_actual), logits(ctx_cf)
    x, td = design()
    col = td["ses_idx"].map({s: i for i, s in enumerate(split.session_ids)}).to_numpy()
    tau = td["trial_idx"].to_numpy() - 1
    p_act = 1 / (1 + np.exp(-z_act[tau, col].astype(np.float64)))
    p_cf = 1 / (1 + np.exp(-z_cf[tau, col].astype(np.float64)))
    parity = float(np.median(np.abs(p_act - td["p_model1_right"].to_numpy())))
    print("replay vs cached P(right), median |diff|", round(parity, 5), flush=True)
    late_rows = td["ses_idx"].map(halves).to_numpy() == "late"
    jobs = []
    for subject, rows in td.groupby("subject_id", sort=False).indices.items():
        rows = rows[late_rows[rows]]
        if len(rows) >= MIN_ROWS:
            jobs += [(subject, "actual", rows), (subject, "early context", rows)]
    coefs = Parallel(n_jobs=24)(delayed(fit_rows)(x, p_act if kind == "actual" else p_cf, rows, True) for _, kind, rows in jobs)
    feat = pd.DataFrame([{"subject_id": s, "kind": k, **features(cf)} for (s, k, _), cf in zip(jobs, coefs)])
    wide = feat.pivot_table(index="subject_id", columns="kind", values=list(FEATURES))
    d_context = pd.DataFrame({f: wide[(f, "actual")] - wide[(f, "early context")] for f in FEATURES})
    deltas = pd.read_parquet(s12.CACHE / f"f4_mouse_deltas_{s12.MODEL_KEY}.parquet").query("split == @SPLIT")
    d_gru = deltas.query("source == 'gru'").pivot_table(index="subject_id", columns="feature", values="delta")
    d_animal = deltas.query("source == 'animal'").pivot_table(index="subject_id", columns="feature", values="delta")
    mice = d_context.index.intersection(d_gru.index)
    effect = {}
    for f in FEATURES:
        dc, dg, da = d_context.loc[mice, f], d_gru.loc[mice, f], d_animal.loc[mice, f]
        slope = float(np.polyfit(dg, dc, 1)[0])
        effect[f] = {"median_delta_context": float(np.median(dc)), "median_delta_gru_total": float(np.median(dg)),
                     "median_delta_animal": float(np.median(da)),
                     "spearman_context_vs_gru_total": float(spearmanr(dc, dg)[0]),
                     "spearman_context_vs_animal": float(spearmanr(dc, da)[0]),
                     "share_of_gru_change_variance_from_context": float(np.var(dc) / np.var(dg)),
                     "slope_context_on_gru_total": slope, "wilcoxon_p_context": float(wilcoxon(dc).pvalue)}
    print({f: {k: round(v, 3) for k, v in d.items()} for f, d in effect.items()}, flush=True)

    fig, axes = plt.subplots(2, 4, figsize=(30, 13))
    ax = axes[0, 0]
    ax.bar(["between mice", "within mice\n(across sessions)"], [1 - share_within, share_within], color=["#4C72B0", "#E69F00"])
    ax.set_ylabel("share of context variance")
    ax.set_title(f"How much does a mouse's context move? ({100 * share_within:.1f}% within)", fontsize=13)
    ax = axes[0, 1]
    parts = ax.violinplot([drift_pc[:, k] / spread_pc[k] for k in range(4)], showmedians=True)
    for b in parts["bodies"]:
        b.set_facecolor("#E69F00")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(range(1, 5), ["PC1\nmixed", "PC2\ndepth", "PC3\nlearning\nrate", "PC4\nbias"])
    ax.set_ylabel("late − early context (in between-mouse SDs)")
    ax.set_title("Drift along the embedding PCs, per mouse", fontsize=13)
    ax = axes[0, 2]
    sub = np.random.default_rng(0).choice(len(early), size=min(150, len(early)), replace=False)
    pe, pl = (early - e.mean(0)) @ evecs, (late - e.mean(0)) @ evecs
    ax.scatter((e - e.mean(0)) @ evecs[:, 2], (e - e.mean(0)) @ evecs[:, 3], s=5, color="0.85", label="all mice (embedding)")
    for i in sub:
        ax.annotate("", xy=(pl[i, 2], pl[i, 3]), xytext=(pe[i, 2], pe[i, 3]), arrowprops={"arrowstyle": "->", "color": "#C44E52", "lw": 0.8})
    ax.set_xlabel("embedding PC3 (learning rate / gain)")
    ax.set_ylabel("embedding PC4 (bias)")
    ax.set_title("Early → late context, 150 mice", fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    ax = axes[0, 3]
    xs = np.arange(len(FEATURES))
    for j, (key, color, lab) in enumerate((("median_delta_animal", "#999999", "animal Δ (F4)"),
                                           ("median_delta_gru_total", "#222222", "GRU Δ, total (F4)"),
                                           ("median_delta_context", "#E69F00", "GRU Δ caused by the context drift"))):
        ax.bar(xs + (j - 1) * 0.27, [effect[f][key] for f in FEATURES], width=0.27, color=color, label=lab)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(xs, ["bias", "reward gain\n(1 back)", "persistence\n(1 back)"])
    ax.set_ylabel("median late − early change")
    ax.set_title("Average change: animal, GRU, and the context's part", fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    for ax, f in zip(axes[1, :3], FEATURES):
        dc, dg = d_context.loc[mice, f], d_gru.loc[mice, f]
        lim = np.nanpercentile(np.abs(np.r_[dc, dg]), 99)
        ax.scatter(dg, dc, s=9, alpha=0.5, color="#4C72B0")
        ax.plot([-lim, lim], [-lim, lim], color="0.6", ls=":", lw=1)
        ax.axhline(0, color="0.85", lw=0.8)
        ax.axvline(0, color="0.85", lw=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"GRU total Δ {f} (late − early)")
        ax.set_ylabel(f"Δ caused by the context drift")
        r = effect[f]
        ax.set_title(f"{f}: Spearman {r['spearman_context_vs_gru_total']:+.2f}, variance share {r['share_of_gru_change_variance_from_context']:.2f}",
                     fontsize=12)
    ax = axes[1, 3]
    ax.barh(range(len(FEATURES)), [effect[f]["spearman_context_vs_animal"] for f in FEATURES], color="#999999")
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_yticks(range(len(FEATURES)), FEATURES)
    ax.set_xlabel("Spearman: context-caused Δ vs the animal's Δ")
    ax.set_title("Does the context drift track the animal's change?", fontsize=13)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_F4b_context_drift.png"))
    plt.close(fig)

    s12.write_json("f4b_context_drift.json", {
        "split": SPLIT, "n_mice_drift": int(len(drift)), "n_mice_counterfactual": int(len(mice)),
        "share_of_context_variance_within_mice": share_within, "drift_on_embedding_pcs": drift_summary,
        "counterfactual": "late sessions rerun with the mouse's mean early-half context; Su2022 soft-label refit on late rows",
        "replay_parity_median_abs_p": parity, "effect": effect,
    }, "analysis/f4b_context_drift.py")


if __name__ == "__main__":
    main()
