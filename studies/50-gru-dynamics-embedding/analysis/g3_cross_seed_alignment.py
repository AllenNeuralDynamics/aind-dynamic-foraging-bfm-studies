"""G3: are seeds 0–2 of model C the same model up to a change of basis?

    python analysis/g3_cross_seed_alignment.py

The three seeds were trained on the same mice and scored on the same eval trials, so their
trial tables are row-aligned (checked). This compares them directly:
  * behaviour: per-trial agreement of logit(right) and P(right) between seeds, set against
    the GRU-vs-CTT gap (C2) for scale;
  * hidden state: canonical correlations between seeds' top-10 PC scores, and held-out ridge
    R² of one seed's PCs 1–4 from another seed's full 128-d state;
  * cluster structure (B3c): share of PC1–4 variance explained by the last input (four
    clusters) and by the last choice alone (two lobes), per seed;
  * v (I2/I4): correlation of the value coordinate across seeds, trial by trial;
  * embedding (F): canonical correlations between seeds' 4-d subject embeddings and the
    Spearman correlation of their between-mouse distance matrices (Mahalanobis).
Writes ``g3_cross_seed_alignment.json``, ``fig_G3a_state_space_by_seed.png`` (PC1–PC2 and
PC1–PC4 by last input, each seed in its own PCA, side by side) and
``fig_G3b_cross_seed_alignment.png``.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402
from g1_seed_robustness import KEYS, SEED_COLORS  # noqa: E402

N_SUB = 200_000
N_CCA = 10
SEED = 31
INPUT_COLORS = {"R rewarded": "#0072B2", "R unrewarded": "#9ECAE1", "L rewarded": "#D55E00", "L unrewarded": "#FDAE6B"}


def canonical_correlations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Canonical correlations of two centred data matrices (rows = samples), via QR + SVD."""
    qa, _ = np.linalg.qr(a - a.mean(0))
    qb, _ = np.linalg.qr(b - b.mean(0))
    return np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0, 1)


def heldout_r2(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Ridge from x to each column of y, fit on even rows, scored on odd rows."""
    xm = np.c_[x - x.mean(0), np.ones(len(x))]
    tr, te = slice(0, None, 2), slice(1, None, 2)
    w = np.linalg.solve(xm[tr].T @ xm[tr] + alpha * np.eye(xm.shape[1]), xm[tr].T @ y[tr])
    resid = y[te] - xm[te] @ w
    return 1 - resid.var(0) / y[te].var(0)


def eta2(scores: np.ndarray, labels: np.ndarray) -> float:
    """Share of the total variance (summed over columns) explained by group means."""
    total = ((scores - scores.mean(0)) ** 2).sum()
    between = sum(((scores[labels == g].mean(0) - scores.mean(0)) ** 2).sum() * (labels == g).sum()
                  for g in np.unique(labels))
    return float(between / total)


def main() -> None:
    s12.apply_presentation_style()
    path0 = s12.trial_table_path(KEYS[0])
    meta_cols = ["ses_idx", "trial_idx", "subject_id", "action", "prev_action", "prev_outcome",
                 "logit_diff", "p_model1_right"]
    base = pq.read_table(path0, columns=meta_cols).to_pandas()
    rng = np.random.default_rng(SEED)
    valid = np.flatnonzero(base["prev_action"].notna().to_numpy())
    sub = np.sort(rng.choice(valid, N_SUB, replace=False))
    sub_t = base.iloc[sub]
    prev_input = np.where(sub_t["prev_action"] == 1, "R", "L").astype(object) + np.where(
        sub_t["prev_outcome"] == 1, " rewarded", " unrewarded")
    prev_choice = sub_t["prev_action"].to_numpy()

    logit, p_right, v, pcs, states, emb = {}, {}, {}, {}, {}, {}
    for key in KEYS:
        t = pq.read_table(s12.trial_table_path(key), columns=["ses_idx", "trial_idx", "logit_diff", "p_model1_right"]).to_pandas()
        if not ((t["ses_idx"].to_numpy() == base["ses_idx"].to_numpy()).all()
                and (t["trial_idx"].to_numpy() == base["trial_idx"].to_numpy()).all()):
            raise AssertionError(f"{key}'s trial table is not row-aligned with {KEYS[0]}'s")
        logit[key], p_right[key] = t["logit_diff"].to_numpy(), t["p_model1_right"].to_numpy()
        v[key] = np.load(s12.CACHE / f"value_coord_{key}.npy")
        x = pq.read_table(s12.trial_table_path(key), columns=s12.STATE_COLUMNS).take(sub)
        states[key] = np.column_stack([c.to_numpy() for c in x.columns]).astype(np.float64)
        pca = s12.load_pca(key)
        pcs[key] = (states[key] - pca["mean"]) @ pca["components"][:N_CCA].T
        emb[key] = s12.subject_embeddings(key)
        del t, x

    # Per-trial log-likelihood of the animal's choice, to put seed-to-seed gaps on C2's scale.
    action = base["action"].to_numpy()
    ll = {k: np.log(np.where(action == 1, p_right[k], 1 - p_right[k])) for k in KEYS}

    pairs = list(itertools.combinations(KEYS, 2))
    out = {"n_subsample": N_SUB, "seed": SEED, "n_cca_pcs": N_CCA, "pairs": {}}
    per_seed = {}
    for key in KEYS:
        per_seed[key] = {
            "eta2_pc1_4_last_input": eta2(pcs[key][:, :4], prev_input),
            "eta2_pc1_4_last_choice": eta2(pcs[key][:, :4], prev_choice),
            "eta2_pc1_last_choice": eta2(pcs[key][:, :1], prev_choice),
            "pooled_likelihood": float(np.exp(ll[key].mean())),
        }
    out["per_seed"] = per_seed

    subjects = emb[KEYS[0]].index
    for a, b in pairs:
        ea, eb = emb[a].loc[subjects].to_numpy(), emb[b].loc[subjects].to_numpy()
        # Mahalanobis distances: whiten each seed's embedding by its own covariance first.
        wa = (ea - ea.mean(0)) @ np.linalg.inv(np.linalg.cholesky(np.cov(ea.T))).T
        wb = (eb - eb.mean(0)) @ np.linalg.inv(np.linalg.cholesky(np.cov(eb.T))).T
        out["pairs"][f"{a} vs {b}"] = {
            "logit_corr": float(np.corrcoef(logit[a], logit[b])[0, 1]),
            "p_right_mean_abs_diff": float(np.abs(p_right[a] - p_right[b]).mean()),
            "ll_mean_abs_diff_nats": float(np.abs(ll[a] - ll[b]).mean()),
            "ll_mean_diff_nats": float((ll[a] - ll[b]).mean()),
            "v_corr": float(np.corrcoef(v[a], v[b])[0, 1]),
            "state_cca_top10": canonical_correlations(pcs[a], pcs[b]).tolist(),
            "pcs1_4_r2_from_other_state": heldout_r2(states[a], pcs[b][:, :4]).tolist(),
            "pcs1_4_r2_reverse": heldout_r2(states[b], pcs[a][:, :4]).tolist(),
            "embedding_cca": canonical_correlations(ea, eb).tolist(),
            "embedding_distance_spearman": float(spearmanr(pdist(wa), pdist(wb)).statistic),
        }
    # Scale reference: how far the GRU and the best RL model are apart on the same trials (C2).
    c2 = __import__("json").loads((s12.STUDY / "analysis" / "c2_compare.json").read_text())
    out["reference_gru_minus_ctt_nats"] = c2["advantage_mean_nats_per_trial"]["ctt"]
    out["reference_gru_ctt_corr_logit"] = c2["agreement"]["ctt"]["corr_logit"]
    # Same unsigned per-trial measures between seed 0 and CTT, so seed-to-seed gaps have a like-for-like scale.
    ctt = pq.read_table(s12.CACHE / f"rl_trials_ctt_{KEYS[0]}.parquet", columns=["ses_idx", "trial_idx", "p_rl_right"]).to_pandas()
    if not (ctt["trial_idx"].to_numpy() == base["trial_idx"].to_numpy()).all():
        raise AssertionError("CTT rollout is not row-aligned with the trial table")
    p_ctt = np.clip(ctt["p_rl_right"].to_numpy(), 1e-6, 1 - 1e-6)
    ll_ctt = np.log(np.where(action == 1, p_ctt, 1 - p_ctt))
    out["reference_seed0_vs_ctt"] = {
        "p_right_mean_abs_diff": float(np.abs(p_right[KEYS[0]] - p_ctt).mean()),
        "ll_mean_abs_diff_nats": float(np.abs(ll[KEYS[0]] - ll_ctt).mean()),
    }
    s12.write_json("g3_cross_seed_alignment.json", out, produced_by="analysis/g3_cross_seed_alignment.py")

    # G3a: each seed in its own PCA, side by side; PC signs flipped to correlate with seed 0's.
    fig, axes = plt.subplots(2, len(KEYS), figsize=(5.2 * len(KEYS), 9.5))
    show = rng.choice(N_SUB, 40_000, replace=False)
    for j, key in enumerate(KEYS):
        sc = pcs[key].copy()
        for c in range(4):
            if np.corrcoef(sc[:, c], pcs[KEYS[0]][:, c])[0, 1] < 0:
                sc[:, c] *= -1
        for i, (cy, label) in enumerate([(1, "PC2"), (3, "PC4")]):
            ax = axes[i, j]
            for inp, col in INPUT_COLORS.items():
                sel = show[prev_input[show] == inp]
                ax.scatter(sc[sel, 0], sc[sel, cy], s=1, alpha=0.25, color=col, rasterized=True,
                           label=inp if (i, j) == (0, 0) else None)
            ax.set_xlabel("PC1")
            ax.set_ylabel(label)
            if i == 0:
                share = s12.load_pca(key)["eigvals"]
                share = share / share.sum()
                ax.set_title(f"{key}  (PC1 {share[0]:.0%}, PC2 {share[1]:.0%}, PC4 {share[3]:.0%})\n"
                             f"η² last input = {per_seed[key]['eta2_pc1_4_last_input']:.2f}",
                             color=SEED_COLORS[key])
    axes[0, 0].legend(markerscale=8, fontsize=8, title="t−1 input", loc="best")
    fig.suptitle("G3a · hidden state by last input, each seed in its own PCA (signs matched to seed 0)", y=1.0)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_G3a_state_space_by_seed.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # G3b: alignment summary, one panel per comparison type.
    fig, axes = plt.subplots(1, 5, figsize=(28, 5.5))
    pair_cols = ["#555555", "#009E73", "#CC79A7"]
    for (name, res), col in zip(out["pairs"].items(), pair_cols):
        axes[0].plot(range(1, N_CCA + 1), res["state_cca_top10"], "o-", color=col, label=name)
        axes[3].plot(range(1, 5), res["embedding_cca"], "o-", color=col, label=name)
        axes[4].bar(name, res["embedding_distance_spearman"], color=col)
    axes[0].set(xlabel="canonical component", ylabel="canonical correlation", ylim=(0, 1.02),
                title="hidden state: top-10 PC scores")
    axes[0].legend(fontsize=8)
    axes[3].set(xlabel="canonical component", ylabel="canonical correlation", ylim=(0, 1.02),
                title="subject embedding (4-d, 614 mice)")
    axes[4].set(ylim=(0, 1), ylabel="Spearman ρ", title="between-mouse distances (Mahalanobis)")
    axes[4].tick_params(axis="x", labelsize=7)
    hb = rng.choice(len(base), 300_000, replace=False)
    for ax, (what, data) in zip(axes[1:3], [("logit(right)", logit), ("value coordinate v", v)]):
        for (a, b), col in zip([(KEYS[0], KEYS[1]), (KEYS[0], KEYS[2])], pair_cols[:2]):
            ax.scatter(data[a][hb], data[b][hb], s=0.5, alpha=0.08, color=col, rasterized=True)
        r1 = out["pairs"][f"{KEYS[0]} vs {KEYS[1]}"]["logit_corr" if what.startswith("logit") else "v_corr"]
        r2 = out["pairs"][f"{KEYS[0]} vs {KEYS[2]}"]["logit_corr" if what.startswith("logit") else "v_corr"]
        lo = min(np.nanmin(data[k]) for k in KEYS)
        hi = max(np.nanmax(data[k]) for k in KEYS)
        ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.8)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel=f"{KEYS[0]} {what}", ylabel=f"other seed {what}",
               title=f"{what}\nr(s0, s1) = {r1:.3f}, r(s0, s2) = {r2:.3f}")
    fig.suptitle("G3b · cross-seed alignment (scatters: seed 1 vs seed 0 in grey, seed 2 vs seed 0 in green; "
                 "v's unit is each seed's own TDR axis, so only its correlation is comparable)", y=1.03)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_G3b_cross_seed_alignment.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(__import__("json").dumps({"per_seed": per_seed, "pairs": {k: {kk: (vv if not isinstance(vv, list) else [round(x, 3) for x in vv])
                                                                    for kk, vv in p.items()} for k, p in out["pairs"].items()}}, indent=1))


if __name__ == "__main__":
    main()
