"""F6: discrete strategy types or a continuum? Mixture models and distance correspondence.

    python analysis/f6_embedding_types.py

1. Mixtures. Gaussian mixtures (full covariance, 10 restarts) with k = 1…8 components on the
   614 × 4 embedding; BIC per k. A Gaussian null: N_NULL samples of 614 points from one
   Gaussian with the embedding's mean and covariance, each fitted the same way, give the null
   distribution of the BIC improvement of the best k over k = 1. An observed improvement above
   the null means structure beyond one Gaussian cloud — but heavy tails alone also produce it,
   so a second, heavy-tailed unimodal null is used: a multivariate t with the same mean and
   covariance and degrees of freedom matched to the mean excess kurtosis of the whitened data
   (df = 4 + 6/κ). Discrete types need a gain above both nulls. Also reported: the fitted
   components' mean separation (Mahalanobis) and covariance-scale ratio. Repeated on the
   per-mouse behaviour (the animals' own kernel features, standardised) as a model-free check.
2. Distance correspondence (Mantel). Spearman correlation between pairwise embedding distances
   (Mahalanobis, i.e. in the embedding's own metric) and pairwise behavioural distances:
   the animals' kernel features, the GRU's kernel features, and the fitted RL parameters
   (standardised, Euclidean); permutation null (shuffling mice of one matrix).
Writes ``f6_embedding_types.json`` and ``fig_F6_embedding_types.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402
from scipy.stats import kurtosis, skew, spearmanr  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402

SEED = 14
KS = range(1, 9)
N_NULL = 100
N_PERM = 1000


def bic_curve(x: np.ndarray, seed: int) -> np.ndarray:
    return np.array([GaussianMixture(k, covariance_type="full", n_init=10, random_state=seed).fit(x).bic(x) for k in KS])


def mantel(d1: np.ndarray, d2: np.ndarray, n: int, rng) -> dict:
    iu = np.triu_indices(n, 1)
    rho = spearmanr(d1, d2)[0]
    from scipy.spatial.distance import squareform  # noqa: PLC0415

    m2 = squareform(d2)
    null = []
    for _ in range(N_PERM):
        p = rng.permutation(n)
        null.append(spearmanr(d1, m2[np.ix_(p, p)][iu])[0])
    return {"rho": float(rho), "null_95": float(np.percentile(null, 95)), "p": float((1 + np.sum(np.array(null) >= rho)) / (N_PERM + 1))}


def main() -> None:
    s12.apply_presentation_style()
    table = pd.read_parquet(s12.CACHE / f"f5_mouse_table_{s12.MODEL_KEY}.parquet")
    emb = s12.subject_embeddings().loc[table.index]
    x = emb.to_numpy()
    rng = np.random.default_rng(SEED)

    def standardise(df):
        df = df.fillna(df.median())
        return ((df - df.mean()) / df.std()).to_numpy()

    feats = {"animal behaviour": standardise(table[[c for c in table.columns if c.startswith("animal ")]]),
             "GRU behaviour": standardise(table[[c for c in table.columns if c.startswith("gru ") and "advantage" not in c
                                                 and "likelihood" not in c]]),
             "RL parameters": standardise(table[[c for c in table.columns if c.startswith(("hattori ", "bari ", "ctt "))]])}

    mix = {}
    for name, data in (("embedding", x), ("animal behaviour", feats["animal behaviour"])):
        bic = bic_curve(data, SEED)
        mean, cov = data.mean(0), np.cov(data, rowvar=False)
        # Eigen-based square roots so collinear features (singular covariance) are handled.
        evals, evecs = np.linalg.eigh(cov)
        keep = evals > 1e-9 * evals.max()
        root = evecs[:, keep] * np.sqrt(evals[keep])              # cov = root @ root.T
        white = (data - mean) @ (evecs[:, keep] / np.sqrt(evals[keep]))
        kappa = float(np.mean(kurtosis(white, axis=0)))
        df = 4 + 6 / kappa if kappa > 0 else np.inf
        null_gain, null_t = [], []
        for i in range(N_NULL):
            sample = rng.multivariate_normal(mean, cov, size=len(data))
            b = bic_curve(sample, SEED + i)
            null_gain.append(b[0] - b.min())
            if np.isfinite(df):
                z = rng.standard_normal((len(data), int(keep.sum()))) @ (root * np.sqrt((df - 2) / df)).T
                sample_t = mean + z / np.sqrt(rng.chisquare(df, len(data)) / df)[:, None]
                bt = bic_curve(sample_t, SEED + 1000 + i)
                null_t.append(bt[0] - bt.min())
        g2 = GaussianMixture(2, covariance_type="full", n_init=10, random_state=SEED).fit(data)
        dmean = g2.means_[0] - g2.means_[1]
        ld = [np.linalg.slogdet(c + 1e-9 * np.eye(len(c)))[1] for c in g2.covariances_]
        scale = float(np.exp((ld[1] - ld[0]) / (2 * data.shape[1])))
        mix[name] = {"bic": bic, "best_k": int(KS[int(np.argmin(bic))]), "gain_best_vs_1": float(bic[0] - bic.min()),
                     "null_gain_95": float(np.percentile(null_gain, 95)), "null_gain": np.array(null_gain),
                     "t_df": float(df), "t_null_gain_median": float(np.median(null_t)) if null_t else None,
                     "t_null_gain_95": float(np.percentile(null_t, 95)) if null_t else None, "t_null_gain": np.array(null_t),
                     "k2_weights": g2.weights_, "k2_mean_separation_mahalanobis": float(np.sqrt(dmean @ np.linalg.pinv(cov) @ dmean)),
                     "k2_cov_scale_ratio": float(max(scale, 1 / scale)),
                     "skew": skew(data, axis=0), "excess_kurtosis": kurtosis(data, axis=0), "mean_excess_kurtosis_whitened": kappa}
        print(name, {k: (np.round(v, 2).tolist() if isinstance(v, np.ndarray) and v.size < 20 else v) for k, v in mix[name].items()
                     if k not in ("bic", "null_gain", "t_null_gain")}, flush=True)

    d_emb = pdist(x, metric="mahalanobis", VI=np.linalg.inv(np.cov(x, rowvar=False)))
    man = {name: mantel(d_emb, pdist(f), len(x), rng) for name, f in feats.items()}
    print("mantel", {k: (round(v["rho"], 3), round(v["null_95"], 3)) for k, v in man.items()}, flush=True)

    xc = x - x.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(xc, rowvar=False))
    pcs = xc @ evecs[:, np.argsort(evals)[::-1]]
    best = mix["embedding"]["best_k"]
    k_show = best if best > 1 else 2
    labels = GaussianMixture(k_show, covariance_type="full", n_init=10, random_state=SEED).fit(x).predict(x)

    fig, axes = plt.subplots(2, 4, figsize=(30, 13))
    for ax, name in zip(axes[0, :2], mix):
        m = mix[name]
        ax.plot(list(KS), m["bic"] - m["bic"][0], color="#222222", marker="o", lw=2, label="observed")
        ax.axhline(-m["null_gain_95"], color="#C44E52", ls="--", label="Gaussian null, 95th pct of the best-k gain")
        ax.set_xlabel("mixture components k")
        ax.set_ylabel("BIC(k) − BIC(1)")
        ax.set_title(f"{name}: best k = {m['best_k']}, gain {m['gain_best_vs_1']:.0f} (null 95% {m['null_gain_95']:.0f})", fontsize=13)
        ax.legend(frameon=False, fontsize=10)
    for ax, name in zip(axes[0, 2:], mix):
        ax.axvline(mix[name]["null_gain_95"], color="0.4", ls=":", lw=2, label=f"Gaussian null, 95th pct ({mix[name]['null_gain_95']:.0f})")
        if len(mix[name]["t_null_gain"]):
            ax.hist(mix[name]["t_null_gain"], bins=25, color="#4C72B0", alpha=0.5,
                    label=f"heavy-tailed null (t, df {mix[name]['t_df']:.1f})")
        ax.axvline(mix[name]["gain_best_vs_1"], color="#C44E52", lw=2.5, label="observed")
        ax.set_xlabel("BIC gain of the best k over k = 1")
        ax.set_ylabel("null samples")
        ax.set_title(f"{name}: beyond one Gaussian / one heavy-tailed cloud?", fontsize=13)
        ax.legend(frameon=False, fontsize=10)
    palette = plt.get_cmap("tab10")
    for ax, (i, j) in zip(axes[1, :2], ((0, 1), (2, 3))):
        for k in range(k_show):
            mk = labels == k
            ax.scatter(pcs[mk, i], pcs[mk, j], s=10, color=palette(k), label=f"component {k + 1} ({mk.sum()})")
        ax.set_xlabel(f"embedding PC{i + 1}")
        ax.set_ylabel(f"embedding PC{j + 1}")
        ax.set_title(f"GMM with k = {k_show}" + (" (best by BIC)" if best > 1 else " (illustration; BIC prefers k = 1)"), fontsize=13)
        ax.legend(frameon=False, fontsize=10)
    ax = axes[1, 2]
    d_beh = pdist(feats["animal behaviour"])
    sub = rng.choice(len(d_emb), size=min(40_000, len(d_emb)), replace=False)
    ax.hexbin(d_emb[sub], d_beh[sub], gridsize=50, cmap="Greys", bins="log")
    ax.set_xlabel("embedding distance (Mahalanobis)")
    ax.set_ylabel("behavioural distance (animals' kernels, standardised)")
    ax.set_title(f"Mantel: Spearman ρ = {man['animal behaviour']['rho']:.2f} (null 95% {man['animal behaviour']['null_95']:.2f})",
                 fontsize=13)
    ax = axes[1, 3]
    names = list(man)
    ax.barh(range(len(names)), [man[n]["rho"] for n in names], color=["#222222", "#E69F00", "#55A868"])
    ax.scatter([man[n]["null_95"] for n in names], range(len(names)), marker="|", s=150, color="#C44E52", label="permutation 95th pct")
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman ρ, embedding vs feature distance")
    ax.set_title("Do mice close in the embedding behave alike?", fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(s12.figure_path("fig_F6_embedding_types.png"))
    plt.close(fig)

    s12.write_json("f6_embedding_types.json", {
        "n_mice": int(len(x)), "ks": list(KS), "mixtures": {k: {kk: vv for kk, vv in v.items() if kk not in ("null_gain", "t_null_gain")} for k, v in mix.items()},
        "n_null": N_NULL, "mantel": man, "n_perm": N_PERM, "embedding_distance": "Mahalanobis",
    }, "analysis/f6_embedding_types.py")


if __name__ == "__main__":
    main()
