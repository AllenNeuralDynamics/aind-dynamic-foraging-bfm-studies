"""F10: how the subject embedding lays out each behavioural aspect (stickiness, memory, matching, bias).

    python analysis/f10_embedding_behaviour_maps.py

Pure post-hoc on cached artifacts: F7's per-mouse feature library
(``_cache/f7_mouse_features_<key>.parquet``, part == "all"), F5's mouse table (RL parameters,
kernel features) and the 4-d subject embedding.

1. Maps (fig_F10a: stickiness family; fig_F10b: memory, adaptation, matching, bias). Each mouse
   in the embedding's standardised PC1–PC2 and PC3–PC4; the background is a Gaussian-kernel
   (bandwidth BW SDs) local mean of the feature's percentile rank across mice, masked where
   fewer than MIN_EFF mice contribute. Animal and GRU side by side. A 2-d map averages over the
   two PCs not shown.
2. Directions (fig_F10c arrows; fig_F10d heatmaps, decodability, PC shares). For each feature and source: the Pearson correlation of its rank with
   each standardised embedding PC. The PCs are uncorrelated, so the squared correlations sum to
   the in-sample linear R² and their shares say which PCs carry the feature. Arrows in the two PC
   planes; heatmaps; and F7's cross-validated rank R² against the reliability ceiling.
3. Tuning (fig_F10e). Each feature's rank is predicted from the embedding by 5-fold CV linear
   regression fitted on the animal feature; mice are binned by deciles of that prediction, and
   the median (IQR) of the animal's and the GRU's raw feature values is shown per decile. The
   top − bottom decile contrast is given as a share of a perfect sort (deciles formed on the
   feature itself). The animal curve is
   held-out evidence; the GRU curve shows the model's feature along the same direction.
4. Direction cosines (JSON only). Are stickiness, learning rate and bias separate directions?
   For each pair of animal features (plus F5's RL learning rates, reward gain and persistence),
   the cosine of their embedding directions in the embedding-covariance metric (= correlation
   of the in-sample embedding predictions; invariant to linear re-parameterisation), the
   Spearman of their held-out predictions, and the Spearman of the raw features across mice.
5. Prototype basis (fig_F10f). One prototype per aspect: switch rate (stickiness level), CTT
   learning rate, bias. Their rank directions in the whitened embedding (where the
   embedding-covariance metric is Euclidean) are Gram–Schmidt-orthogonalised in that order;
   the fourth axis is the remainder. Mice get uncorrelated unit-variance coordinates. Plane 1–2
   holds the switch-rate and learning-rate gradients exactly, at their true angle; bias lives
   almost entirely on axis 3. Panels: each plane coloured by each prototype, with the three
   gradient arrows (unit-length directions projected on the plane). A points-only twin
   (``_points``) drops the smoothed background; ``_best`` shows each prototype once, in the plane
   that shows its gradient best (switch rate: 1–3, learning rate: 1–2, bias: 2–3).
Writes ``_cache/f10_prototype_basis_<key>.parquet`` (for F11), ``_cache/f10_prototype_transform_<key>.npz`` (for F12), ``f10_embedding_behaviour_maps.json`` and ``fig_F10[a-f]_*.png``.
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
from scipy.stats import rankdata  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.model_selection import KFold, cross_val_predict  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402

SEED = 15
BW, MIN_EFF, GRID = 0.35, 4.0, 120
N_BINS = 10
STICKY = ["determinism", "win-stay", "lose-stay", "stay through losses", "stickiness ratio", "switch rate"]
OTHER = ["choice memory span", "reward memory span", "poorer-side rate", "return after failed switch",
         "matching sensitivity", "matching bias", "bias"]
FEATURES = STICKY + OTHER
SOURCES = ("animal", "gru")
PLANES = ((0, 1), (2, 3))
COSINE_FEATURES = ["determinism", "win-stay", "lose-stay", "switch rate", "stickiness ratio", "choice memory span", "bias"]
COSINE_F5 = ["animal reward gain (1 back)", "animal persistence (1 back)", "ctt learn_rate", "bari learn_rate"]


def embedding_pcs(emb: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Standardised PC scores of the embedding (same PCs as F5), and their variance shares."""
    x = emb.to_numpy()
    xc = x - x.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(xc, rowvar=False))
    order = np.argsort(evals)[::-1]
    pcs = xc @ evecs[:, order]
    return pcs / pcs.std(0), evals[order] / evals.sum()


def pct_rank(y: np.ndarray) -> np.ndarray:
    out = np.full(len(y), np.nan)
    ok = np.isfinite(y)
    out[ok] = (rankdata(y[ok]) - 0.5) / ok.sum()
    return out


def smooth_map(px: np.ndarray, py: np.ndarray, v: np.ndarray, xg: np.ndarray, yg: np.ndarray) -> np.ndarray:
    ok = np.isfinite(v)
    gx, gy = np.meshgrid(xg, yg)
    d2 = (gx[..., None] - px[ok]) ** 2 + (gy[..., None] - py[ok]) ** 2
    w = np.exp(-0.5 * d2 / BW ** 2)
    wsum = w.sum(-1)
    m = (w * v[ok]).sum(-1) / np.maximum(wsum, 1e-12)
    return np.where(wsum >= MIN_EFF, m, np.nan)


def plot_maps(feats: dict, pcs: np.ndarray, share: np.ndarray, rel: dict, names: list[str], fname: str, title: str) -> None:
    lims = [(pcs[:, k].min() - 0.2, pcs[:, k].max() + 0.2) for k in range(4)]
    cols = [(src, pl) for src in SOURCES for pl in PLANES]
    fig, axes = plt.subplots(len(names), 4, figsize=(19, 4.3 * len(names)), layout="constrained")
    for i, f in enumerate(names):
        for j, (src, (pa, pb)) in enumerate(cols):
            ax = axes[i, j]
            v = pct_rank(feats[src][f].to_numpy(dtype=float))
            xg, yg = np.linspace(*lims[pa], GRID), np.linspace(*lims[pb], GRID)
            im = ax.imshow(smooth_map(pcs[:, pa], pcs[:, pb], v, xg, yg), origin="lower", aspect="auto",
                           extent=(*lims[pa], *lims[pb]), cmap="RdBu_r", vmin=0.15, vmax=0.85)
            ok = np.isfinite(v)
            ax.scatter(pcs[ok, pa], pcs[ok, pb], s=5, c=v[ok], cmap="RdBu_r", vmin=0, vmax=1,
                       edgecolors="k", linewidths=0.15)
            ax.scatter(pcs[~ok, pa], pcs[~ok, pb], s=5, color="0.6", marker="x", linewidths=0.5)
            r = rel[src][f]
            ax.set_title(f"{src}: {f}\nCV rank R² {r['embedding_rank_r2']:.2f} (reliability² {r['reliability_spearman_brown'] ** 2:.2f})",
                         fontsize=11)
            ax.set_xlabel(f"embedding PC{pa + 1} ({100 * share[pa]:.0f}%, SD units)")
            ax.set_ylabel(f"embedding PC{pb + 1} ({100 * share[pb]:.0f}%, SD units)")
            ax.axhline(0, color="0.7", lw=0.5)
            ax.axvline(0, color="0.7", lw=0.5)
    fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01,
                 label=f"local mean percentile rank across mice (Gaussian kernel {BW} SD; dots: each mouse)")
    fig.suptitle(title + "\nleft two columns: animal's own choices; right two: GRU predictions on the same trials. "
                 "Each 2-d map averages over the two PCs not shown.", fontsize=15)
    fig.savefig(s12.figure_path(fname), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    s12.apply_presentation_style()
    table = pd.read_parquet(s12.CACHE / f"f7_mouse_features_{s12.MODEL_KEY}.parquet").query("part == 'all'")
    emb = s12.subject_embeddings()
    feats = {src: table.query("source == @src").set_index("subject_id")[FEATURES] for src in SOURCES}
    mice = emb.index.intersection(feats["animal"].index).intersection(feats["gru"].index)
    feats = {src: f.loc[mice] for src, f in feats.items()}
    pcs, share = embedding_pcs(emb.loc[mice])
    f7 = json.loads((s12.ANALYSIS / "f7_feature_library.json").read_text())["per_feature"]
    rel = {src: {f: f7[f"{src} | {f}"] for f in FEATURES} for src in SOURCES}
    print("mice", len(mice), "| PC shares", np.round(share, 3), flush=True)

    # (1) Maps.
    plot_maps(feats, pcs, share, rel, STICKY, "fig_F10a_embedding_stickiness_maps.png",
              "How the embedding lays out stickiness: determinism, win-stay, lose-stay, staying through losses, "
              "stickiness ratio (UnrC/RewC), switch rate")
    plot_maps(feats, pcs, share, rel, OTHER, "fig_F10b_embedding_memory_matching_maps.png",
              "How the embedding lays out memory, adaptation, matching and bias")

    # (2) Directions: correlation of each feature's rank with each standardised PC.
    corr = {}
    for src in SOURCES:
        rows = {}
        for f in FEATURES:
            v = pct_rank(feats[src][f].to_numpy(float))
            ok = np.isfinite(v)
            rows[f] = [np.corrcoef(v[ok], pcs[ok, k])[0, 1] for k in range(4)]
        corr[src] = pd.DataFrame(rows, index=[f"PC{k + 1}" for k in range(4)]).T
    fig, axes = plt.subplots(1, 4, figsize=(30, 8.6), layout="constrained")
    groups = {f: ("C3" if f in STICKY else "C0") for f in FEATURES}
    for ax, (src, (pa, pb)) in zip(axes, [(s, p) for s in SOURCES for p in PLANES]):
        c = corr[src]
        for n, f in enumerate(FEATURES, 1):
            u, w = c.loc[f, f"PC{pa + 1}"], c.loc[f, f"PC{pb + 1}"]
            ax.annotate("", (u, w), (0, 0), arrowprops={"arrowstyle": "-|>", "color": groups[f], "lw": 1.5, "alpha": 0.8})
            ax.text(u * 1.09, w * 1.09, str(n), fontsize=10, color=groups[f], ha="center", va="center", fontweight="bold")
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(t), np.sin(t), color="0.8", lw=0.8)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.axhline(0, color="0.8", lw=0.6)
        ax.axvline(0, color="0.8", lw=0.6)
        ax.set_xlabel(f"correlation with embedding PC{pa + 1} ({100 * share[pa]:.0f}%)")
        ax.set_ylabel(f"correlation with embedding PC{pb + 1} ({100 * share[pb]:.0f}%)")
        ax.set_title(f"{src}: PC{pa + 1}–PC{pb + 1}", fontsize=14)
    key = "   ".join(f"{n} {f}" for n, f in enumerate(FEATURES, 1))
    fig.text(0.5, -0.03, "red: stickiness family; blue: memory, adaptation, matching, bias.   " + key, ha="center", fontsize=12,
             wrap=True)
    fig.suptitle("Where each behavioural aspect points in the embedding: the correlation of each feature's rank with the "
                 "standardised embedding PCs (arrow length ≤ 1; unit circle grey)", fontsize=15)
    fig.savefig(s12.figure_path("fig_F10c_embedding_feature_directions.png"), bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(28, 8.5), layout="constrained", gridspec_kw={"width_ratios": [1, 1, 1.1, 1.1]})
    for ax, src in zip(axes, SOURCES):
        c = corr[src]
        im = ax.imshow(c.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        for (a, b), val in np.ndenumerate(c.to_numpy()):
            ax.text(b, a, f"{val:+.2f}", ha="center", va="center", fontsize=10)
        r2 = (c ** 2).sum(1)
        ax.set_yticks(range(len(FEATURES)), [f"{f}  (R² {r2[f]:.2f})" for f in FEATURES])
        ax.set_xticks(range(4), [f"PC{k + 1}\n({100 * share[k]:.0f}%)" for k in range(4)])
        ax.set_title(f"{src}: r(feature rank, embedding PC)\nin-sample R² = sum of squares", fontsize=13)
    fig.colorbar(im, ax=axes[:2], fraction=0.03, pad=0.01, label="Pearson r")
    ax = axes[2]
    y = np.arange(len(FEATURES))
    for src, dy, col in (("animal", -0.18, "k"), ("gru", 0.18, "C1")):
        ax.barh(y + dy, [rel[src][f]["embedding_rank_r2"] for f in FEATURES], height=0.34, color=col, alpha=0.75,
                label=f"{src}: CV rank R² from the embedding")
        ax.scatter([rel[src][f]["reliability_spearman_brown"] ** 2 for f in FEATURES], y + dy, marker="|", s=160,
                   color=col, label=f"{src}: reliability² (ceiling)")
    ax.set_yticks(y, FEATURES)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("rank R² (F7: 5-fold CV)")
    ax.legend(frameon=False, fontsize=10, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.set_title("How much of each feature the embedding carries,\nagainst what split-half reliability allows", fontsize=13)
    ax = axes[3]
    sh = (corr["animal"] ** 2).div((corr["animal"] ** 2).sum(1), axis=0)
    left = np.zeros(len(FEATURES))
    for k in range(4):
        vals = sh[f"PC{k + 1}"].to_numpy()
        ax.barh(y, vals, left=left, color=plt.get_cmap("tab10")(k + 4), label=f"PC{k + 1}")
        left += vals
    ax.set_yticks(y, FEATURES)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the animal feature's in-sample R² carried by each PC")
    ax.legend(frameon=False, ncol=4, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.set_title("Which embedding PCs carry each animal feature", fontsize=13)
    fig.suptitle("How much of each behavioural aspect the embedding carries, and on which PCs", fontsize=16)
    fig.savefig(s12.figure_path("fig_F10d_embedding_feature_encoding.png"), bbox_inches="tight")
    plt.close(fig)

    # (3) Tuning along each feature's held-out embedding direction, in raw units.
    folds = KFold(5, shuffle=True, random_state=SEED)
    x = emb.loc[mice].to_numpy()
    tuning = {}
    ncol = 5
    nrow = int(np.ceil(len(FEATURES) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.4 * nrow), layout="constrained")
    xs = (np.arange(N_BINS) + 0.5) / N_BINS
    for ax, f in zip(axes.ravel(), FEATURES):
        ya = feats["animal"][f].to_numpy(float)
        ok = np.isfinite(ya)
        pred = np.full(len(ya), np.nan)
        pred[ok] = cross_val_predict(LinearRegression(), x[ok], pct_rank(ya)[ok], cv=folds)
        b = np.full(len(ya), -1)
        b[ok] = np.minimum((pct_rank(pred)[ok] * N_BINS).astype(int), N_BINS - 1)
        entry = {}
        for src, col in (("animal", "k"), ("gru", "C1")):
            v = feats[src][f].to_numpy(float)
            q = np.array([np.nanpercentile(v[b == k], [25, 50, 75]) for k in range(N_BINS)])
            ax.fill_between(xs, q[:, 0], q[:, 2], color=col, alpha=0.15, lw=0)
            ax.plot(xs, q[:, 1], "o-", color=col, ms=4, label=f"{src} (median, IQR)")
            # The contrast a perfect predictor would reach: the same deciles formed on the feature itself.
            vv = v[np.isfinite(v)]
            own = np.minimum((pct_rank(vv) * N_BINS).astype(int), N_BINS - 1)
            best = np.median(vv[own == N_BINS - 1]) - np.median(vv[own == 0])
            entry[src] = {"decile_quartiles": q.tolist(), "top_minus_bottom_decile": float(q[-1, 1] - q[0, 1]),
                          "perfect_sort_contrast": float(best),
                          "share_of_perfect_contrast": float((q[-1, 1] - q[0, 1]) / best)}
        tuning[f] = entry
        ax.set_title(f"{f}\ntop − bottom decile: animal {100 * entry['animal']['share_of_perfect_contrast']:.0f}%, "
                     f"GRU {100 * entry['gru']['share_of_perfect_contrast']:.0f}% of a perfect sort", fontsize=11)
        ax.set_xlabel("held-out embedding prediction (decile of mice)")
        ax.set_ylabel(f)
        ax.set_xlim(0, 1)
    for ax in axes.ravel()[len(FEATURES):]:
        ax.axis("off")
    axes.ravel()[0].legend(frameon=False)
    fig.suptitle("Each behavioural feature along its held-out embedding direction (fitted on the animal feature, 5-fold CV): "
                 "median and interquartile range of the mice in each decile", fontsize=14)
    fig.savefig(s12.figure_path("fig_F10e_embedding_feature_tuning.png"), bbox_inches="tight")
    plt.close(fig)

    # (4) Direction cosines between the main aspects.
    t5 = pd.read_parquet(s12.CACHE / f"f5_mouse_table_{s12.MODEL_KEY}.parquet")
    aspects = pd.DataFrame({f"animal {f}": feats["animal"][f] for f in COSINE_FEATURES}
                           | {c: t5.loc[mice, c] for c in COSINE_F5})
    sigma = np.cov(x, rowvar=False)
    coef, held = {}, {}
    for c in aspects:
        r = pct_rank(aspects[c].to_numpy(float))
        ok = np.isfinite(r)
        coef[c] = LinearRegression().fit(x[ok], r[ok]).coef_
        held[c] = np.full(len(r), np.nan)
        held[c][ok] = cross_val_predict(LinearRegression(), x[ok], r[ok], cv=folds)
    cosine = pd.DataFrame({a: {b: float(coef[a] @ sigma @ coef[b] / np.sqrt((coef[a] @ sigma @ coef[a]) * (coef[b] @ sigma @ coef[b])))
                               for b in aspects} for a in aspects})
    print("embedding-covariance cosines\n", cosine.round(2).to_string(), flush=True)

    # (5) Prototype basis: switch rate, learning rate (⊥), bias (⊥⊥), remainder.
    evals, evecs = np.linalg.eigh(sigma)
    xw = (x - x.mean(0)) @ evecs @ np.diag(evals ** -0.5) @ evecs.T
    proto = {"switch rate": aspects["animal switch rate"], "learning rate (CTT)": aspects["ctt learn_rate"],
             "bias": aspects["animal bias"]}
    dirs = {}
    for name, v in proto.items():
        r = pct_rank(v.to_numpy(float))
        ok = np.isfinite(r)
        d = LinearRegression().fit(xw[ok], r[ok]).coef_
        dirs[name] = d / np.linalg.norm(d)
    basis = []
    for d in list(dirs.values()) + list(np.eye(4)):
        u = d - sum((d @ b) * b for b in basis)
        if np.linalg.norm(u) > 1e-6 and len(basis) < 4:
            basis.append(u / np.linalg.norm(u))
    basis = np.array(basis)
    coords = xw @ basis.T
    scale = coords.std(0)
    coords /= scale
    np.savez(s12.CACHE / f"f10_prototype_transform_{s12.MODEL_KEY}.npz", mean=x.mean(0),
             whiten=evecs @ np.diag(evals ** -0.5) @ evecs.T, basis=basis, scale=scale)  # coords = ((x − mean) whiten basisᵀ) / scale
    in_basis = {n: (basis @ d).tolist() for n, d in dirs.items()}
    # Cached for F11: basis coordinates and each prototype's embedding-predicted percentile.
    cache = pd.DataFrame(coords, index=pd.Index(mice, name="subject_id"), columns=[f"axis{k + 1}" for k in range(4)])
    for name, v in proto.items():
        r = pct_rank(v.to_numpy(float))
        ok = np.isfinite(r)
        cache[f"pred pct: {name}"] = pct_rank(LinearRegression().fit(xw[ok], r[ok]).predict(xw))
        cache[f"animal: {name}"] = v.to_numpy(float)
    cache.to_parquet(s12.CACHE / f"f10_prototype_basis_{s12.MODEL_KEY}.parquet")
    axis_names = ["axis 1: switch rate", "axis 2: learning rate ⊥ switch rate", "axis 3: bias ⊥ both", "axis 4: remainder"]
    plane_pairs = [(0, 1), (0, 2), (1, 2)]
    arrow_col = {"switch rate": "C3", "learning rate (CTT)": "C2", "bias": "C0"}
    for smoothed, fname in ((True, "fig_F10f_prototype_basis.png"), (False, "fig_F10f_prototype_basis_points.png")):
        fig, axes = plt.subplots(len(plane_pairs), 3, figsize=(19, 6.0 * len(plane_pairs)), layout="constrained")
        for i, (pa, pb) in enumerate(plane_pairs):
            lim_a = (coords[:, pa].min() - 0.2, coords[:, pa].max() + 0.2)
            lim_b = (coords[:, pb].min() - 0.2, coords[:, pb].max() + 0.2)
            for j, (name, v) in enumerate(proto.items()):
                ax = axes[i, j]
                r = pct_rank(v.to_numpy(float))
                ok = np.isfinite(r)
                if smoothed:
                    xg, yg = np.linspace(*lim_a, GRID), np.linspace(*lim_b, GRID)
                    im = ax.imshow(smooth_map(coords[:, pa], coords[:, pb], r, xg, yg), origin="lower", aspect="equal",
                                   extent=(*lim_a, *lim_b), cmap="RdBu_r", vmin=0.15, vmax=0.85)
                    ax.scatter(coords[ok, pa], coords[ok, pb], s=5, c=r[ok], cmap="RdBu_r", vmin=0, vmax=1,
                               edgecolors="k", linewidths=0.15)
                else:
                    im = ax.scatter(coords[ok, pa], coords[ok, pb], s=16, c=r[ok], cmap="RdBu_r", vmin=0, vmax=1,
                                    edgecolors="0.3", linewidths=0.2)
                    ax.scatter(coords[~ok, pa], coords[~ok, pb], s=10, color="0.6", marker="x", linewidths=0.6)
                    ax.set_aspect("equal")
                for n, comp in in_basis.items():
                    u, w = 2.2 * comp[pa], 2.2 * comp[pb]
                    ax.annotate("", (u, w), (0, 0), arrowprops={"arrowstyle": "-|>", "color": arrow_col[n], "lw": 2.2})
                    if np.hypot(u, w) > 0.5:
                        ax.text(1.1 * u, 1.1 * w, n, color=arrow_col[n], fontsize=10, fontweight="bold", ha="center", va="center")
                ax.set_xlim(*lim_a)
                ax.set_ylim(*lim_b)
                ax.set_xlabel(axis_names[pa] + " (SD)")
                ax.set_ylabel(axis_names[pb] + " (SD)")
                ax.set_title(f"colour: animal {name} (percentile rank)", fontsize=12)
        fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01,
                     label=f"local mean percentile rank (Gaussian kernel {BW} SD); dots: mice" if smoothed
                     else "percentile rank across mice (each dot one mouse)")
        cos_sl = float(dirs["switch rate"] @ dirs["learning rate (CTT)"])
        fig.suptitle("The embedding in a basis built from three prototype features (whitened embedding, Gram–Schmidt: switch rate → "
                     "learning rate → bias)\narrows: each prototype's gradient direction (length 2.2 × its component in the plane); "
                     f"switch rate vs learning rate cosine {cos_sl:+.2f} ({np.degrees(np.arccos(cos_sl)):.0f}°)", fontsize=14)
        fig.savefig(s12.figure_path(fname), bbox_inches="tight")
        plt.close(fig)
    # One panel per prototype, in the plane that shows its gradient best (points only), titled with
    # its CV rank R² from the 4-d embedding as already frozen by F7 (switch rate, bias) and F5 (CTT).
    best_plane = {"switch rate": (0, 2), "learning rate (CTT)": (0, 1), "bias": (1, 2)}
    f5_dec = json.loads((s12.ANALYSIS / "f5_embedding_atlas.json").read_text())["numeric_decoding"]
    cv_r2 = {"switch rate": rel["animal"]["switch rate"]["embedding_rank_r2"],
             "learning rate (CTT)": f5_dec["ctt learn_rate"]["rank_r2"], "bias": rel["animal"]["bias"]["embedding_rank_r2"]}
    short_axis = ["axis 1 (switch rate)", "axis 2 (learning rate ⊥)", "axis 3 (bias ⊥)", "axis 4"]
    used = sorted({k for pair in best_plane.values() for k in pair})
    lim = (coords[:, used].min() - 0.2, coords[:, used].max() + 0.2)  # one square range: equal panels, one scale
    fig, axes = plt.subplots(1, 3, figsize=(20, 7.4), layout="constrained")
    for ax, (name, (pa, pb)) in zip(axes, best_plane.items()):
        r = pct_rank(proto[name].to_numpy(float))
        ok = np.isfinite(r)
        im = ax.scatter(coords[ok, pa], coords[ok, pb], s=22, c=r[ok], cmap="RdBu_r", vmin=0, vmax=1,
                        edgecolors="0.3", linewidths=0.2)
        ax.scatter(coords[~ok, pa], coords[~ok, pb], s=14, color="0.6", marker="x", linewidths=0.7)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal")
        ax.set_xlabel(short_axis[pa], fontsize=18)
        ax.set_ylabel(short_axis[pb], fontsize=18)
        ax.tick_params(labelsize=15)
        ax.set_title(f"{name}\ndecoding R² = {cv_r2[name]:.2f}", fontsize=21)
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("percentile rank across mice", fontsize=17)
    cb.ax.tick_params(labelsize=14)
    fig.suptitle("Embedding space", fontsize=23)
    fig.savefig(s12.figure_path("fig_F10f_prototype_basis_best.png"), bbox_inches="tight")
    plt.close(fig)
    var_basis = {n: {axis_names[k]: float(np.corrcoef(pct_rank(v.to_numpy(float))[np.isfinite(v.to_numpy(float))],
                                                      coords[np.isfinite(v.to_numpy(float)), k])[0, 1]) for k in range(4)}
                 for n, v in proto.items()}
    print("prototype directions in the basis", {n: np.round(c, 2) for n, c in in_basis.items()}, flush=True)
    print("r(prototype rank, basis axis)", {n: {k[:6]: round(c, 2) for k, c in d.items()} for n, d in var_basis.items()}, flush=True)

    for f in FEATURES:
        print(f"{f:28s} animal r(PC1-4) {np.round(corr['animal'].loc[f].to_numpy(), 2)}  gru {np.round(corr['gru'].loc[f].to_numpy(), 2)}"
              f"  decile contrast {tuning[f]['animal']['share_of_perfect_contrast']:+.2f} / {tuning[f]['gru']['share_of_perfect_contrast']:+.2f}",
              flush=True)
    s12.write_json("f10_embedding_behaviour_maps.json", {
        "n_mice": int(len(mice)), "embedding_pc_variance_share": share.tolist(),
        "smoothing": {"bandwidth_sd": BW, "min_effective_mice": MIN_EFF},
        "pc_correlation": {src: corr[src].to_dict(orient="index") for src in SOURCES},
        "in_sample_r2": {src: (corr[src] ** 2).sum(1).to_dict() for src in SOURCES},
        "tuning": tuning,
        "direction_cosines": {"embedding_covariance": cosine.to_dict(),
                              "heldout_prediction_spearman": pd.DataFrame(held).corr(method="spearman").to_dict(),
                              "raw_feature_spearman": aspects.corr(method="spearman").to_dict()},
        "prototype_basis": {"order": list(proto), "axes": axis_names, "directions_in_basis": in_basis,
                            "prototype_axis_correlation": var_basis},
    }, "analysis/f10_embedding_behaviour_maps.py")


if __name__ == "__main__":
    main()
