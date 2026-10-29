"""Producer for r2: how each GLM-HMM state spreads across subjects (offline).

Reads ``frozen_r2.json`` and ``frozen_r2_weights_<emission>.npz`` (one GLM per {subject,
state} under each K-grid run's global fit; see ``freeze_r2.py``) and writes
``r2_state_spread.json``, the figures ``fig_r2_*.svg`` and the result blocks of
``reports/r2-state-spread.md``:
- result-1: cluster scores against K, per emission and cohort;
- result-2: every {subject, state} in weight space, one panel per K.

Every distance is Euclidean in the raw weight space (logit units, bias included).
"""
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, pairwise_distances, silhouette_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
import figure_style as fs  # noqa: E402
from _meta import build_meta, write_artifact  # noqa: E402

WANDB_GROUPS = ["hurdle@20261027-212905", "reference@20261027-213006"]
FROZEN = HERE / "frozen_r2.json"
EMISSIONS = ("hurdle", "reference")
WEIGHTS = {e: HERE / f"frozen_r2_weights_{e}.npz" for e in EMISSIONS}
REPORT = HERE / "reports" / "r2-state-spread.md"
COHORTS = ("heldout", "training")
COHORT_LABEL = {"heldout": "held-out mice", "training": "training mice"}
COHORT_COLOR = {"heldout": fs.OKABE_ITO["blue"], "training": fs.OKABE_ITO["vermillion"]}
STATE_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442",
                "#000000", "#999999", "#882255"]
MIN_TRIALS = 200        # a {subject, state} cell with fewer Viterbi trials is dropped
N_SHUFFLES = 100        # within-subject label shuffles for the silhouette null
SEED = 0


def load():
    """The frozen runs, each with its kept cells' weights, state ranks and cohorts."""
    frozen = json.loads(FROZEN.read_text())
    cohort_of = np.array([s["cohort"] for s in frozen["subjects"]])
    arrays = {e: np.load(WEIGHTS[e]) for e in EMISSIONS}
    runs = []
    for run in frozen["runs"]:
        c = {k: np.asarray(v) for k, v in run["cells"].items()}
        w = arrays[run["emission"]][run["weights_key"]].astype(np.float64)
        K = run["num_states"]
        # States ranked by their trials over every frozen cell: state 1 is the most used.
        used = np.bincount(c["state"], weights=c["n_trials"], minlength=K)
        rank = np.empty(K, int)
        rank[np.argsort(-used, kind="stable")] = np.arange(K)
        keep = c["n_trials"] >= MIN_TRIALS
        runs.append({**run, "W": w[keep], "state": rank[c["state"][keep]],
                     "subject": c["subject"][keep], "cohort": cohort_of[c["subject"][keep]],
                     "n_trials": c["n_trials"][keep],
                     "G": np.asarray(run["global_weights"])[np.argsort(rank)]})
    return frozen, runs


def shuffle_within(labels, groups, rng):
    """``labels`` permuted among the cells of each group."""
    by_group = np.argsort(groups, kind="stable")
    permuted = np.lexsort((rng.random(len(groups)), groups))
    out = np.empty_like(labels)
    out[by_group] = labels[permuted]
    return out


def eta_squared(X, labels):
    """Share of the total sum of squares between the groups of ``labels``."""
    total = ((X - X.mean(0)) ** 2).sum()
    between = 0.0
    for g in np.unique(labels):
        Xg = X[labels == g]
        between += len(Xg) * ((Xg.mean(0) - X.mean(0)) ** 2).sum()
    return float(between / total)


def scores(X, state, subject, K, rng):
    """Cluster scores of one cohort at one K."""
    out = {"n_cells": int(len(X)), "n_subjects": int(len(np.unique(subject))),
           "eta2_subject": eta_squared(X, subject)}
    if K == 1 or len(np.unique(state)) < 2:
        return {**out, "silhouette": None, "silhouette_null_mean": None,
                "silhouette_null_95": None, "eta2_state": 0.0, "eta2_state_null_mean": None,
                "kmeans_ari": None}
    D = pairwise_distances(X)
    null_sil, null_eta = [], []
    for _ in range(N_SHUFFLES):
        shuffled = shuffle_within(state, subject, rng)
        null_sil.append(silhouette_score(D, shuffled, metric="precomputed"))
        null_eta.append(eta_squared(X, shuffled))
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(X)
    return {**out,
            "silhouette": float(silhouette_score(D, state, metric="precomputed")),
            "silhouette_null_mean": float(np.mean(null_sil)),
            "silhouette_null_95": [float(q) for q in np.percentile(null_sil, [2.5, 97.5])],
            "eta2_state": eta_squared(X, state),
            "eta2_state_null_mean": float(np.mean(null_eta)),
            "kmeans_ari": float(adjusted_rand_score(state, kmeans))}


def pca_axes(X):
    """Mean and first two principal axes of ``X``, and their share of the variance."""
    mean = X.mean(0)
    _, s, vt = np.linalg.svd(X - mean, full_matrices=False)
    # Fix each axis's sign so regenerating gives the same picture.
    vt = vt * np.sign(vt[np.arange(len(vt)), np.abs(vt).argmax(1)])[:, None]
    return mean, vt[:2], (s[:2] ** 2 / (s ** 2).sum()).tolist()


def fig_scores(result):
    """Cluster scores against K: silhouette vs its null, η² by state and subject, k-means ARI."""
    fig, axes = plt.subplots(2, 3, figsize=fs.mm(fs.fit_width(square_panels=3, axes=3), 85),
                             sharex=True)
    for row, emission in enumerate(EMISSIONS):
        for cohort in COHORTS:
            rows = result[emission][cohort]
            Ks = [r["K"] for r in rows if r["silhouette"] is not None]
            sel = [r for r in rows if r["silhouette"] is not None]
            color = COHORT_COLOR[cohort]
            ax = axes[row, 0]
            ax.plot(Ks, [r["silhouette"] for r in sel], "o-", color=color, ms=2,
                    label=COHORT_LABEL[cohort])
            ax.fill_between(Ks, [r["silhouette_null_95"][0] for r in sel],
                            [r["silhouette_null_95"][1] for r in sel], color=color, alpha=0.2, lw=0)
            ax = axes[row, 1]     # from K = 2: at K = 1 η² is 0 by state and 1 by subject
            ax.plot(Ks, [r["eta2_state"] for r in sel], "o-", color=color, ms=2)
            ax.plot(Ks, [r["eta2_subject"] for r in sel], "o--", color=color, ms=2, mfc="white")
            ax.plot(Ks, [r["eta2_state_null_mean"] for r in sel], ":", color=color, lw=0.8)
            axes[row, 2].plot(Ks, [r["kmeans_ari"] for r in sel], "o-", color=color, ms=2)
        axes[row, 0].set_ylabel(f"{emission}\nsilhouette")
        axes[row, 1].set_ylabel("η² (variance share)")
        axes[row, 2].set_ylabel("k-means ARI")
        axes[row, 1].set_ylim(0, 1)
        axes[row, 2].set_ylim(0, 1)
        axes[row, 0].axhline(0, color="0.6", lw=0.5)
    axes[0, 0].legend(loc="upper right")
    axes[0, 1].plot([], [], "k-", label="by state")
    axes[0, 1].plot([], [], "k--", mfc="white", label="by subject")
    axes[0, 1].plot([], [], "k:", label="by state, shuffled")
    axes[0, 1].legend(loc="upper right")
    for ax in axes[1]:
        ax.set_xlabel("K")
        ax.set_xticks(range(2, 11, 2))
        ax.set_xlim(1.5, 10.5)
    for ax, letter in zip(axes.flat, "abcdef"):
        fs.panel_label(ax, letter)
    return fig


def fig_scatter(runs, emission, pcs):
    """Every {subject, state} on the first two PCs of the training cells, one panel per K."""
    fig, axes = plt.subplots(4, 5, figsize=fs.mm(183, 160))
    by_k = {r["num_states"]: r for r in runs if r["emission"] == emission}
    for i, cohort in enumerate(COHORTS):
        for K in range(1, 11):
            ax = axes[2 * i + (K - 1) // 5, (K - 1) % 5]
            run = by_k[K]
            mean, axes_pc, share = pcs[emission][K]
            m = run["cohort"] == cohort
            P = (run["W"][m] - mean) @ axes_pc.T
            colors = np.asarray(STATE_COLORS)[run["state"][m]]
            ax.scatter(P[:, 0], P[:, 1], s=1.2, c=colors, lw=0, alpha=0.6, rasterized=True)
            # Limits from both cohorts' dots, so one far global star cannot squash the panel.
            P_all = (run["W"] - mean) @ axes_pc.T
            lo, hi = np.percentile(P_all, [0.5, 99.5], axis=0)
            pad = 0.05 * (hi - lo)
            G = (run["G"] - mean) @ axes_pc.T
            ax.scatter(G[:, 0], G[:, 1], s=18, c=STATE_COLORS[:K], marker="*", edgecolor="k",
                       lw=0.4, zorder=3, clip_on=True)
            ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
            ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
            ax.set_title(f"{COHORT_LABEL[cohort]}, K = {K}", pad=2)
            ax.set_xlabel(f"PC1 ({share[0]:.0%})", labelpad=1)
            ax.set_ylabel(f"PC2 ({share[1]:.0%})", labelpad=1)
    return fig


def fmt(x, spec=".2f"):
    """A table cell: the number, or a dash."""
    return "—" if x is None else format(x, spec)


def block_scores(result, frozen):
    """result-1: the cluster-score table and figure."""
    lines = [
        f"One dot = one {{subject, state}} GLM, fitted to the subject's trials on that state's "
        f"Viterbi path under the global fit (MAP, prior SD {frozen['data']['prior_sigma']:g}). "
        f"Cells with fewer than {MIN_TRIALS} trials are dropped. Distances are Euclidean in the "
        "raw weight space. *Silhouette*: how much closer a dot is to its own state's dots than to "
        "the nearest other state's (1 = tight separate clusters, 0 = overlapping). *Null*: the "
        f"same with state labels shuffled among each mouse's own dots ({N_SHUFFLES} shuffles; "
        "95% range). *η²*: share of the weight variance between states, or between subjects. "
        "*ARI*: agreement of k-means with K clusters with the state labels (0 = chance, 1 = "
        "identical). One global fit per K, so no error bars across fits.",
        "",
        "| emission | K | cohort | cells (dropped) | silhouette | null (95%) | η² state | η² state, null | η² subject | k-means ARI |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for emission in EMISSIONS:
        for K in range(1, 11):
            for cohort in COHORTS:
                r = next(x for x in result[emission][cohort] if x["K"] == K)
                null = (f"{r['silhouette_null_mean']:.2f} ({r['silhouette_null_95'][0]:.2f}–"
                        f"{r['silhouette_null_95'][1]:.2f})" if r["silhouette_null_95"] else "—")
                lines.append(
                    f"| {emission} | {K} | {cohort} | {r['n_cells']} ({r['n_dropped']}) | "
                    f"{fmt(r['silhouette'])} | {null} | {r['eta2_state']:.2f} | "
                    f"{fmt(r['eta2_state_null_mean'])} | {r['eta2_subject']:.2f} | "
                    f"{fmt(r['kmeans_ari'])} |")
    lines += ["", "![cluster scores](../fig_r2_scores.svg)"]
    return "\n".join(lines)


def block_scatter(pcs):
    """result-2: the weight-space scatter figures."""
    lines = [
        "Each panel: every {subject, state} GLM of one cohort at one K, on the first two principal "
        "components of that K's training-mouse dots (both cohorts share the axes). Colour = state, "
        "ranked by trials (state 1, blue, is the most used); colours match only within a K, "
        "because states are not matched across K. Stars: the global fit's own state weights (fitted "
        "with a weak prior, SD 100); axes span 99% of the dots, so a far star is not shown.",
        "",
    ]
    for emission in EMISSIONS:
        lines += [f"**{emission}**", "", f"![{emission} weight space](../fig_r2_scatter_{emission}.svg)", ""]
    return "\n".join(lines)


def replace_block(text, n, body):
    """Replace the content between the result-n markers."""
    pattern = re.compile(rf"(<!-- BEGIN result-{n} -->\n).*?(\n<!-- END result-{n} -->)", re.S)
    if not pattern.search(text):
        raise ValueError(f"result-{n} markers missing")
    return pattern.sub(lambda m: m.group(1) + body.strip("\n") + m.group(2), text)


def main() -> None:
    frozen, runs = load()
    rng = np.random.default_rng(SEED)
    cohort_of = np.array([s["cohort"] for s in frozen["subjects"]])
    result, pcs = {e: {c: [] for c in COHORTS} for e in EMISSIONS}, {e: {} for e in EMISSIONS}
    for run in sorted(runs, key=lambda r: (r["emission"], r["num_states"])):
        emission, K = run["emission"], run["num_states"]
        pcs[emission][K] = pca_axes(run["W"][run["cohort"] == "training"])
        for cohort in COHORTS:
            m = run["cohort"] == cohort
            row = {"K": K, "wandb_run_id": run["wandb_run_id"],
                   **scores(run["W"][m], run["state"][m], run["subject"][m], K, rng)}
            all_cells = np.asarray(run["cells"]["subject"])
            row["n_dropped"] = int(((cohort_of[all_cells] == cohort)
                                    & (np.asarray(run["cells"]["n_trials"]) < MIN_TRIALS)).sum()
                                   + run["n_cells_not_frozen"][cohort])
            result[emission][cohort].append(row)
        print(f"{emission} K={K}: " + ", ".join(
            f"{c} sil {fmt(result[emission][c][-1]['silhouette'])}" for c in COHORTS))

    fs.apply_paper_style()
    figs = [fs.save_figure(fig_scores(result), HERE / "fig_r2_scores.svg")]
    for emission in EMISSIONS:
        figs.append(fs.save_figure(fig_scatter(runs, emission, pcs), HERE / f"fig_r2_scatter_{emission}.svg"))
    plt.close("all")

    text = REPORT.read_text()
    text = replace_block(text, 1, block_scores(result, frozen))
    text = replace_block(text, 2, block_scatter(pcs))
    REPORT.write_text(text)

    out = {"min_trials": MIN_TRIALS, "n_shuffles": N_SHUFFLES, "scores": result,
           "pca_variance_share": {e: {str(K): v[2] for K, v in pcs[e].items()} for e in EMISSIONS}}
    meta = build_meta("analysis/r2_state_spread.py", WANDB_GROUPS, study_root=HERE.parent,
                      code=[__file__], inputs=[FROZEN, *WEIGHTS.values()])
    write_artifact(HERE / "r2_state_spread.json", {"_meta": meta, **out},
                   outputs=figs + [f"{REPORT}#result-{n}" for n in (1, 2)])


if __name__ == "__main__":
    main()
