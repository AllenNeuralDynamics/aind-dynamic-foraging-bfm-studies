"""Producer for r3: how each GLM-HMM state spreads across subjects (offline).

Reads ``frozen_r3.json`` and its npz files (see ``freeze_r3.py``): one GLM per {subject,
state} under each K-grid run's global fit, for 18 ways of cutting trials into states (two
labellings, global fit or each mouse's round-2 refit, times nine posterior and run-length
cuts), with each setting's cluster scores, which the freeze computes with this file's
``scores``. Writes ``r3_state_spread.json``, the figures ``fig_r3_*.svg``, the result blocks
of ``reports/r3-state-spread.md``:
- result-1: cluster scores against K of the uncut global setting (``global_p0_m1``);
- result-2: that setting's {subject, state} cells in weight space, one panel per K;
- result-3: how the cuts and the refit labelling change the scores;
- result-4: checks of the round-2 refits;
and ``interactive/r3_state_spread.html`` with its data files, all of it to explore.

Every distance is Euclidean in the raw weight space (logit units, bias included).
"""
import base64
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

WANDB_GROUPS = ["hurdle@20261027-212905", "reference@20261027-213006", "refit-prior@20261101-130226"]
FROZEN = HERE / "frozen_r3.json"
EMISSIONS = ("hurdle", "reference")
LABELLINGS = ("global", "refit")
CELLS = {(e, lab): HERE / f"frozen_r3_cells_{e}_{lab}.npz" for e in EMISSIONS for lab in LABELLINGS}
WEIGHTS = {(e, lab): HERE / f"frozen_r3_weights_{e}_{lab}.npz" for e in EMISSIONS for lab in LABELLINGS}
REFITS = {e: HERE / f"frozen_r3_refits_{e}.npz" for e in EMISSIONS}
ANCHOR = "global_p0_m1"
REPORT = HERE / "reports" / "r3-state-spread.md"
COHORTS = ("heldout", "training")
COHORT_LABEL = {"heldout": "held-out mice", "training": "training mice"}
COHORT_COLOR = {"heldout": fs.OKABE_ITO["blue"], "training": fs.OKABE_ITO["vermillion"]}
STATE_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442",
                "#000000", "#999999", "#882255"]
CUT_COLORS = {0.0: "#0072B2", 0.8: "#E69F00", 0.95: "#D55E00"}   # posterior cut
CUT_DASH = {1: "-", 10: "--", 30: ":"}                              # run-length cut
INTERACTIVE = HERE / "interactive" / "r3_state_spread.html"
PLOTLY = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.34.0/plotly.min.js"
MIN_TRIALS = 200        # a {subject, state} cell with fewer kept trials is dropped
N_SHUFFLES = 100        # within-subject label shuffles for the silhouette null
BIG_CELL = 1000         # check: the silhouette on cells with at least this many trials
SEED = 0


def load():
    """The frozen runs, each with every setting's kept cells (state ranks, cohorts, coordinates).

    States are ranked by the anchor setting's trials (state 1 = most used) in every setting.
    Coordinates: PC1-3 of the anchor's training cells, then the six named weights.
    """
    frozen = json.loads(FROZEN.read_text())
    cohort_of = np.array([s["cohort"] for s in frozen["subjects"]])
    cells = {k: np.load(p) for k, p in CELLS.items()}
    refits = {e: np.load(p) for e, p in REFITS.items()}
    runs = []
    for run in frozen["runs"]:
        emission, K = run["emission"], run["num_states"]
        rank = np.asarray(run["state_rank"])
        settings = {}
        for key, entry in run["settings"].items():
            z, pre = cells[(emission, entry["labelling"])], f"K{K}_{key}_"
            n = z[pre + "n_trials"]
            keep = n >= MIN_TRIALS
            subject = z[pre + "subject"].astype(int)[keep]
            settings[key] = {**entry, "state": rank[z[pre + "state"].astype(int)[keep]],
                             "subject": subject, "cohort": cohort_of[subject], "n_trials": n[keep],
                             "coords": z[pre + "coords"].astype(np.float64)[keep]}
        mean, axes = np.asarray(run["pca_mean"]), np.asarray(run["pca_axes"])
        G = np.asarray(run["global_weights"])[np.argsort(rank)]          # ranked order
        R = refits[emission][f"K{K}_W"].astype(np.float64)                 # (mice, K, D), global order
        refit_subject = np.repeat(run["refit"]["subject"], K)
        runs.append({**run, "settings": settings, "mean": mean, "axes": axes,
                     "star_coords": coordinates(emission, run["feature_names"], G, mean, axes),
                     "refit_coords": coordinates(emission, run["feature_names"], R.reshape(-1, R.shape[2]),
                                                 mean, axes),
                     "refit_state": np.tile(rank, len(R)), "refit_subject": refit_subject})
    return frozen, runs


def coordinates(emission, names, W, mean, axes):
    """PC coordinates, then the six named weights, of weight vectors ``W``."""
    return np.concatenate([(W - mean) @ axes.T,
                           np.stack([f for _, f in named_features(emission, names, W)], 1)], 1)


def check_anchor(run, frozen):
    """The committed weights reproduce the anchor's frozen axes and coordinates.

    The freeze scored and projected the cells from these float16 weights; recomputing here
    guards against the cells file and the weights file drifting apart.
    """
    K, emission = run["num_states"], run["emission"]
    cohort_of = np.array([s["cohort"] for s in frozen["subjects"]])
    for labelling in LABELLINGS:
        key = f"{labelling}_p0_m1"
        z, pre = np.load(CELLS[(emission, labelling)]), f"K{K}_{key}_"
        W = np.load(WEIGHTS[(emission, labelling)])[f"K{K}"].astype(np.float64)
        if len(W) != len(z[pre + "n_trials"]):
            raise RuntimeError(f"{emission} K={K} {key}: weights and cells differ in length")
        coords = coordinates(emission, run["feature_names"], W, run["mean"], run["axes"])
        if not np.allclose(coords, z[pre + "coords"].astype(np.float64), rtol=2e-3, atol=2e-3 * np.abs(coords).max()):
            raise RuntimeError(f"{emission} K={K} {key}: coordinates do not follow from the weights")
        if labelling == "global":
            fit = (z[pre + "n_trials"] >= MIN_TRIALS) & (cohort_of[z[pre + "subject"].astype(int)] == "training")
            mean, axes, _ = pca_axes(W[fit], n=len(run["axes"]))
            if not (np.allclose(mean, run["mean"]) and np.allclose(axes, run["axes"])):
                raise RuntimeError(f"{emission} K={K}: PCA axes do not follow from the anchor weights")


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


def scores(X, state, subject, K, rng, plane, n_trials):
    """Cluster scores of one cohort at one K.

    ``plane``: the same cells on PC1-2 of the training cells (the plotted plane), for the
    silhouette there. ``n_trials``: each cell's trials, for the large-cell check.
    """
    out = {"n_cells": int(len(X)), "n_subjects": int(len(np.unique(subject))),
           "eta2_subject": eta_squared(X, subject)}
    if K == 1 or len(np.unique(state)) < 2:
        return {**out, "silhouette": None, "silhouette_null_mean": None,
                "silhouette_null_95": None, "silhouette_pc12": None, "silhouette_big_cells": None,
                "eta2_state": 0.0, "eta2_state_null_mean": None, "kmeans_ari": None}
    D = pairwise_distances(X)
    null_sil, null_eta = [], []
    for _ in range(N_SHUFFLES):
        shuffled = shuffle_within(state, subject, rng)
        null_sil.append(silhouette_score(D, shuffled, metric="precomputed"))
        null_eta.append(eta_squared(X, shuffled))
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(X)
    big = n_trials >= BIG_CELL
    return {**out,
            "silhouette_pc12": float(silhouette_score(plane, state)),
            "silhouette_big_cells": (float(silhouette_score(X[big], state[big]))
                                     if len(np.unique(state[big])) > 1 else None),
            "silhouette": float(silhouette_score(D, state, metric="precomputed")),
            "silhouette_null_mean": float(np.mean(null_sil)),
            "silhouette_null_95": [float(q) for q in np.percentile(null_sil, [2.5, 97.5])],
            "eta2_state": eta_squared(X, state),
            "eta2_state_null_mean": float(np.mean(null_eta)),
            "kmeans_ari": float(adjusted_rand_score(state, kmeans))}


def side_only(X, state, K):
    """Silhouette and k-means ARI on the hurdle side weights alone (left vs right, signed lags).

    A check on PLAN.md's identifiability caveat: in a state that almost never ignores, the
    engagement weights are barely constrained, so clusters seen only through them would not
    be strategy.
    """
    if K == 1:
        return {"silhouette": None, "kmeans_ari": None}
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(X)
    return {"silhouette": float(silhouette_score(X, state)),
            "kmeans_ari": float(adjusted_rand_score(state, kmeans))}


def pca_axes(X, n=4):
    """Mean and first ``n`` principal axes of ``X``, and their share of the variance."""
    mean = X.mean(0)
    _, s, vt = np.linalg.svd(X - mean, full_matrices=False)
    # Fix each axis's sign so regenerating gives the same picture.
    vt = vt * np.sign(vt[np.arange(len(vt)), np.abs(vt).argmax(1)])[:, None]
    return mean, vt[:n], (s[:n] ** 2 / (s ** 2).sum()).tolist()


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
            ax.plot(Ks, [r["silhouette_pc12"] for r in sel], "o--", color=color, ms=2, mfc="white")
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
    axes[0, 0].plot([], [], "k-", label="all weights")
    axes[0, 0].plot([], [], "k--", mfc="white", label="PC1–2 plane")
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



def named_features(emission, names, W):
    """A few readable weights per dot, as (label, column) pairs.

    hurdle: lag-1 and bias weights of the side and engagement GLMs. reference: the same, as the
    right-minus-left contrast (side) and the mean of the left and right logits against ignore
    (engagement).
    """
    col = {n: i for i, n in enumerate(names)}
    if emission == "hurdle":
        picks = [("side bias", "side:bias"), ("side RewC₁", "side:RewC_1"),
                 ("side UnrC₁", "side:UnrC_1"), ("engage bias", "engage:bias"),
                 ("engage Ignored₁", "engage:Ignored_1"), ("engage Rewarded₁", "engage:Rewarded_1")]
        return [(label, W[:, col[name]]) for label, name in picks]
    out = []
    for label, name in [("R−L bias", "bias"), ("R−L RewC₁", "RewC_1"), ("R−L UnrC₁", "UnrC_1")]:
        out.append((label, W[:, col[f"right:{name}"]] - W[:, col[f"left:{name}"]]))
    for label, name in [("mean(L,R) bias", "bias"), ("mean(L,R) Ignored₁", "Ignored_1"),
                        ("mean(L,R) Rewarded₁", "Rewarded_1")]:
        out.append((label, (W[:, col[f"right:{name}"]] + W[:, col[f"left:{name}"]]) / 2))
    return out




def fig_scatter(runs, emission):
    """Every anchor {subject, state} on PC1-2 of the training cells, one panel per K."""
    fig, axes = plt.subplots(4, 5, figsize=fs.mm(183, 160))
    by_k = {r["num_states"]: r for r in runs if r["emission"] == emission}
    for i, cohort in enumerate(COHORTS):
        for K in range(1, 11):
            ax = axes[2 * i + (K - 1) // 5, (K - 1) % 5]
            run = by_k[K]
            s = run["settings"][ANCHOR]
            m = s["cohort"] == cohort
            P = s["coords"][m]
            ax.scatter(P[:, 0], P[:, 1], s=1.2, c=np.asarray(STATE_COLORS)[s["state"][m]], lw=0,
                       alpha=0.6, rasterized=True)
            # Limits from both cohorts' dots, so one far global star cannot squash the panel.
            lo, hi = np.percentile(s["coords"][:, :2], [0.5, 99.5], axis=0)
            pad = 0.05 * (hi - lo)
            G = run["star_coords"]
            ax.scatter(G[:, 0], G[:, 1], s=18, c=STATE_COLORS[:K], marker="*", edgecolor="k",
                       lw=0.4, zorder=3, clip_on=True)
            ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
            ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
            share = run["pca_variance_share"]
            ax.set_title(f"{COHORT_LABEL[cohort]}, K = {K}", pad=2)
            ax.set_xlabel(f"PC1 ({share[0]:.0%})", labelpad=1)
            ax.set_ylabel(f"PC2 ({share[1]:.0%})", labelpad=1)
    return fig


def cut_rows(runs, emission, key, cohort):
    """One setting's score rows over K, for one emission and cohort."""
    return [dict(r["settings"][key]["scores"][cohort], K=r["num_states"])
            for r in sorted((r for r in runs if r["emission"] == emission), key=lambda r: r["num_states"])]


def fig_cuts(runs):
    """Silhouette and trials kept against K for every cut, both labellings (training mice)."""
    fig, axes = plt.subplots(2, 4, figsize=fs.mm(fs.fit_width(square_panels=4, axes=4), 85),
                             sharex=True)
    for row, emission in enumerate(EMISSIONS):
        anchor = cut_rows(runs, emission, ANCHOR, "training")
        for col, labelling in enumerate(LABELLINGS):
            for key, entry in runs[0]["settings"].items():
                if entry["labelling"] != labelling:
                    continue
                rows = cut_rows(runs, emission, key, "training")
                sel = [r for r in rows if r.get("silhouette") is not None]
                style = dict(color=CUT_COLORS[entry["posterior_min"]], ls=CUT_DASH[entry["run_min"]], lw=0.8)
                axes[row, col].plot([r["K"] for r in sel], [r["silhouette"] for r in sel], **style)
                axes[row, 2 + col].plot([r["K"] for r in rows],
                                        [r["n_trials_kept"] / a["n_trials_kept"] for r, a in zip(rows, anchor)],
                                        **style)
            axes[row, col].set_ylabel(f"{emission}\nsilhouette" if col == 0 else "silhouette")
            axes[row, col].set_title(f"{labelling} labels", pad=2)
            axes[row, 2 + col].set_title(f"{labelling} labels", pad=2)
            axes[row, 2 + col].set_ylabel("trials kept\n(share of uncut)")
            axes[row, 2 + col].set_ylim(0, 1.05)
            axes[row, col].axhline(0, color="0.6", lw=0.5)
        lo = min(min(ax.get_ylim()[0] for ax in axes[row, :2]), 0)
        hi = max(ax.get_ylim()[1] for ax in axes[row, :2])
        for ax in axes[row, :2]:
            ax.set_ylim(lo, hi)
    for p, color in CUT_COLORS.items():
        axes[0, 0].plot([], [], color=color, lw=0.8, label=f"posterior ≥ {p:g}" if p else "every trial")
    for m, dash in CUT_DASH.items():
        axes[0, 1].plot([], [], color="0.3", ls=dash, lw=0.8, label=f"run ≥ {m}")
    axes[0, 0].legend(loc="upper right")
    axes[0, 1].legend(loc="upper right")
    for ax in axes[1]:
        ax.set_xlabel("K")
        ax.set_xticks(range(2, 11, 2))
        ax.set_xlim(0.5, 10.5)
    for ax, letter in zip(axes.flat, "abcdefgh"):
        fs.panel_label(ax, letter)
    return fig


def fmt(x, spec=".2f"):
    """A table cell: the number, or a dash."""
    return "—" if x is None else format(x, spec)


def block_scores(result, frozen):
    """result-1: the anchor setting's cluster-score table and figure."""
    lines = [
        f"One dot = one {{subject, state}} GLM, fitted to the subject's trials on that state's "
        f"Viterbi path under the global fit (MAP, prior SD {frozen['data']['prior_sigma']:g}); every "
        f"trial counts (setting `{ANCHOR}`; result-3 cuts them). "
        f"Cells with fewer than {MIN_TRIALS} trials are dropped. Distances are Euclidean in the "
        "raw weight space. *Silhouette*: how much closer a dot is to its own state's dots than to "
        "the nearest other state's (1 = tight separate clusters, 0 = overlapping). *Null*: the "
        f"same with state labels shuffled among each mouse's own dots ({N_SHUFFLES} shuffles; "
        "95% range). *Silhouette, PC1–2*: the same on the plane of result-2's scatter (the first two "
        "principal components of the training cells). *Cells ≥ "
        f"{BIG_CELL} trials*: the all-weights silhouette on large cells only, a check on estimation "
        "noise. *η²*: share of the weight variance between states, or between subjects. "
        "*ARI*: agreement of k-means with K clusters with the state labels (0 = chance, 1 = "
        "identical). One global fit per K, so no error bars across fits.",
        "",
        "| emission | K | cohort | cells (dropped) | silhouette | null (95%) | silhouette, PC1–2 | silhouette, cells ≥ "
        f"{BIG_CELL} trials | η² state | η² state, null | η² subject | k-means ARI |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for emission in EMISSIONS:
        for K in range(1, 11):
            for cohort in COHORTS:
                r = next(x for x in result[emission][cohort] if x["K"] == K)
                null = (f"{r['silhouette_null_mean']:.2f} ({r['silhouette_null_95'][0]:.2f}–"
                        f"{r['silhouette_null_95'][1]:.2f})" if r["silhouette_null_95"] else "—")
                lines.append(
                    f"| {emission} | {K} | {cohort} | {r['n_cells']} ({r['n_dropped']}) | "
                    f"{fmt(r['silhouette'])} | {null} | {fmt(r['silhouette_pc12'])} | "
                    f"{fmt(r['silhouette_big_cells'])} | {r['eta2_state']:.2f} | "
                    f"{fmt(r['eta2_state_null_mean'])} | {r['eta2_subject']:.2f} | "
                    f"{fmt(r['kmeans_ari'])} |")
    lines += ["", "**Check: hurdle, side weights only** (the 21 left-vs-right weights, which every "
              "state constrains; engagement weights left out).", "",
              "| K | held-out silhouette | held-out k-means ARI | training silhouette | training k-means ARI |",
              "|---|---|---|---|---|"]
    for K in range(2, 11):
        h, t = (next(x for x in result["hurdle"][c] if x["K"] == K)["side_only"] for c in COHORTS)
        lines.append(f"| {K} | {fmt(h['silhouette'])} | {fmt(h['kmeans_ari'])} | "
                     f"{fmt(t['silhouette'])} | {fmt(t['kmeans_ari'])} |")
    lines += ["", "![cluster scores](../fig_r3_scores.svg)"]
    return "\n".join(lines)


def block_scatter():
    """result-2: the anchor setting's weight-space scatter figures."""
    lines = [
        "Each panel: every {subject, state} GLM of one cohort at one K (every trial, global "
        "Viterbi), on the first two principal components of that K's training-mouse dots (both "
        "cohorts share the axes). Colour = state, ranked by trials (state 1, blue, is the most "
        "used); colours match only within a K, because states are not matched across K. Stars: "
        "the global fit's own state weights (fitted with a weak prior, SD 100); axes span 99% of "
        "the dots, so a far star is not shown.",
        "",
    ]
    for emission in EMISSIONS:
        lines += [f"**{emission}**", "", f"![{emission} weight space](../fig_r3_scatter_{emission}.svg)", ""]
    return "\n".join(lines)


def block_cuts(runs, frozen):
    """result-3: the cuts and the refit labelling against the anchor."""
    cuts = frozen["data"]["cuts"]
    lines = [
        "The same cells, cut and labelled 18 ways. *Labels*: the Viterbi path and smoothed "
        "posteriors under the global fit (`global`, as in result-1) or under each mouse's own "
        "round-2 refit (`refit`: r2's pick at that K, centred on the global fit). *Cuts*: a trial "
        "counts for its Viterbi state only if that state's smoothed posterior is at least p "
        f"(p ∈ {{{', '.join(f'{p:g}' for p in cuts['posterior_min'])}}}) and it lies in a same-state "
        f"run of at least m trials within its session (m ∈ {{{', '.join(map(str, cuts['run_min']))}}}). "
        "Each cell is refitted to its kept trials; states keep result-1's ranks and the plotted "
        "axes stay the uncut global setting's. Cutting keeps the trials that best fit a state's "
        "weights, so a higher silhouette after a cut is partly selection, not evidence.",
        "",
        "Silhouette (all weights, training mice) by K, and the trials kept as a share of the uncut "
        "setting's:",
        "",
    ]
    Ks = list(range(2, 11))
    for emission in EMISSIONS:
        lines += [f"**{emission}**", "",
                  "| setting | " + " | ".join(f"K = {K}" for K in Ks) + " | trials kept, K = 2 | trials kept, K = 10 |",
                  "|---" * (len(Ks) + 3) + "|"]
        anchor = {r["K"]: r for r in cut_rows(runs, emission, ANCHOR, "training")}
        for key in runs[0]["settings"]:
            rows = {r["K"]: r for r in cut_rows(runs, emission, key, "training")}
            kept = [rows[K]["n_trials_kept"] / anchor[K]["n_trials_kept"] for K in (2, 10)]
            lines.append(f"| `{key}` | " + " | ".join(fmt(rows[K].get("silhouette")) for K in Ks)
                         + f" | {kept[0]:.0%} | {kept[1]:.0%} |")
        lines.append("")
    lines += ["![cuts](../fig_r3_cuts.svg)"]
    return "\n".join(lines)


def block_refits(runs):
    """result-4: the round-2 refits behind the refit labelling."""
    lines = [
        "Each mouse's round-2 refit, recomputed by the freeze (EM from the global fit on the "
        "mouse's train sessions, r2's picked prior at that K). *Held-out check*: largest "
        "difference from the refits round 2 saved for the 149 held-out mice (weights and "
        "transitions; training-mouse refits were not saved). *Not aligned*: refits whose states "
        "match the global states other than one to one in order (least-squares weight matching). "
        "*Second step*: the largest change of a second M-step from any {subject, state} fit, a "
        "convergence check.",
        "",
        "| emission | K | σ | c | round-2 run | refits | held-out check | not aligned | not converged | second step |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(runs, key=lambda r: (r["emission"], r["num_states"])):
        r2, ref = r["round2"], r["refit"]
        step = max(s["max_second_step_change"] for s in r["settings"].values())
        c = r2["transition_strength_subject"]
        lines.append(
            f"| {r['emission']} | {r['num_states']} | {r2['prior_sigma_subject']:g} | "
            f"{'off' if c is None else format(c, 'g')} | {r2['wandb_run_id']} | {len(ref['subject'])} | "
            f"{r2['heldout_refits_checked']} mice, ≤ {r2['heldout_refit_max_abs_diff']:.0e} | "
            f"{sum(not x for x in ref['matched_identity'])} | {sum(not x for x in ref['converged'])} | "
            f"{step:.0e} |")
    return "\n".join(lines)


# -- interactive page ---------------------------------------------------------------------

def pack(values, dtype):
    """Base64 of a numeric array as ``dtype`` (little-endian)."""
    return base64.b64encode(np.ascontiguousarray(values, dtype=dtype).tobytes()).decode()


def pack_cols(X):
    """Columns of ``X`` quantised to uint16 between their min and max: (base64, lo, step) each."""
    out = []
    for col in X.T:
        lo, hi = (float(col.min()), float(col.max())) if len(col) else (0.0, 0.0)
        step = (hi - lo) / 65535 or 1.0
        out.append([pack(np.round((col - lo) / step), "<u2"), round(lo, 6), step])
    return out


def pack_cells(state, subject, n, coords):
    return {"count": int(len(state)), "state": pack(state, "u1"), "subject": pack(subject, "<u2"),
            "n": pack(np.minimum(n, 65535), "<u2"), "cols": pack_cols(coords)}


def interactive_files(runs):
    """One data file per (emission, labelling, K), plus the refit dots per emission.

    Each is a script that hands its data to the page (``R3.load``), so the page also works when
    opened as a local file. Coordinates are quantised to 1/65535 of each column's range.
    """
    files = {}
    for emission in EMISSIONS:
        by_k = {r["num_states"]: r for r in runs if r["emission"] == emission}
        for labelling in LABELLINGS:
            for K, run in by_k.items():
                files[f"{emission}_{labelling}_K{K}"] = {
                    key: pack_cells(s["state"], s["subject"], s["n_trials"], s["coords"])
                    for key, s in run["settings"].items() if s["labelling"] == labelling}
        files[f"{emission}_refits"] = {
            str(K): pack_cells(run["refit_state"], run["refit_subject"],
                               np.zeros(len(run["refit_state"]), int), run["refit_coords"])
            for K, run in by_k.items()}
    return files


def interactive_page(frozen, runs):
    """The page's own data: settings, stars, scores; the cells load from the data files."""
    data = {"subjects": [s["subject_id"] for s in frozen["subjects"]],
            "cohort": [int(s["cohort"] == "heldout") for s in frozen["subjects"]],
            "stateColors": STATE_COLORS, "minTrials": MIN_TRIALS, "anchor": ANCHOR,
            "cuts": frozen["data"]["cuts"], "emissions": {}}
    keep = ("n_cells", "n_trials_kept", "silhouette", "silhouette_null_95", "silhouette_pc12",
            "eta2_state", "eta2_subject", "eta2_state_null_mean", "kmeans_ari")
    for emission in EMISSIONS:
        by_k = {r["num_states"]: r for r in runs if r["emission"] == emission}
        names = by_k[1]["feature_names"]
        features = [label for label, _ in named_features(emission, names, np.zeros((1, len(names))))]
        data["emissions"][emission] = {
            "features": features,
            "K": {str(K): {"run": r["wandb_run_id"], "pcShare": [round(x, 4) for x in r["pca_variance_share"]],
                           "stars": np.round(r["star_coords"], 4).T.tolist(),
                           "round2": {k: r["round2"][k] for k in ("wandb_run_id", "prior_sigma_subject",
                                                                  "transition_strength_subject")}}
                  for K, r in by_k.items()},
            "scores": {key: {c: [{k: x.get(k) for k in keep} for x in cut_rows(runs, emission, key, c)]
                             for c in COHORTS} for key in by_k[1]["settings"]},
        }
    return data


def write_interactive(frozen, runs):
    """The interactive page and its data files; returns their paths."""
    folder = INTERACTIVE.parent / "r3_data"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, content in interactive_files(runs).items():
        path = folder / f"{name}.js"
        path.write_text(f'R3.load("{name}", {json.dumps(content, separators=(",", ":"))});\n')
        paths.append(path)
    html = INTERACTIVE_TEMPLATE.replace("__PLOTLY__", PLOTLY).replace(
        "__DATA__", json.dumps(interactive_page(frozen, runs), separators=(",", ":"), ensure_ascii=False))
    INTERACTIVE.write_text(html)
    return [INTERACTIVE] + paths


INTERACTIVE_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>State spread explorer</title>
<script src="__PLOTLY__"></script>
<style>
:root { --bg:#ffffff; --fg:#1d1d1f; --muted:#6e6e73; --line:#d2d2d7; --panel:#f5f5f7; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#141416; --fg:#f2f2f4; --muted:#a1a1a6; --line:#3a3a3c; --panel:#1f1f22; color-scheme:dark; } }
:root[data-theme="dark"] { --bg:#141416; --fg:#f2f2f4; --muted:#a1a1a6; --line:#3a3a3c; --panel:#1f1f22; color-scheme:dark; }
body { background:var(--bg); color:var(--fg); margin:0; font:14px/1.45 -apple-system,"Segoe UI",Arial,sans-serif; }
main { max-width:1100px; margin:0 auto; padding:16px; }
h1 { font-size:20px; margin:4px 0 2px; } h2 { font-size:16px; margin:22px 0 4px; }
p { margin:4px 0 10px; color:var(--muted); max-width:820px; }
.controls { display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; padding:10px 12px;
  background:var(--panel); border:1px solid var(--line); border-radius:8px; margin-bottom:6px; }
label { font-size:13px; color:var(--muted); } select { font-size:13px; margin-left:4px; }
#scatter { height:560px; } #scores { height:300px; }
.note { font-size:12px; }
</style></head>
<body><main>
<h1>How each GLM-HMM state spreads across subjects</h1>
<p>Study 14 r3. One dot = one {subject, state} GLM, fitted to the subject's kept trials of that
state (prior SD 2). <b>Labels</b>: the Viterbi path under the global fit, or under each mouse's
own round-2 refit (r2's pick at that K). <b>Cuts</b>: keep a trial only if its state's smoothed
posterior and its same-state run length pass the thresholds. Cells with fewer than
<span id="minTrials"></span> kept trials are left out. Stars are the global fit's states; states
keep the uncut global ranks and colours in every setting, and the axes are the uncut global
setting's. <b>Dots</b> can add each held-out or training mouse's refit itself (K open circles per
mouse, pulled toward the stars by the refit's prior). Colours match only within one K; click a
legend entry to hide it.</p>
<div class="controls">
 <label>Emission<select id="emission"></select></label>
 <label>K<select id="K"></select></label>
 <label>Labels<select id="labelling"><option value="global">global fit</option><option value="refit">mouse's refit</option></select></label>
 <label>Posterior ≥<select id="p"></select></label>
 <label>Run ≥<select id="m"></select></label>
 <label>Mice<select id="cohort"><option value="1">held-out</option><option value="0">training</option><option value="both">both</option></select></label>
 <label>Dots<select id="dots"><option value="cells">cell GLMs</option><option value="refit">refits</option><option value="both">both</option></select></label>
</div>
<div class="controls">
 <label>x<select id="x"></select></label>
 <label>y<select id="y"></select></label>
 <label><input type="checkbox" id="threeD"> 3D</label>
 <label id="zLabel" hidden>z<select id="z"></select></label>
</div>
<div id="scatter"></div>
<p class="note" id="runNote"></p>
<h2>Cluster scores against K</h2>
<p id="scoresNote">x axis: the number of states K, for the setting chosen above (grey: the uncut
global setting, for reference). Silhouette: 1 = tight separate clusters, 0 = overlap; solid = all
weights, dashed = the PC1–2 plane; the band is the null with state labels shuffled among each
mouse's own dots (95% range). η²: share of the weight variance between states (solid) or subjects
(dashed); dotted = by state after the shuffle. ARI: k-means with K clusters against the states.
A cut keeps the trials that best fit each state, so part of any rise after a cut is selection.</p>
<div id="scores"></div>
</main>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const COH = {heldout: "#0072B2", training: "#D55E00"};
const R3 = {files: {}, waiting: {}};
R3.load = (name, data) => { R3.files[name] = data; (R3.waiting[name] || []).forEach(f => f()); delete R3.waiting[name]; };
function need(name, then) {
  if (R3.files[name]) return then();
  if (!R3.waiting[name]) { R3.waiting[name] = []; const s = document.createElement("script");
    s.src = "r3_data/" + name + ".js"; document.head.appendChild(s); }
  R3.waiting[name].push(then);
}
function bytes(b64) { const s = atob(b64), a = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i); return a.buffer; }
const decoded = new WeakMap();
function cells(packed) {
  if (decoded.has(packed)) return decoded.get(packed);
  const out = {count: packed.count, state: new Uint8Array(bytes(packed.state)),
    subject: new Uint16Array(bytes(packed.subject)), n: new Uint16Array(bytes(packed.n)),
    cols: packed.cols.map(([b, lo, step]) => Array.from(new Uint16Array(bytes(b)), q => lo + q * step))};
  decoded.set(packed, out); return out;
}
const key = () => `${$("labelling").value}_p${Math.round(100 * +$("p").value)}_m${$("m").value}`;
$("minTrials").textContent = D.minTrials;
for (const e of Object.keys(D.emissions)) $("emission").add(new Option(e, e));
for (let k = 1; k <= 10; k++) $("K").add(new Option(k, k));
D.cuts.posterior_min.forEach(p => $("p").add(new Option(p ? p : "any", p)));
D.cuts.run_min.forEach(m => $("m").add(new Option(m, m)));
$("K").value = 3;

function axisLabels(e, k) {
  const c = D.emissions[e].K[k];
  return c.pcShare.map((s, i) => `PC${i + 1} (${Math.round(100 * s)}%)`).concat(D.emissions[e].features);
}
function fillAxes() {
  const e = $("emission").value, k = $("K").value, labels = axisLabels(e, k);
  const xi = $("x").selectedIndex, yi = $("y").selectedIndex, zi = $("z").selectedIndex;
  for (const id of ["x", "y", "z"]) { const s = $(id); s.innerHTML = ""; labels.forEach((l, i) => s.add(new Option(l, i))); }
  $("x").selectedIndex = xi >= 0 ? xi : 0; $("y").selectedIndex = yi >= 0 ? yi : 1;
  $("z").selectedIndex = zi >= 0 ? zi : 2;
}
function layoutBase(extra) {
  return Object.assign({paper_bgcolor: css("--bg"), plot_bgcolor: css("--bg"),
    font: {color: css("--fg"), size: 12}, margin: {l: 56, r: 10, t: 10, b: 46}}, extra);
}
function draw() {
  const e = $("emission").value, dots = $("dots").value;
  need(`${e}_${$("labelling").value}_K${$("K").value}`, () => dots === "cells" ? drawScatter() : need(`${e}_refits`, drawScatter));
}
function drawScatter() {
  const e = $("emission").value, k = $("K").value, setting = key(), dots = $("dots").value;
  const C = cells(R3.files[`${e}_${$("labelling").value}_K${k}`][setting]);
  const R = dots === "cells" ? null : cells(R3.files[`${e}_refits`][k]);
  const xi = +$("x").value, yi = +$("y").value, zi = +$("z").value, which = $("cohort").value;
  const labels = axisLabels(e, k), three = $("threeD").checked, info = D.emissions[e].K[k];
  $("zLabel").hidden = !three;
  const shown = i => which === "both" || D.cohort[i] === +which;
  const traces = [], xs = [], ys = [], zs = [];
  const add = (src, s, name, marker, hover) => {
    const idx = []; for (let i = 0; i < src.count; i++) if (src.state[i] === s && shown(src.subject[i])) idx.push(i);
    idx.forEach(i => { xs.push(src.cols[xi][i]); ys.push(src.cols[yi][i]); zs.push(src.cols[zi][i]); });
    traces.push({type: three ? "scatter3d" : "scattergl", mode: "markers", name: `${name} ${s + 1} (${idx.length})`,
      x: idx.map(i => src.cols[xi][i]), y: idx.map(i => src.cols[yi][i]), z: three ? idx.map(i => src.cols[zi][i]) : undefined,
      marker, text: idx.map(hover), hovertemplate: "%{text}<extra></extra>"});
  };
  for (let s = 0; s < +k; s++) {
    if (dots !== "refit") add(C, s, "state", {color: D.stateColors[s], size: three ? 2.5 : 5, opacity: 0.55},
      i => `subject ${D.subjects[C.subject[i]]} · ${D.cohort[C.subject[i]] ? "held-out" : "training"}<br>state ${s + 1} · ${C.n[i]} trials kept`);
    if (R) add(R, s, "refit", {color: dots === "both" ? "rgba(0,0,0,0)" : D.stateColors[s], size: three ? 3 : 7, opacity: 0.8,
      line: {color: D.stateColors[s], width: 1.2}},
      i => `subject ${D.subjects[R.subject[i]]} · ${D.cohort[R.subject[i]] ? "held-out" : "training"}<br>round-2 refit, state ${s + 1}`);
  }
  traces.push({type: three ? "scatter3d" : "scatter", mode: "markers", name: "global fit",
    x: info.stars[xi], y: info.stars[yi], z: three ? info.stars[zi] : undefined,
    marker: {symbol: three ? "diamond" : "star", size: three ? 7 : 15, color: D.stateColors.slice(0, +k),
             line: {color: css("--fg"), width: 1}},
    text: info.stars[xi].map((_, s) => `global state ${s + 1}`), hovertemplate: "%{text}<extra></extra>"});
  const range = a => { if (!a.length) return undefined; const v = [...a].sort((p, q) => p - q), lo = v[Math.floor(0.005 * (v.length - 1))],
    hi = v[Math.ceil(0.995 * (v.length - 1))], pad = 0.06 * (hi - lo || 1); return [lo - pad, hi + pad]; };
  const axis = (label, values) => ({title: {text: label}, range: range(values), zeroline: false,
    gridcolor: css("--line"), backgroundcolor: css("--bg"), color: css("--fg")});
  const layout = three
    ? {scene: {xaxis: axis(labels[xi], xs), yaxis: axis(labels[yi], ys), zaxis: axis(labels[zi], zs), aspectmode: "cube"},
       legend: {orientation: "v", x: 1.01, y: 0.92}}
    : {xaxis: axis(labels[xi], xs), yaxis: axis(labels[yi], ys), legend: {orientation: "v", x: 1.01, y: 0.92}, hovermode: "closest"};
  Plotly.react("scatter", traces, layoutBase(layout), {responsive: true});
  const S = D.emissions[e].scores, cohorts = which === "both" ? ["heldout", "training"] : [which === "1" ? "heldout" : "training"];
  const kept = cohorts.map(c => { const a = S[D.anchor][c][k - 1].n_trials_kept, b = S[setting][c][k - 1].n_trials_kept;
    return `${c === "heldout" ? "held-out" : "training"} ${Math.round(100 * b / a)}%`; }).join(", ");
  const r2 = info.round2;
  $("runNote").textContent = `Setting ${setting}: trials kept vs the uncut global setting, ${kept}. Global fit: W&B run ${info.run} (mice_glmhmm); ` +
    `refits: r2's pick at K = ${k}, σ = ${r2.prior_sigma_subject}, c = ${r2.transition_strength_subject ?? "off"} (run ${r2.wandb_run_id}). ` +
    "Axis ranges cover 99% of the dots shown; " + (three ? "drag to rotate, scroll to zoom." : "double-click to reset zoom.");
}
function drawScores() {
  const e = $("emission").value, S = D.emissions[e].scores, setting = key(), traces = [];
  const series = (rows, col, dash, width, name, legend, group) => [
    {x: rows.map(r => r.K), y: rows.map(r => r.silhouette), line: {color: col, dash, width}, name, showlegend: legend, legendgroup: group, xaxis: "x", yaxis: "y"},
    {x: rows.map(r => r.K), y: rows.map(r => r.eta2_state), line: {color: col, dash, width}, name: name + " · η² state", showlegend: false, legendgroup: group, xaxis: "x2", yaxis: "y2"},
    {x: rows.map(r => r.K), y: rows.map(r => r.kmeans_ari), line: {color: col, dash, width}, name: name + " · ARI", showlegend: false, legendgroup: group, xaxis: "x3", yaxis: "y3"}];
  for (const coh of ["heldout", "training"]) {
    const ok = r => r.silhouette !== null && r.silhouette !== undefined;
    const rows = S[setting][coh].map((r, i) => Object.assign({K: i + 1}, r)).filter(ok), col = COH[coh];
    const base = S[D.anchor][coh].map((r, i) => Object.assign({K: i + 1}, r)).filter(ok);
    const name = coh === "heldout" ? "held-out" : "training";
    if (setting !== D.anchor) traces.push(...series(base, "#9a9aa0", "solid", 1, name + " · uncut global", false, coh));
    traces.push({x: rows.map(r => r.K).concat(rows.map(r => r.K).reverse()),
      y: rows.map(r => r.silhouette_null_95[1]).concat(rows.map(r => r.silhouette_null_95[0]).reverse()),
      fill: "toself", fillcolor: col + "33", mode: "lines", line: {width: 0}, hoverinfo: "skip", showlegend: false, xaxis: "x", yaxis: "y"});
    traces.push(...series(rows, col, "solid", 2, name, true, coh));
    traces.push({x: rows.map(r => r.K), y: rows.map(r => r.silhouette_pc12), name: name + " · PC1–2", legendgroup: coh, showlegend: false, line: {color: col, dash: "dash"}, xaxis: "x", yaxis: "y"});
    traces.push({x: rows.map(r => r.K), y: rows.map(r => r.eta2_subject), name: name + " · η² subject", legendgroup: coh, showlegend: false, line: {color: col, dash: "dash"}, xaxis: "x2", yaxis: "y2"});
    traces.push({x: rows.map(r => r.K), y: rows.map(r => r.eta2_state_null_mean), name: name + " · η² state, shuffled", legendgroup: coh, showlegend: false, line: {color: col, dash: "dot", width: 1}, xaxis: "x2", yaxis: "y2"});
  }
  traces.forEach(t => { t.type = "scatter"; if (!t.fill) { t.mode = "lines+markers"; t.marker = {size: 4}; } });
  const ax = (dom, title, range) => Object.assign({title: {text: title}, gridcolor: css("--line"), zeroline: false},
    dom ? {domain: dom} : {}, range ? {range} : {});
  Plotly.react("scores", traces, layoutBase({
    legend: {orientation: "h", y: 1.14},
    xaxis: Object.assign(ax([0, 0.28], "K"), {anchor: "y"}), xaxis2: Object.assign(ax([0.36, 0.64], "K"), {anchor: "y2"}),
    xaxis3: Object.assign(ax([0.72, 1], "K"), {anchor: "y3"}),
    yaxis: ax(null, "silhouette"), yaxis2: Object.assign(ax(null, "η²", [0, 1]), {anchor: "x2"}),
    yaxis3: Object.assign(ax(null, "k-means ARI", [0, 1]), {anchor: "x3"}), margin: {l: 56, r: 10, t: 30, b: 50}}),
    {responsive: true});
}
$("emission").onchange = () => { fillAxes(); draw(); drawScores(); };
$("K").onchange = () => { fillAxes(); draw(); };
for (const id of ["labelling", "p", "m"]) $(id).onchange = () => { draw(); drawScores(); };
for (const id of ["cohort", "dots"]) $(id).onchange = draw;
for (const id of ["x", "y", "z", "threeD"]) $(id).onchange = draw;
fillAxes(); draw(); drawScores();
</script></body></html>
"""


def replace_block(text, n, body):
    """Replace the content between the result-n markers."""
    pattern = re.compile(rf"(<!-- BEGIN result-{n} -->\n).*?(\n<!-- END result-{n} -->)", re.S)
    if not pattern.search(text):
        raise ValueError(f"result-{n} markers missing")
    return pattern.sub(lambda m: m.group(1) + body.strip("\n") + m.group(2), text)


def main() -> None:
    frozen, runs = load()
    for run in runs:
        check_anchor(run, frozen)
    result = {e: {c: cut_rows(runs, e, ANCHOR, c) for c in COHORTS} for e in EMISSIONS}

    fs.apply_paper_style()
    figs = [fs.save_figure(fig_scores(result), HERE / "fig_r3_scores.svg")]
    for emission in EMISSIONS:
        figs.append(fs.save_figure(fig_scatter(runs, emission), HERE / f"fig_r3_scatter_{emission}.svg"))
    figs.append(fs.save_figure(fig_cuts(runs), HERE / "fig_r3_cuts.svg"))
    plt.close("all")
    pages = write_interactive(frozen, runs)

    text = REPORT.read_text()
    text = replace_block(text, 1, block_scores(result, frozen))
    text = replace_block(text, 2, block_scatter())
    text = replace_block(text, 3, block_cuts(runs, frozen))
    text = replace_block(text, 4, block_refits(runs))
    REPORT.write_text(text)

    out = {"min_trials": MIN_TRIALS, "n_shuffles": N_SHUFFLES, "anchor": ANCHOR,
           "scores": {e: {key: {c: cut_rows(runs, e, key, c) for c in COHORTS} for key in runs[0]["settings"]}
                      for e in EMISSIONS},
           "pca_variance_share": {e: {str(r["num_states"]): r["pca_variance_share"] for r in runs if r["emission"] == e}
                                  for e in EMISSIONS},
           "refit_checks": [{"emission": r["emission"], "K": r["num_states"], **{k: r["round2"][k] for k in (
               "wandb_run_id", "prior_sigma_subject", "transition_strength_subject", "heldout_refits_checked",
               "heldout_refit_max_abs_diff")}, "n_refits": len(r["refit"]["subject"]),
               "n_not_aligned": sum(not x for x in r["refit"]["matched_identity"]),
               "n_not_converged": sum(not x for x in r["refit"]["converged"])} for r in runs]}
    meta = build_meta("analysis/r3_state_spread.py", WANDB_GROUPS, study_root=HERE.parent,
                      code=[__file__], inputs=[FROZEN, *CELLS.values(), *WEIGHTS.values(), *REFITS.values()])
    write_artifact(HERE / "r3_state_spread.json", {"_meta": meta, **out},
                   outputs=figs + pages + [f"{REPORT}#result-{n}" for n in (1, 2, 3, 4)])


if __name__ == "__main__":
    main()
