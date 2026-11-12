"""Cluster scores for r3 (state spread), shared by the freeze and the producer.

``freeze_r3.py`` scores every setting with these functions; ``r3_state_spread.py`` uses the
same PCA and named weights to check and plot the frozen cells. Every distance is Euclidean in
the raw weight space (logit units, bias included).
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, pairwise_distances, silhouette_score

COHORTS = ("heldout", "training")
MIN_TRIALS = 200        # a {subject, state} cell with fewer kept trials is dropped
N_SHUFFLES = 100        # within-subject label shuffles for the silhouette null
BIG_CELL = 1000         # check: the silhouette on cells with at least this many trials
SEED = 0


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
