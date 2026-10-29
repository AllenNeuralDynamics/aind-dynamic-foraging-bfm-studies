"""Producer for r2: how each GLM-HMM state spreads across subjects (offline).

Reads ``frozen_r2.json`` and ``frozen_r2_weights_<emission>.npz`` (one GLM per {subject,
state} under each K-grid run's global fit; see ``freeze_r2.py``) and writes
``r2_state_spread.json``, the figures ``fig_r2_*.svg`` and the result blocks of
``reports/r2-state-spread.md``:
- result-1: cluster scores against K, per emission and cohort;
- result-2: every {subject, state} in weight space, one panel per K;
and ``interactive/r2_state_spread.html``, the same data to explore (any K, cohort and axes).

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
INTERACTIVE = HERE / "interactive" / "r2_state_spread.html"
PLOTLY = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.34.0/plotly.min.js"
MIN_TRIALS = 200        # a {subject, state} cell with fewer Viterbi trials is dropped
N_SHUFFLES = 100        # within-subject label shuffles for the silhouette null
BIG_CELL = 1000         # check: the silhouette on cells with at least this many trials
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


def interactive_data(frozen, runs, pcs, result):
    """Everything the interactive page draws, rounded to 3 decimals."""
    def r3(a):
        return np.round(np.asarray(a, float), 3).tolist()

    data = {"subjects": [s["subject_id"] for s in frozen["subjects"]],
            "stateColors": STATE_COLORS, "minTrials": MIN_TRIALS, "emissions": {}}
    for emission in EMISSIONS:
        by_k = {}
        for run in sorted((r for r in runs if r["emission"] == emission), key=lambda r: r["num_states"]):
            K = run["num_states"]
            mean, axes_pc, share = pcs[emission][K]
            feats = named_features(emission, run["feature_names"], run["W"])
            gfeats = named_features(emission, run["feature_names"], run["G"])
            P, PG = (run["W"] - mean) @ axes_pc.T, (run["G"] - mean) @ axes_pc.T
            by_k[str(K)] = {
                "run": run["wandb_run_id"],
                "pcShare": r3(share),
                "cohort": [int(c == "heldout") for c in run["cohort"]],
                "state": run["state"].tolist(), "subject": run["subject"].tolist(),
                "n": run["n_trials"].tolist(),
                "cols": [r3(P[:, i]) for i in range(P.shape[1])] + [r3(f) for _, f in feats],
                "global": [r3(PG[:, i]) for i in range(PG.shape[1])] + [r3(f) for _, f in gfeats],
            }
        names = run["feature_names"]
        data["emissions"][emission] = {
            "features": [label for label, _ in named_features(emission, names, np.zeros((1, len(names))))],
            "K": by_k,
            "scores": {c: [{k: v for k, v in row.items() if k != "wandb_run_id"} for row in result[emission][c]]
                       for c in COHORTS},
        }
    return data


def write_interactive(data):
    """The interactive page: weight-space explorer and cluster scores vs K."""
    html = INTERACTIVE_TEMPLATE.replace("__PLOTLY__", PLOTLY).replace(
        "__DATA__", json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    INTERACTIVE.parent.mkdir(exist_ok=True)
    INTERACTIVE.write_text(html)
    return INTERACTIVE


INTERACTIVE_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>State spread explorer</title>
<script src="__PLOTLY__"></script>
<style>
:root { --bg:#ffffff; --fg:#1d1d1f; --muted:#6e6e73; --line:#d2d2d7; --panel:#f5f5f7; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#141416; --fg:#f2f2f4; --muted:#a1a1a6; --line:#3a3a3c; --panel:#1f1f22; } }
:root[data-theme="dark"] { --bg:#141416; --fg:#f2f2f4; --muted:#a1a1a6; --line:#3a3a3c; --panel:#1f1f22; }
body { background:var(--bg); color:var(--fg); margin:0; font:14px/1.45 -apple-system,"Segoe UI",Arial,sans-serif; }
main { max-width:1100px; margin:0 auto; padding:16px; }
h1 { font-size:20px; margin:4px 0 2px; } h2 { font-size:16px; margin:22px 0 4px; }
p { margin:4px 0 10px; color:var(--muted); max-width:820px; }
.controls { display:flex; flex-wrap:wrap; gap:10px 18px; align-items:center; padding:10px 12px;
  background:var(--panel); border:1px solid var(--line); border-radius:8px; }
label { font-size:13px; color:var(--muted); } select { font-size:13px; margin-left:4px; }
#scatter { height:560px; } #scores { height:300px; }
.note { font-size:12px; }
</style></head>
<body><main>
<h1>How each GLM-HMM state spreads across subjects</h1>
<p>Study 14 r2. One dot = one {subject, state} GLM, fitted to the subject's trials on that
state's Viterbi path under the global fit (prior SD 2). Cells with fewer than
<span id="minTrials"></span> trials are left out. Stars are the global fit's own states.
Colours match only within one K. Click a legend entry to hide a state.</p>
<div class="controls">
 <label>Emission<select id="emission"></select></label>
 <label>K<select id="K"></select></label>
 <label>Mice<select id="cohort"><option value="1">held-out</option><option value="0">training</option><option value="both">both</option></select></label>
 <label>x<select id="x"></select></label>
 <label>y<select id="y"></select></label>
 <label><input type="checkbox" id="threeD"> 3D</label>
 <label id="zLabel" hidden>z<select id="z"></select></label>
</div>
<div id="scatter"></div>
<p class="note" id="runNote"></p>
<h2>Cluster scores against K</h2>
<p>x axis: the number of states K. Silhouette: 1 = tight separate clusters, 0 = overlap; solid = all weights, dashed = the PC1–2
plane; the band is the null with state labels shuffled among each mouse's own dots (95% range). η²: share of the weight variance between
states (solid) or subjects (dashed); dotted = by state after the shuffle. ARI: k-means with K
clusters against the states.</p>
<div id="scores"></div>
</main>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const COH = {heldout: "#0072B2", training: "#D55E00"};
$("minTrials").textContent = D.minTrials;
for (const e of Object.keys(D.emissions)) $("emission").add(new Option(e, e));
for (let k = 1; k <= 10; k++) $("K").add(new Option(k, k));
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
function drawScatter() {
  const e = $("emission").value, k = $("K").value, c = D.emissions[e].K[k];
  const xi = +$("x").value, yi = +$("y").value, zi = +$("z").value, which = $("cohort").value;
  const labels = axisLabels(e, k), three = $("threeD").checked;
  $("zLabel").hidden = !three;
  const traces = [];
  for (let s = 0; s < +k; s++) {
    const idx = c.state.map((v, i) => i).filter(i => c.state[i] === s &&
      (which === "both" || c.cohort[i] === +which));
    traces.push({type: three ? "scatter3d" : "scattergl", mode: "markers", name: `state ${s + 1} (${idx.length})`,
      x: idx.map(i => c.cols[xi][i]), y: idx.map(i => c.cols[yi][i]),
      z: three ? idx.map(i => c.cols[zi][i]) : undefined,
      marker: {color: D.stateColors[s], size: three ? 2.5 : 5, opacity: 0.55,
               symbol: idx.map(i => c.cohort[i] ? "circle" : "diamond")},
      text: idx.map(i => `subject ${D.subjects[c.subject[i]]} · ${c.cohort[i] ? "held-out" : "training"}<br>state ${s + 1} · ${c.n[i]} trials`),
      hovertemplate: "%{text}<extra></extra>"});
  }
  traces.push({type: three ? "scatter3d" : "scatter", mode: "markers", name: "global fit",
    x: c.global[xi], y: c.global[yi], z: three ? c.global[zi] : undefined,
    marker: {symbol: three ? "diamond" : "star", size: three ? 7 : 15, color: D.stateColors.slice(0, +k),
             line: {color: css("--fg"), width: 1}},
    text: c.global[xi].map((_, s) => `global state ${s + 1}`), hovertemplate: "%{text}<extra></extra>"});
  const xs = c.cols[xi], ys = c.cols[yi];
  const range = a => { const v = [...a].sort((p, q) => p - q), lo = v[Math.floor(0.005 * (v.length - 1))],
    hi = v[Math.ceil(0.995 * (v.length - 1))], pad = 0.06 * (hi - lo || 1); return [lo - pad, hi + pad]; };
  const axis = (label, values) => ({title: {text: label}, range: range(values), zeroline: false,
    gridcolor: css("--line"), backgroundcolor: css("--bg"), color: css("--fg")});
  const layout = three
    ? {scene: {xaxis: axis(labels[xi], xs), yaxis: axis(labels[yi], ys), zaxis: axis(labels[zi], c.cols[zi]),
               aspectmode: "cube"}, legend: {orientation: "v", x: 1.01, y: 0.92}}
    : {xaxis: axis(labels[xi], xs), yaxis: axis(labels[yi], ys),
       legend: {orientation: "v", x: 1.01, y: 0.92}, hovermode: "closest"};
  Plotly.react("scatter", traces, layoutBase(layout), {responsive: true});
  $("runNote").textContent = `W&B run ${c.run} (mice_glmhmm). Axis ranges cover 99% of the dots; ` +
    (three ? "drag to rotate, scroll to zoom; global states are outlined diamonds. " : "double-click to reset zoom. ") +
    (which === "both" ? "Held-out mice: circles; training mice: diamonds." : "");
}
function drawScores() {
  const e = $("emission").value, S = D.emissions[e].scores, traces = [];
  for (const [coh, rows0] of Object.entries(S)) {
    const rows = rows0.filter(r => r.silhouette !== null), K = rows.map(r => r.K), col = COH[coh];
    const name = coh === "heldout" ? "held-out" : "training";
    traces.push({x: K.concat([...K].reverse()), y: rows.map(r => r.silhouette_null_95[1]).concat(rows.map(r => r.silhouette_null_95[0]).reverse()),
      fill: "toself", fillcolor: col + "33", mode: "lines", line: {width: 0}, hoverinfo: "skip", showlegend: false, xaxis: "x", yaxis: "y"});
    traces.push({x: K, y: rows.map(r => r.silhouette), name, legendgroup: coh, line: {color: col}, xaxis: "x", yaxis: "y"});
    traces.push({x: K, y: rows.map(r => r.silhouette_pc12), name: name + " · PC1–2", legendgroup: coh, showlegend: false, line: {color: col, dash: "dash"}, xaxis: "x", yaxis: "y"});
    traces.push({x: K, y: rows.map(r => r.eta2_state), name: name + " · state", legendgroup: coh, showlegend: false, line: {color: col}, xaxis: "x2", yaxis: "y2"});
    traces.push({x: K, y: rows.map(r => r.eta2_subject), name: name + " · subject", legendgroup: coh, showlegend: false, line: {color: col, dash: "dash"}, xaxis: "x2", yaxis: "y2"});
    traces.push({x: K, y: rows.map(r => r.eta2_state_null_mean), name: name + " · state, shuffled", legendgroup: coh, showlegend: false, line: {color: col, dash: "dot", width: 1}, xaxis: "x2", yaxis: "y2"});
    traces.push({x: K, y: rows.map(r => r.kmeans_ari), name: name + " · ARI", legendgroup: coh, showlegend: false, line: {color: col}, xaxis: "x3", yaxis: "y3"});
  }
  traces.forEach(t => { t.type = "scatter"; if (!t.fill) { t.mode = "lines+markers"; t.marker = {size: 5}; } });
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
$("emission").onchange = () => { fillAxes(); drawScatter(); drawScores(); };
$("K").onchange = () => { fillAxes(); drawScatter(); };
for (const id of ["cohort", "x", "y", "z", "threeD"]) $(id).onchange = drawScatter;
fillAxes(); drawScatter(); drawScores();
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
    rng = np.random.default_rng(SEED)
    cohort_of = np.array([s["cohort"] for s in frozen["subjects"]])
    result, pcs = {e: {c: [] for c in COHORTS} for e in EMISSIONS}, {e: {} for e in EMISSIONS}
    for run in sorted(runs, key=lambda r: (r["emission"], r["num_states"])):
        emission, K = run["emission"], run["num_states"]
        pcs[emission][K] = pca_axes(run["W"][run["cohort"] == "training"])
        for cohort in COHORTS:
            m = run["cohort"] == cohort
            mean, axes_pc, _ = pcs[emission][K]
            plane = (run["W"][m] - mean) @ axes_pc[:2].T
            row = {"K": K, "wandb_run_id": run["wandb_run_id"],
                   **scores(run["W"][m], run["state"][m], run["subject"][m], K, rng, plane,
                            run["n_trials"][m])}
            all_cells = np.asarray(run["cells"]["subject"])
            row["n_dropped"] = int(((cohort_of[all_cells] == cohort)
                                    & (np.asarray(run["cells"]["n_trials"]) < MIN_TRIALS)).sum()
                                   + run["n_cells_not_frozen"][cohort])
            if emission == "hurdle":
                side = [i for i, name in enumerate(run["feature_names"]) if name.startswith("side:")]
                row["side_only"] = side_only(run["W"][m][:, side], run["state"][m], K)
            result[emission][cohort].append(row)
        print(f"{emission} K={K}: " + ", ".join(
            f"{c} sil {fmt(result[emission][c][-1]['silhouette'])}" for c in COHORTS))

    fs.apply_paper_style()
    figs = [fs.save_figure(fig_scores(result), HERE / "fig_r2_scores.svg")]
    for emission in EMISSIONS:
        figs.append(fs.save_figure(fig_scatter(runs, emission, pcs), HERE / f"fig_r2_scatter_{emission}.svg"))
    plt.close("all")
    page = write_interactive(interactive_data(frozen, runs, pcs, result))

    text = REPORT.read_text()
    text = replace_block(text, 1, block_scores(result, frozen))
    text = replace_block(text, 2, block_scatter(pcs))
    REPORT.write_text(text)

    out = {"min_trials": MIN_TRIALS, "n_shuffles": N_SHUFFLES, "scores": result,
           "pca_variance_share": {e: {str(K): v[2] for K, v in pcs[e].items()} for e in EMISSIONS}}
    meta = build_meta("analysis/r2_state_spread.py", WANDB_GROUPS, study_root=HERE.parent,
                      code=[__file__], inputs=[FROZEN, *WEIGHTS.values()])
    write_artifact(HERE / "r2_state_spread.json", {"_meta": meta, **out},
                   outputs=figs + [page] + [f"{REPORT}#result-{n}" for n in (1, 2)])


if __name__ == "__main__":
    main()
