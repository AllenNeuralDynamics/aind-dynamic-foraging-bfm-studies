"""Producer for r3: how each GLM-HMM state spreads across subjects (offline).

Reads ``frozen_r3.json`` and ``frozen_r3_figure.npz`` (see ``freeze_r3.py``): one GLM per {subject,
state} under each K-grid run's global fit, for 18 ways of cutting trials into states (two
labellings, global fit or each mouse's round-2 refit, times nine posterior and run-length
cuts), with each setting's cluster scores, which the freeze computes with ``r3_scores.py``.
Writes ``r3_state_spread.json``, the figures ``fig_r3_*.svg``, the result blocks
of ``reports/r3-state-spread.md``:
- result-1: cluster scores against K of the uncut global setting (``global_p0_m1``);
- result-2: that setting's {subject, state} cells in weight space, one panel per K;
- result-3: how the cuts and the refit labelling change the scores;
- result-4: checks of the round-2 refits.
The interactive explorer of every setting is ``explorer_r3.py`` (``make r3-explorer``), which
reads the freeze's W&B archive.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
import figure_style as fs  # noqa: E402
from _meta import build_meta, write_artifact  # noqa: E402

sys.path.insert(0, str(HERE))
from r3_scores import BIG_CELL, COHORTS, MIN_TRIALS, N_SHUFFLES  # noqa: E402

WANDB_GROUPS = ["hurdle@20261027-212905", "reference@20261027-213006", "refit-prior@20261101-130226"]
FROZEN = HERE / "frozen_r3.json"
EMISSIONS = ("hurdle", "reference")
LABELLINGS = ("global", "refit")
FIGURE = HERE / "frozen_r3_figure.npz"
ANCHOR = "global_p0_m1"
REPORT = HERE / "reports" / "r3-state-spread.md"
COHORT_LABEL = {"heldout": "held-out mice", "training": "training mice"}
COHORT_COLOR = {"heldout": fs.OKABE_ITO["blue"], "training": fs.OKABE_ITO["vermillion"]}
STATE_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442",
                "#000000", "#999999", "#882255"]
CUT_COLORS = {0.0: "#0072B2", 0.8: "#E69F00", 0.95: "#D55E00"}   # posterior cut
CUT_DASH = {1: "-", 10: "--", 30: ":"}                              # run-length cut


def load():
    """The frozen runs: every setting's scores, and the uncut global setting's kept cells.

    States are ranked by the anchor setting's trials (state 1 = most used) in every setting.
    The anchor's cells (``frozen_r3_figure.npz``) carry PC1-3 of its training cells; the stars
    are the global fit's states on the same axes.
    """
    frozen = json.loads(FROZEN.read_text())
    cohort_of = np.array([s["cohort"] for s in frozen["subjects"]])
    figure = np.load(FIGURE)
    runs = []
    for run in frozen["runs"]:
        emission, K = run["emission"], run["num_states"]
        rank = np.asarray(run["state_rank"])
        pre = f"{emission}_K{K}_"
        n = figure[pre + "n_trials"]
        keep = n >= MIN_TRIALS
        subject = figure[pre + "subject"].astype(int)[keep]
        anchor = {**run["settings"][ANCHOR], "state": rank[figure[pre + "state"].astype(int)[keep]],
                  "subject": subject, "cohort": cohort_of[subject], "n_trials": n[keep],
                  "coords": figure[pre + "pc"].astype(np.float64)[keep]}
        G = np.asarray(run["global_weights"])[np.argsort(rank)]          # ranked order
        stars = (G - np.asarray(run["pca_mean"])) @ np.asarray(run["pca_axes"]).T
        runs.append({**run, "settings": {**run["settings"], ANCHOR: anchor}, "star_coords": stars})
    return frozen, runs


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
                                        [r["n_trials_passing"] / a["n_trials_passing"] for r, a in zip(rows, anchor)],
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
        "Silhouette (all weights, training mice) by K, and the trials the cut keeps (counted before "
        "any cell-size floor) as a share of the uncut setting's:",
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
            kept = [rows[K]["n_trials_passing"] / anchor[K]["n_trials_passing"] for K in (2, 10)]
            lines.append(f"| `{key}` | " + " | ".join(fmt(rows[K].get("silhouette")) for K in Ks)
                         + f" | {kept[0]:.0%} | {kept[1]:.0%} |")
        lines.append("")
    lines += ["![cuts](../fig_r3_cuts.svg)"]
    return "\n".join(lines)


def block_refits(runs, frozen):
    """result-4: the round-2 refits behind the refit labelling."""
    MIN_FROZEN = frozen["data"]["min_frozen_trials"]
    lines = [
        "Each mouse's round-2 refit, recomputed by the freeze (EM from the global fit on the "
        "mouse's train sessions, r2's picked prior at that K). *Held-out check*: largest "
        "difference from the refits round 2 saved for the 149 held-out mice (weights and "
        "transitions; training-mouse refits were not saved). *Not aligned*: refits whose states "
        "match the global states other than one to one in order (least-squares weight matching). "
        "*Second step*: the largest change of a second M-step from any {subject, state} fit, a "
        f"convergence check over the frozen cells (at least {MIN_FROZEN} kept trials): Newton passes "
        f"of {frozen['data']['newton_steps']} steps repeat until one more pass moves no frozen cell by more "
        f"than {frozen['data']['converged']:g}, at most {frozen['data']['max_passes']}. *Passes*: the most "
        "any cell needed.",
        "",
        "| emission | K | σ | c | round-2 run | refits | held-out check | not aligned | not converged | second step | passes |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
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
            f"{step:.0e} | {r['max_newton_passes']} |")
    return "\n".join(lines)


def replace_block(text, n, body):
    """Replace the content between the result-n markers."""
    pattern = re.compile(rf"(<!-- BEGIN result-{n} -->\n).*?(\n<!-- END result-{n} -->)", re.S)
    if not pattern.search(text):
        raise ValueError(f"result-{n} markers missing")
    return pattern.sub(lambda m: m.group(1) + body.strip("\n") + m.group(2), text)


def main() -> None:
    frozen, runs = load()
    result = {e: {c: cut_rows(runs, e, ANCHOR, c) for c in COHORTS} for e in EMISSIONS}

    fs.apply_paper_style()
    figs = [fs.save_figure(fig_scores(result), HERE / "fig_r3_scores.svg")]
    for emission in EMISSIONS:
        figs.append(fs.save_figure(fig_scatter(runs, emission), HERE / f"fig_r3_scatter_{emission}.svg"))
    figs.append(fs.save_figure(fig_cuts(runs), HERE / "fig_r3_cuts.svg"))
    plt.close("all")

    text = REPORT.read_text()
    text = replace_block(text, 1, block_scores(result, frozen))
    text = replace_block(text, 2, block_scatter())
    text = replace_block(text, 3, block_cuts(runs, frozen))
    text = replace_block(text, 4, block_refits(runs, frozen))
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
                      code=[__file__, HERE / "r3_scores.py"], inputs=[FROZEN, FIGURE])
    write_artifact(HERE / "r3_state_spread.json", {"_meta": meta, **out},
                   outputs=figs + [f"{REPORT}#result-{n}" for n in (1, 2, 3, 4)])


if __name__ == "__main__":
    main()
