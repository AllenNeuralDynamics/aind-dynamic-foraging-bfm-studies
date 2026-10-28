"""Producer for r3: what the two-level HB learns that the per-subject MLE cannot.

Offline. Reads the frozen posterior summaries (``hattori_posteriors.json``) and run table
(``hattori_final_runs.json``), the per-subject MLE Hattori fits that study 08 froze from study 01's run
(``ml_hattori_heldout.csv``), and study 01's per-subject GRU rows. Three questions, one figure each:

1. Do HB and MLE agree on what each subject is like (subject-level parameters)?
2. How much do a subject's parameters move between sessions (the spread the MLE assumes is zero)?
3. Is the HB's per-subject gain over the MLE larger where that spread is larger?
"""
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
from _meta import build_meta, write_artifact  # noqa: E402
import figure_style as fs  # noqa: E402

WANDB_GROUPS = ["hattori@20261027-181415", "hattori@20261027-181430"]   # final set (ADR 0004), via the freeze
POSTERIORS = HERE / "hattori_posteriors.json"
RUNS = HERE / "hattori_final_runs.json"   # refit + rescore (ADR 0004)
MLE_SOURCE = HERE.parents[1] / "08-hb-vs-gru-heldout" / "analysis" / "ml_params" / "ml_hattori_heldout.csv"
GRU_SOURCE = HERE.parents[1] / "01-gru-scaling-law" / "analysis" / "per_subject_rows.json"
REPORT = HERE / "reports" / "r3-posteriors-vs-mle.md"
FIG_PARAMS = HERE / "fig_r3_params_vs_mle.svg"
FIG_SPREAD = HERE / "fig_r3_session_spread.svg"
FIG_GAIN = HERE / "fig_r3_gain_vs_spread.svg"
FIG_EXAMPLES = HERE / "fig_r3_example_subjects.svg"
MLE_NAME = {"bias_l": "biasL"}   # the RL package's column names; biasL is the HB's bias_l
LABEL = {"learn_rate_rew": "learning rate, rewarded", "learn_rate_unrew": "learning rate, unrewarded",
         "forget_rate_unchosen": "forget rate, unchosen", "softmax_inverse_temperature":
         "inverse temperature", "bias_l": "left bias"}


def _ranks(values) -> np.ndarray:
    """Ranks with ties given their average rank, as Spearman's correlation requires."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    for value in np.unique(values):
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def _spearman(x, y) -> float:
    """Spearman rank correlation, tie-aware (the MLE has rates at exactly 0 and 1)."""
    return float(np.corrcoef(_ranks(x), _ranks(y))[0, 1])


def load() -> dict:
    """Per-subject arrays aligned on the subjects every source has."""
    post = json.loads(POSTERIORS.read_text())
    names = post["param_order"]
    hb = {r["subject_id"]: r["params"] for r in post["rows"]}
    runs = {r["subject_id"]: r for r in json.loads(RUNS.read_text())["rows"] if r["state"] == "finished"}
    mle = {r["subject_id"]: r for r in csv.DictReader(MLE_SOURCE.open())}
    gru = {}
    for row in json.loads(GRU_SOURCE.read_text())["rows"]:
        if row["variant"] == "v1" and row["ratio"] == 1.0:
            gru.setdefault(str(row["subject"]), []).append(row["ll"])
    subjects = sorted(set(hb) & set(mle) & set(runs))
    beta_max = {r["beta_max"] for r in post["rows"]}
    assert len(beta_max) == 1, beta_max
    return {"names": names, "subjects": subjects, "hb": hb, "runs": runs, "mle": mle,
            "examples": post["examples"],
            "beta_max": beta_max.pop(),
            "gru": {s: statistics.fmean(v) for s, v in gru.items()}}


def compute(d: dict) -> dict:
    """Parameter agreement, between-session spread, and the gain-vs-spread relation."""
    names, subjects = d["names"], d["subjects"]
    agreement, spread = {}, {}
    for n in names:
        hb_mid = [d["hb"][s][n]["subject"][1] for s in subjects]
        mle = [float(d["mle"][s][MLE_NAME.get(n, n)]) for s in subjects]
        agreement[n] = {"spearman": _spearman(hb_mid, mle),
                        "median_hb": statistics.median(hb_mid), "median_mle": statistics.median(mle)}
        widths = [d["hb"][s][n]["session_width"][1] for s in subjects]
        spread[n] = {"median_session_10_90_width": statistics.median(widths),
                     "median_sigma": statistics.median(d["hb"][s][n]["sigma"][1] for s in subjects)}
    mean_sigma = [statistics.fmean(d["hb"][s][n]["sigma"][1] for n in names) for s in subjects]
    gain = [d["runs"][s]["heldout/eval_likelihood"] - float(d["mle"][s]["eval_likelihood"])
            for s in subjects]
    with_gru = [i for i, s in enumerate(subjects) if s in d["gru"]]
    gap = [d["runs"][subjects[i]]["heldout/eval_likelihood"] - d["gru"][subjects[i]] for i in with_gru]
    return {"n_subjects": len(subjects), "agreement": agreement, "spread": spread,
            "gain_vs_mle": {"spearman_with_mean_sigma": _spearman(mean_sigma, gain),
                            "median": statistics.median(gain)},
            "gap_to_gru_d614": {"spearman_with_mean_sigma": _spearman([mean_sigma[i] for i in with_gru], gap),
                                "median": statistics.median(gap), "n": len(with_gru)},
            "_plot": {"mean_sigma": mean_sigma, "gain": gain, "gap": gap,
                      "gap_sigma": [mean_sigma[i] for i in with_gru]}}


SHORT = {"learn_rate_rew": "Learning rate\n(rewarded)", "learn_rate_unrew": "Learning rate\n(unrewarded)",
         "forget_rate_unchosen": "Forget rate\n(unchosen)",
         "softmax_inverse_temperature": "Inverse\ntemperature", "bias_l": "Left bias"}
HB_COLOR, MLE_COLOR = fs.OKABE_ITO["vermillion"], "0.2"


def plot(d: dict, r: dict) -> None:
    """Four figures."""
    fs.apply_paper_style()
    names, subjects = d["names"], d["subjects"]
    fig, axes = plt.subplots(1, 5, figsize=fs.mm(183, 44))
    for ax, n in zip(axes, names):
        q = np.array([d["hb"][s][n]["subject"] for s in subjects])
        x = np.array([float(d["mle"][s][MLE_NAME.get(n, n)]) for s in subjects])
        if n == "softmax_inverse_temperature":
            # The MLE's temperature is unbounded; the HB's is capped at beta_max. Log scale, and
            # count MLE fits beyond the HB's cap rather than letting one squash the panel.
            beta_max = d["beta_max"]
            lo, hi = 0.5, beta_max * 1.2
            above = int(np.sum(x > beta_max))
            ax.set_xscale("log"); ax.set_yscale("log")
            fs.plain_log_axis(ax)
            ax.text(0.97, 0.03, f"{above} MLE fit{'s' if above != 1 else ''} > {beta_max:g}\noff-axis",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=5, color="0.35")
        else:
            lo, hi = min(x.min(), q[:, 0].min()), max(x.max(), q[:, 2].max())
            pad = 0.03 * (hi - lo)
            lo, hi = lo - pad, hi + pad
        keep = x <= hi
        ax.plot([lo, hi], [lo, hi], color="0.6", ls=(0, (3, 2)), lw=0.5, zorder=0)
        ax.errorbar(x[keep], q[keep, 1], yerr=[q[keep, 1] - q[keep, 0], q[keep, 2] - q[keep, 1]], fmt="o",
                    ms=1.6, mfc=fs.OKABE_ITO["blue"], mec="none", ecolor=fs.OKABE_ITO["blue"],
                    elinewidth=0.3, alpha=0.55)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_box_aspect(1)
        ax.set_title(SHORT[n])
        ax.text(0.04, 0.96, f"ρ = {fs.num(r['agreement'][n]['spearman'], '.2f')}", transform=ax.transAxes,
                ha="left", va="top", fontsize=6)
    axes[0].set_ylabel("HB subject level\n(median, 5–95%)")
    fig.supxlabel("Per-subject MLE")
    fs.save_figure(fig, FIG_PARAMS); plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=fs.mm(fs.fit_width(groups=5, axes=5), 42))
    for ax, n in zip(axes, names):
        widths = [d["hb"][s][n]["session_width"][1] for s in subjects]
        fs.violins(ax, [widths], [fs.OKABE_ITO["vermillion"]], width=0.8)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.set_title(SHORT[n])
        ax.set_xlabel(f"median {fs.num(statistics.median(widths), '.3g')}")
    axes[0].set_ylabel("Between-session width\n(10–90% of sessions)")
    fs.save_figure(fig, FIG_SPREAD); plt.close(fig)

    ex = d["examples"]
    fig, axes = plt.subplots(3, 5, figsize=fs.mm(183, 100), squeeze=False)
    for row, (label, subject) in enumerate(zip(("Smallest", "Median", "Largest"), ex["subjects_by_gain"])):
        gain = d["runs"][subject]["heldout/eval_likelihood"] - float(d["mle"][subject]["eval_likelihood"])
        for col, n in enumerate(names):
            ax = axes[row, col]
            samples = ex["samples"][subject][n]
            ax.hist(samples["session"], bins=40, density=True, alpha=0.35, color=HB_COLOR,
                    label="Across sessions", rasterized=True)
            ax.hist(samples["typical"], bins=40, density=True, alpha=0.8, color=HB_COLOR,
                    histtype="step", lw=0.6, label="Typical session")
            mle_value = float(d["mle"][subject][MLE_NAME.get(n, n)])
            lo = min(min(samples["session"]), min(samples["typical"]))
            hi = max(max(samples["session"]), max(samples["typical"]))
            pad = 0.05 * (hi - lo)
            if lo - pad <= mle_value <= hi + pad:
                ax.axvline(mle_value, color=MLE_COLOR, lw=0.8, label="MLE")
            else:   # far outside the posterior: mark it at the edge rather than rescale the panel
                right = mle_value > hi
                ax.annotate(f"MLE {mle_value:.3g}", xy=(1 if right else 0, 0.88), xycoords="axes fraction",
                            xytext=(-14 if right else 14, 0), textcoords="offset points",
                            ha="right" if right else "left", va="center", fontsize=5,
                            arrowprops={"arrowstyle": "->", "lw": 0.5, "color": MLE_COLOR})
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            if row == 0:
                ax.set_title(SHORT[n])
            if col == 0:
                ax.set_ylabel(f"{label} gain\n{subject}, {fs.num(gain, '+.4f')}")
    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(0, 1.0), fontsize=5)
    fs.save_figure(fig, FIG_EXAMPLES); plt.close(fig)

    p = r["_plot"]
    fig, (a, b) = plt.subplots(1, 2, figsize=fs.mm(fs.fit_width(square_panels=2, axes=2), 55))
    for ax, xs, ys, ylabel, rho, letter in (
            (a, p["mean_sigma"], p["gain"], "HB − MLE Hattori", r["gain_vs_mle"]["spearman_with_mean_sigma"], "a"),
            (b, p["gap_sigma"], p["gap"], "HB − GRU D≈614", r["gap_to_gru_d614"]["spearman_with_mean_sigma"], "b")):
        ax.scatter(xs, ys, s=3, color=fs.OKABE_ITO["blue"], alpha=0.7, linewidths=0)
        ax.axhline(0, color="0.5", ls=(0, (3, 2)), lw=0.5, zorder=0)
        ax.set_xlabel("Between-session spread (mean σ)")
        ax.set_ylabel(f"{ylabel}\n(held-out likelihood)")
        ax.text(0.96, 0.96, f"ρ = {fs.num(rho, '.2f')}", transform=ax.transAxes, ha="right", va="top", fontsize=6)
        fs.panel_label(ax, letter)
    fs.save_figure(fig, FIG_GAIN); plt.close(fig)


def render(r: dict) -> str:
    """The result-1 block."""
    lines = [f"![parameters vs MLE](../{FIG_PARAMS.name})", "",
             "*HB subject-level posterior (median, 5–95%) against the per-subject MLE, one dot per "
             "subject; dashed line = identity; ρ = Spearman.*", "",
             "| parameter | Spearman HB vs MLE | median HB | median MLE | median between-session width (10–90% session) |",
             "|---|---|---|---|---|"]
    for n, a in r["agreement"].items():
        lines.append(f"| {LABEL[n]} | {a['spearman']:.2f} | {a['median_hb']:.3g} | {a['median_mle']:.3g} | "
                     f"{r['spread'][n]['median_session_10_90_width']:.3g} |")
    g, c = r["gain_vs_mle"], r["gap_to_gru_d614"]
    lines += ["", f"![between-session spread](../{FIG_SPREAD.name})", "",
              "*How far each parameter moves between a subject’s sessions (10–90% session width, "
              "own scale), one value per subject.*", "",
              f"![gain vs spread](../{FIG_GAIN.name})", "",
              "*Per held-out subject: held-out likelihood difference against the subject’s "
              "between-session spread; a, vs MLE Hattori; b, vs GRU D≈614.*", "",
              f"![example subjects](../{FIG_EXAMPLES.name})", "",
              "*HB posterior for the subjects with the smallest, median and largest gain over the MLE: "
              "the typical session (outline) vs across sessions (filled), against the single MLE value.*", "",
              f"- HB − MLE Hattori per subject: median {g['median']:+.4f}; Spearman with the subject's "
              f"between-session spread {g['spearman_with_mean_sigma']:+.2f} (n={r['n_subjects']}).",
              f"- HB − GRU D≈614 per subject: median {c['median']:+.4f}; Spearman with the same spread "
              f"{c['spearman_with_mean_sigma']:+.2f} (n={c['n']}, subjects in both cohorts)."]
    return "\n".join(lines)


def main() -> None:
    """Compute, plot, rewrite the result block, and write the JSON."""
    d = load()
    result = compute(d)
    plot(d, result)
    result.pop("_plot")
    begin, end = "<!-- BEGIN result-1 -->", "<!-- END result-1 -->"
    head, rest = REPORT.read_text().split(begin)
    _, tail = rest.split(end)
    REPORT.write_text(f"{head}{begin}\n{render(result)}\n{end}{tail}")
    meta = build_meta("analysis/r3_posteriors_vs_mle.py", WANDB_GROUPS, study_root=HERE.parent,
                      code=[__file__], inputs=[POSTERIORS, RUNS, MLE_SOURCE, GRU_SOURCE])
    write_artifact(HERE / "r3_posteriors_vs_mle.json", {"_meta": meta, **result},
                   outputs=[FIG_PARAMS, FIG_SPREAD, FIG_GAIN, FIG_EXAMPLES,
                            HERE / "reports" / "r3-posteriors-vs-mle.md#result-1"])


if __name__ == "__main__":
    main()
