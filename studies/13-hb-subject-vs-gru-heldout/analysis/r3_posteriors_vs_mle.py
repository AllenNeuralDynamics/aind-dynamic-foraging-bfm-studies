"""Producer for r3: what the two-level HB learns that the per-subject MLE cannot.

Offline. Reads the frozen posterior summaries (``hattori_posteriors.json``) and run table
(``hattori_runs.json``), the per-subject MLE Hattori fits that study 08 froze from study 01's run
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
import plot_style  # noqa: E402

WANDB_GROUPS = ["hattori@20261026-002131"]   # read via the committed freezes
POSTERIORS = HERE / "hattori_posteriors.json"
RUNS = HERE / "hattori_runs.json"
MLE_SOURCE = HERE.parents[1] / "08-hb-vs-gru-heldout" / "analysis" / "ml_params" / "ml_hattori_heldout.csv"
GRU_SOURCE = HERE.parents[1] / "01-gru-scaling-law" / "analysis" / "per_subject_rows.json"
REPORT = HERE / "reports" / "r3-posteriors-vs-mle.md"
FIG_PARAMS = HERE / "fig_r3_params_vs_mle.png"
FIG_SPREAD = HERE / "fig_r3_session_spread.png"
FIG_GAIN = HERE / "fig_r3_gain_vs_spread.png"
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


def plot(d: dict, r: dict) -> None:
    """Three figures."""
    plot_style.apply_presentation_style()
    names, subjects = d["names"], d["subjects"]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
    for ax, n in zip(axes, names):
        q = np.array([d["hb"][s][n]["subject"] for s in subjects])
        x = np.array([float(d["mle"][s][MLE_NAME.get(n, n)]) for s in subjects])
        ax.errorbar(x, q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]], fmt="o", ms=3,
                    alpha=0.6, elinewidth=0.6)
        title = f"{LABEL[n]}\nSpearman {r['agreement'][n]['spearman']:.2f}"
        if n == "softmax_inverse_temperature":
            # The MLE's temperature is unbounded; the HB's is capped at beta_max. Log scale, and
            # count MLE fits beyond the HB's cap rather than letting one squash the panel.
            beta_max = d["beta_max"]
            above = int(np.sum(x > beta_max))
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(0.5, beta_max * 1.2); ax.set_ylim(0.5, beta_max * 1.2)
            title += f"\n({above} MLE fit{'s' if above != 1 else ''} > {beta_max:g} off-axis)"
            lo, hi = 0.5, beta_max * 1.2
        else:
            lo, hi = min(x.min(), q[:, 0].min()), max(x.max(), q[:, 2].max())
        ax.plot([lo, hi], [lo, hi], color="0.5", ls="--", lw=1)
        ax.set_title(title)
        ax.set_xlabel("per-subject MLE")
    axes[0].set_ylabel("HB subject level (median, 5–95%)")
    fig.tight_layout(); fig.savefig(FIG_PARAMS, dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
    for ax, n in zip(axes, names):
        widths = [d["hb"][s][n]["session_width"][1] for s in subjects]
        ax.violinplot(widths, showmedians=True)
        ax.set_xticks([])
        ax.set_title(f"{LABEL[n]}\nmedian {statistics.median(widths):.3g}")
    axes[0].set_ylabel("between-session width\n(10–90% session, own scale)")
    fig.tight_layout(); fig.savefig(FIG_SPREAD, dpi=150); plt.close(fig)

    p = r["_plot"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
    a.scatter(p["mean_sigma"], p["gain"], s=12, alpha=0.7)
    a.axhline(0, color="0.5", ls="--", lw=1)
    a.set_xlabel("between-session spread (mean σ)")
    a.set_ylabel("HB − MLE Hattori")
    a.set_title(f"Gain over MLE (Spearman {r['gain_vs_mle']['spearman_with_mean_sigma']:.2f})")
    b.scatter(p["gap_sigma"], p["gap"], s=12, alpha=0.7, color="C1")
    b.axhline(0, color="0.5", ls="--", lw=1)
    b.set_xlabel("between-session spread (mean σ)")
    b.set_ylabel("HB − GRU D≈614")
    b.set_title(f"Gap to GRU (Spearman {r['gap_to_gru_d614']['spearman_with_mean_sigma']:.2f})")
    fig.suptitle("Per held-out subject: held-out likelihood difference vs the subject's between-session spread")
    fig.tight_layout(); fig.savefig(FIG_GAIN, dpi=150); plt.close(fig)


def render(r: dict) -> str:
    """The result-1 block."""
    lines = [f"![parameters vs MLE](../{FIG_PARAMS.name})", "",
             "| parameter | Spearman HB vs MLE | median HB | median MLE | median between-session width (10–90% session) |",
             "|---|---|---|---|---|"]
    for n, a in r["agreement"].items():
        lines.append(f"| {LABEL[n]} | {a['spearman']:.2f} | {a['median_hb']:.3g} | {a['median_mle']:.3g} | "
                     f"{r['spread'][n]['median_session_10_90_width']:.3g} |")
    g, c = r["gain_vs_mle"], r["gap_to_gru_d614"]
    lines += ["", f"![between-session spread](../{FIG_SPREAD.name})", "",
              f"![gain vs spread](../{FIG_GAIN.name})", "",
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
                   outputs=[FIG_PARAMS, FIG_SPREAD, FIG_GAIN,
                            HERE / "reports" / "r3-posteriors-vs-mle.md#result-1"])


if __name__ == "__main__":
    main()
