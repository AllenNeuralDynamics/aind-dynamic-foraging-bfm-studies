"""Producer for r4: the HB ladder, from per-subject MLE to the full two-level HB, step by step.

Offline. Every rung is scored on the same held-out eval sessions of the same subjects:

- MLE: one parameter set per subject, maximum likelihood (study 01 r8 pooled; study 08's per-subject
  table for the paired steps, the same run).
- plug-in: one parameter set per subject, the HB's typical-session posterior median.
- zero-spread: every posterior draw of the typical session, no between-session spread; each session
  is scored by the session-marginal, so within a session the draws are reweighted by its earlier trials.
- full HB: adds between-session spread (the r1 headline).
- GRU H128 (study 01 r1), for reference.

Three figures: the pooled ladder, the per-subject distribution of each step, and a schematic of what
each rung does (illustrative, not data).
"""
import csv
import json
import math
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
import cross_study  # noqa: E402
import figure_style as fs  # noqa: E402

WANDB_GROUPS = ["hattori@20261027-181415", "hattori@20261027-181430"]   # final set (ADR 0004), via the freeze
RUNS = HERE / "hattori_final_runs.json"   # refit + rescore (ADR 0004)
PLUGIN = HERE / "hattori_plugin.json"
MLE_SOURCE = HERE.parents[1] / "08-hb-vs-gru-heldout" / "analysis" / "ml_params" / "ml_hattori_heldout.csv"
GRU_SOURCE = HERE.parents[1] / "01-gru-scaling-law" / "analysis" / "per_subject_rows.json"
REPORT = HERE / "reports" / "r4-ladder.md"
FIG_LADDER = HERE / "fig_r4_ladder.svg"
FIG_STEPS = HERE / "fig_r4_ladder_steps.svg"
FIG_SCHEMA = HERE / "fig_r4_ladder_schematic.svg"
RUNGS = ["MLE", "plug-in", "zero-spread", "full HB"]


def load() -> dict:
    """Per-subject log-likelihood sums and trials for every HB rung and the MLE."""
    runs = {r["subject_id"]: r for r in json.loads(RUNS.read_text())["rows"] if r["state"] == "finished"}
    plugin = {r["subject_id"]: r for r in json.loads(PLUGIN.read_text())["rows"]}
    mle = {r["subject_id"]: r for r in csv.DictReader(MLE_SOURCE.open())}
    subjects = sorted(set(runs) & set(plugin) & set(mle))
    assert len(subjects) == 149, len(subjects)
    per = {}
    for s in subjects:
        n = runs[s]["heldout/num_eval_trials"]
        assert plugin[s]["n_trials"] == n and int(mle[s]["eval_total_trials"]) == n, s
        per[s] = {"n": n, "MLE": float(mle[s]["eval_total_log_likelihood"]),
                  "plug-in": plugin[s]["plugin_log_lik_sum"],
                  "zero-spread": plugin[s]["zero_spread_log_lik_sum"],
                  "full HB": runs[s]["heldout/eval_log_lik_sum"]}
    gru = {}
    for row in json.loads(GRU_SOURCE.read_text())["rows"]:
        if row["variant"] == "v1" and row["ratio"] == 1.0:
            gru.setdefault(str(row["subject"]), []).append(row["ll"])
    return {"per": per, "gru_subject": {s: statistics.fmean(v) for s, v in gru.items()}}


def compute(d: dict) -> dict:
    """Pooled rungs, pooled steps, and the per-subject step distributions."""
    per = d["per"]
    trials = sum(p["n"] for p in per.values())
    pooled = {rung: math.exp(sum(p[rung] for p in per.values()) / trials) for rung in RUNGS}
    lik = {rung: {s: math.exp(p[rung] / p["n"]) for s, p in per.items()} for rung in RUNGS}
    steps = {}
    for lo, hi in zip(RUNGS, RUNGS[1:]):
        diffs = [lik[hi][s] - lik[lo][s] for s in per]
        steps[f"{hi} − {lo}"] = {"pooled": pooled[hi] - pooled[lo], "median": statistics.median(diffs),
                                  "frac_positive": sum(x > 0 for x in diffs) / len(diffs), "n": len(diffs),
                                  "_values": diffs}
    common = [s for s in per if s in d["gru_subject"]]
    diffs = [d["gru_subject"][s] - lik["full HB"][s] for s in common]
    steps["GRU D≈614 − full HB"] = {"pooled": None, "median": statistics.median(diffs),
                                    "frac_positive": sum(x > 0 for x in diffs) / len(diffs),
                                    "n": len(diffs), "_values": diffs}
    return {"n_subjects": len(per), "n_trials": trials, "pooled": pooled,
            "mle_published": cross_study.rl_baseline("hattori"),
            "gru_by_d": {str(k): v for k, v in cross_study.gru_h128_curve().items()}, "steps": steps}


RUNG_COLOR = {"MLE": "0.45", "plug-in": fs.OKABE_ITO["orange"], "zero-spread": fs.OKABE_ITO["green"],
              "full HB": fs.OKABE_ITO["vermillion"]}
RUNG_TICK = ["Per-subject\nMLE", "HB plug-in\n(1 fixed θ)", "HB zero-\nspread (σ = 0)", "HB full\n(+ σ)"]


def plot_ladder(r: dict) -> None:
    """Pooled likelihood per rung, steps annotated, GRU curve as reference lines."""
    fig, ax = plt.subplots(figsize=fs.mm(fs.fit_width(groups=4), 60))
    xs = np.arange(len(RUNGS))
    ys = [r["pooled"][k] for k in RUNGS]
    ax.bar(xs, [y - 0.700 for y in ys], bottom=0.700, color=[RUNG_COLOR[k] for k in RUNGS], width=0.6)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.4f}", xy=(x, y), xytext=(0, 1.5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6)
    for i in range(len(RUNGS) - 1):
        gain = ys[i + 1] - ys[i]
        ax.annotate("", xy=(i + 0.7, ys[i + 1]), xytext=(i + 0.3, ys[i]),
                    arrowprops={"arrowstyle": "->", "color": "0.2", "lw": 0.6, "shrinkA": 0, "shrinkB": 0})
        rising = gain > 0     # label on the arrow's upper side: left of a rising arrow, right of a falling one
        ax.annotate(fs.num(gain, "+.4f"), xy=(i + 0.5, (ys[i] + ys[i + 1]) / 2), xytext=(-4 if rising else 4, 3),
                    textcoords="offset points", ha="right" if rising else "left", va="bottom", fontsize=6,
                    color=fs.OKABE_ITO["green"] if rising else fs.OKABE_ITO["vermillion"])
    gru = list(r["gru_by_d"].items())
    for k, (d, v) in enumerate(gru):
        ax.axhline(v, color=fs.OKABE_ITO["blue"], lw=0.5, ls=(0, (3, 2)), alpha=0.3 + 0.7 * k / (len(gru) - 1),
                   zorder=0)
    ax.annotate(f"GRU, D≈{gru[0][0]} (lowest line) to D≈{gru[-1][0]}",
                xy=(0, r["gru_by_d"][gru[-1][0]]), xycoords=("axes fraction", "data"), xytext=(2, 1.5),
                textcoords="offset points", color=fs.OKABE_ITO["blue"], va="bottom", fontsize=6)
    ax.set_xticks(xs, RUNG_TICK)
    ax.set_ylim(0.700, 0.730)
    ax.set_ylabel(f"Held-out likelihood\n({r['n_subjects']} subjects, trial-weighted)")
    fs.save_figure(fig, FIG_LADDER); plt.close(fig)


def plot_steps(r: dict) -> None:
    """Per-subject distribution of each step (paired by subject)."""
    names = list(r["steps"])
    colors = [RUNG_COLOR[n.split(" − ")[0]] if n.split(" − ")[0] in RUNG_COLOR else fs.OKABE_ITO["blue"]
              for n in names]
    fig, ax = plt.subplots(figsize=fs.mm(fs.fit_width(groups=4), 55))
    fs.violins(ax, [r["steps"][n]["_values"] for n in names], colors, [n.replace(" − ", "\n− ") for n in names])
    ax.axhline(0, color="0.5", ls=(0, (3, 2)), lw=0.5, zorder=0)
    for i, n in enumerate(names, start=1):
        st = r["steps"][n]
        ax.annotate(f"{st['frac_positive']:.0%} gain\n(n = {st['n']})", xy=(i, 1), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=6)
    ax.set_ylabel("Per-subject step\n(held-out likelihood)")
    fs.save_figure(fig, FIG_STEPS); plt.close(fig)


def plot_schematic() -> None:
    """What each rung predicts with, early vs late in one held-out session (illustrative)."""
    rng = np.random.default_rng(0)
    mu_draws = rng.normal(0.0, 0.25, 60)                 # posterior over the typical session
    session_draws = mu_draws + rng.normal(0.0, 0.6, 60)  # plus between-session spread
    truth = 0.9                                          # this session's own parameter
    fig, axes = plt.subplots(2, 4, figsize=fs.mm(183, 72), sharex=True, sharey=True)
    titles = ["Per-subject MLE\none fitted value, fixed", "HB plug-in\none value from the posterior, fixed",
              "HB zero-spread\nposterior draws, reweighted in-session",
              "HB full\ndraws incl. session spread, reweighted"]
    for col, title in enumerate(titles):
        color = RUNG_COLOR[RUNGS[col]]
        for row, (label, n_seen) in enumerate((("Start of session", 0), ("After many trials", 60))):
            ax = axes[row, col]
            ax.axvline(truth, color="0.2", lw=0.6, ls=":")
            if col in (0, 1):
                ax.scatter([0.15 if col == 0 else 0.0], [0.5], s=16, color=color, zorder=3, linewidths=0)
            else:
                draws = mu_draws if col == 2 else session_draws
                w = np.exp(-n_seen * 0.02 * (draws - truth) ** 2)
                w = w / w.max()
                ax.scatter(draws, rng.uniform(0.3, 0.7, len(draws)), s=1 + 10 * w, color=color,
                           alpha=0.25 + 0.7 * w, linewidths=0)
            ax.set_yticks([]); ax.set_xlim(-2, 2.2); ax.set_ylim(0, 1)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(label)
    axes[0, 3].text(truth, 0.95, "this session", color="0.2", ha="center", va="top", fontsize=6)
    fig.supxlabel("A session-level parameter (illustrative)")
    fs.save_figure(fig, FIG_SCHEMA); plt.close(fig)


def render(r: dict) -> str:
    """The result-1 block."""
    lines = [f"![ladder](../{FIG_LADDER.name})", "",
             "*The HB ladder: what each ingredient adds over the per-subject MLE; dashed lines, the GRU "
             "H128 curve (study 01 r1).*", "",
             "| rung | pooled held-out likelihood | step |", "|---|---|---|"]
    prev = None
    for rung in RUNGS:
        v = r["pooled"][rung]
        lines.append(f"| {rung} | {v:.4f} | {'' if prev is None else f'{v - prev:+.4f}'} |")
        prev = v
    lines += [f"| GRU H128 (study 01 r1) | " + ", ".join(f"D≈{d} {v:.4f}" for d, v in r["gru_by_d"].items())
              + " | |", "",
              f"MLE pooled from study 08's per-subject table: {r['pooled']['MLE']:.4f} "
              f"(study 01 r8 publishes {r['mle_published']:.4f}).", "",
              f"![steps](../{FIG_STEPS.name})", "",
              "*Each step, subject by subject (paired), with the share of subjects that gain.*", "",
              "| step, per subject | median | subjects with a gain |", "|---|---|---|"]
    for name, st in r["steps"].items():
        lines.append(f"| {name} | {st['median']:+.4f} | {st['frac_positive']:.0%} ({round(st['frac_positive'] * st['n'])}/{st['n']}) |")
    lines += ["", f"![schematic](../{FIG_SCHEMA.name})", "",
              "*Schematic, not data: what each rung predicts with, at the start of a session and after many "
              "trials. Dot size = weight while predicting the next trial; dotted line = this session’s "
              "own parameter.*"]
    return "\n".join(lines)


def main() -> None:
    """Compute, plot, rewrite the result block, and write the JSON."""
    fs.apply_paper_style()
    result = compute(load())
    plot_ladder(result); plot_steps(result); plot_schematic()
    for st in result["steps"].values():
        st.pop("_values")
    begin, end = "<!-- BEGIN result-1 -->", "<!-- END result-1 -->"
    head, rest = REPORT.read_text().split(begin)
    _, tail = rest.split(end)
    REPORT.write_text(f"{head}{begin}\n{render(result)}\n{end}{tail}")
    meta = build_meta("analysis/r4_ladder.py", WANDB_GROUPS, study_root=HERE.parent, code=[__file__],
                      inputs=[RUNS, PLUGIN, MLE_SOURCE, GRU_SOURCE, cross_study.RL_SOURCE,
                              cross_study.GRU_CURVE_SOURCE])
    write_artifact(HERE / "r4_ladder.json", {"_meta": meta, **result},
                   outputs=[FIG_LADDER, FIG_STEPS, FIG_SCHEMA, HERE / "reports" / "r4-ladder.md#result-1"])


if __name__ == "__main__":
    main()
