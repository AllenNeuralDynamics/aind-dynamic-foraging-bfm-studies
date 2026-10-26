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
import plot_style  # noqa: E402

WANDB_GROUPS = ["hattori@20261026-002131"]   # read via the committed freezes
RUNS = HERE / "hattori_runs.json"
PLUGIN = HERE / "hattori_plugin.json"
MLE_SOURCE = HERE.parents[1] / "08-hb-vs-gru-heldout" / "analysis" / "ml_params" / "ml_hattori_heldout.csv"
GRU_SOURCE = HERE.parents[1] / "01-gru-scaling-law" / "analysis" / "per_subject_rows.json"
REPORT = HERE / "reports" / "r4-ladder.md"
FIG_LADDER = HERE / "fig_r4_ladder.png"
FIG_STEPS = HERE / "fig_r4_ladder_steps.png"
FIG_SCHEMA = HERE / "fig_r4_ladder_schematic.png"
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


def plot_ladder(r: dict) -> None:
    """Pooled likelihood per rung, steps annotated, GRU curve as reference lines."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    xs = np.arange(len(RUNGS))
    ys = [r["pooled"][k] for k in RUNGS]
    colors = ["0.45", "C1", "C2", "C0"]
    ax.bar(xs, [y - 0.700 for y in ys], bottom=0.700, color=colors, width=0.6)
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.0004, f"{y:.4f}", ha="center", va="bottom", fontsize=11)
    for i in range(len(RUNGS) - 1):
        gain = ys[i + 1] - ys[i]
        ax.annotate("", xy=(i + 0.7, ys[i + 1]), xytext=(i + 0.3, ys[i]),
                    arrowprops={"arrowstyle": "->", "color": "k", "lw": 1.2})
        ax.text(i + 0.5, max(ys[i], ys[i + 1]) + 0.0018, f"{gain:+.4f}", ha="center", fontsize=11,
                color="darkgreen" if gain > 0 else "firebrick")
    for d, v in r["gru_by_d"].items():
        ax.axhline(v, color="C3", lw=0.8, ls="--", alpha=0.35 + 0.1 * list(r["gru_by_d"]).index(d))
    ax.text(-0.45, r["gru_by_d"]["614"] + 0.0003, "GRU, D≈10 (lowest line) to D≈614 (highest)",
            color="C3", va="bottom", fontsize=10)
    ax.set_xticks(xs, ["per-subject\nMLE", "HB plug-in\n(1 fixed θ)", "HB zero-spread\n(posterior, σ=0)",
                       "HB full\n(posterior + σ)"])
    ax.set_ylim(0.700, 0.730)
    ax.set_ylabel("held-out likelihood\n(149 subjects, trial-weighted)")
    ax.set_title("The HB ladder: what each ingredient adds over the per-subject MLE")
    fig.tight_layout(); fig.savefig(FIG_LADDER, dpi=150); plt.close(fig)


def plot_steps(r: dict) -> None:
    """Per-subject distribution of each step (paired by subject)."""
    names = list(r["steps"])
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.violinplot([r["steps"][n]["_values"] for n in names], showmedians=True)
    ax.axhline(0, color="0.4", ls="--", lw=1)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.25 * (hi - lo))
    for i, n in enumerate(names, start=1):
        st = r["steps"][n]
        ax.text(i, hi + 0.06 * (hi - lo), f"{st['frac_positive']:.0%} of subjects gain\n(n={st['n']})",
                ha="center", fontsize=10)
    ax.set_xticks(range(1, len(names) + 1), names, rotation=12)
    ax.set_ylabel("per-subject step\n(held-out likelihood)")
    ax.set_title("Each step, subject by subject")
    fig.tight_layout(); fig.savefig(FIG_STEPS, dpi=150); plt.close(fig)


def plot_schematic() -> None:
    """What each rung predicts with, early vs late in one held-out session (illustrative)."""
    rng = np.random.default_rng(0)
    mu_draws = rng.normal(0.0, 0.25, 60)                 # posterior over the typical session
    session_draws = mu_draws + rng.normal(0.0, 0.6, 60)  # plus between-session spread
    truth = 0.9                                          # this session's own parameter
    fig, axes = plt.subplots(2, 4, figsize=(17, 6.2), sharex=True, sharey=True)
    titles = ["per-subject MLE\none fitted value, fixed", "HB plug-in\none value from the posterior, fixed",
              "HB zero-spread\nposterior draws, reweighted in-session",
              "HB full\ndraws incl. session spread, reweighted"]
    for col, title in enumerate(titles):
        for row, (label, n_seen) in enumerate((("start of session", 0), ("after many trials", 60))):
            ax = axes[row, col]
            ax.axvline(truth, color="C3", lw=1, ls=":")
            if col in (0, 1):
                x = 0.15 if col == 0 else 0.0
                ax.scatter([x], [0.5], s=260, color="0.45" if col == 0 else "C1", zorder=3)
            else:
                draws = mu_draws if col == 2 else session_draws
                w = np.exp(-n_seen * 0.02 * (draws - truth) ** 2)
                w = w / w.max()
                ax.scatter(draws, rng.uniform(0.3, 0.7, len(draws)), s=12 + 160 * w,
                           color="C2" if col == 2 else "C0", alpha=0.25 + 0.7 * w)
            ax.set_yticks([]); ax.set_xlim(-2, 2.2); ax.set_ylim(0, 1)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(label)
    axes[1, 0].set_xlabel("a session-level parameter (illustrative)")
    axes[0, 3].text(truth, 0.95, "this session", color="C3", ha="center", fontsize=9)
    fig.suptitle("What each rung predicts with. Dot size = weight while predicting the next trial "
                 "(SCHEMATIC, not data)", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG_SCHEMA, dpi=150); plt.close(fig)


def render(r: dict) -> str:
    """The result-1 block."""
    lines = [f"![ladder](../{FIG_LADDER.name})", "",
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
              "| step, per subject | median | subjects with a gain |", "|---|---|---|"]
    for name, st in r["steps"].items():
        lines.append(f"| {name} | {st['median']:+.4f} | {st['frac_positive']:.0%} ({round(st['frac_positive'] * st['n'])}/{st['n']}) |")
    lines += ["", f"![schematic](../{FIG_SCHEMA.name})"]
    return "\n".join(lines)


def main() -> None:
    """Compute, plot, rewrite the result block, and write the JSON."""
    plot_style.apply_presentation_style()
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
