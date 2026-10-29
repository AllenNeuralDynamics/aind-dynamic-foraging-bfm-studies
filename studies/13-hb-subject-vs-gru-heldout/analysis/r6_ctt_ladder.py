"""Producer for r6: the HB ladder for CTT, beside the HB-Hattori ladder of r4.

Offline. The same rungs as r4 (``r4_ladder.py``'s docstring says what each one isolates), scored on
the same held-out eval sessions of the same subjects, for the ``ctt`` variant:

- MLE: study 08's per-subject MLE CTT table (the study 01 r8 run);
- plug-in and zero-spread: ``ctt_plugin.json`` (``make freeze-plugin-ctt``);
- full HB: the ctt final set (``ctt_final_runs.json``, r5's headline).

The loading, step and per-subject plots are r4's own functions, pointed at the CTT inputs, so the two
ladders are computed identically. One extra figure draws both ladders side by side.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
sys.path.insert(0, str(HERE))
from _meta import build_meta, write_artifact  # noqa: E402
import cross_study  # noqa: E402
import figure_style as fs  # noqa: E402
import r4_ladder as r4  # noqa: E402

WANDB_GROUPS = ["ctt@20261028-003745", "ctt@20261028-020002"]   # via the ctt freezes
RUNS = HERE / "ctt_final_runs.json"
PLUGIN = HERE / "ctt_plugin.json"
MLE_SOURCE = HERE.parents[1] / "08-hb-vs-gru-heldout" / "analysis" / "ml_params" / "ml_ctt_heldout.csv"
HATTORI_LADDER = HERE / "r4_ladder.json"           # r4's committed result, for the side-by-side
REPORT = HERE / "reports" / "r6-ctt-ladder.md"
FIG_LADDER = HERE / "fig_r6_ctt_ladder.svg"
FIG_STEPS = HERE / "fig_r6_ctt_ladder_steps.svg"
FIG_BOTH = HERE / "fig_r6_both_ladders.svg"


def _ctt_ladder() -> dict:
    """r4's computation on the CTT inputs."""
    r4.RUNS, r4.PLUGIN, r4.MLE_SOURCE = RUNS, PLUGIN, MLE_SOURCE
    r4.FIG_LADDER, r4.FIG_STEPS = FIG_LADDER, FIG_STEPS
    result = r4.compute(r4.load())
    result["mle_published"] = cross_study.rl_baseline("ctt")
    return result


def plot_both(ctt: dict, hattori: dict) -> None:
    """Both ladders on one axis: rungs on x, one line per forager."""
    fig, ax = plt.subplots(figsize=fs.mm(fs.fit_width(groups=4), 60))
    xs = np.arange(len(r4.RUNGS))
    for name, r, color in (("HB-Hattori (r4)", hattori, fs.OKABE_ITO["sky"]),
                           ("HB-CTT", ctt, fs.OKABE_ITO["vermillion"])):
        ys = [r["pooled"][k] for k in r4.RUNGS]
        ax.plot(xs, ys, "o-", color=color, ms=3, lw=0.8, label=name)
        ax.annotate(f"{ys[-1]:.4f}", xy=(xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=6, color=color)
    gru = list(ctt["gru_by_d"].items())
    for k, (_, v) in enumerate(gru):
        ax.axhline(v, color=fs.OKABE_ITO["blue"], lw=0.5, ls=(0, (3, 2)),
                   alpha=0.3 + 0.7 * k / (len(gru) - 1), zorder=0)
    ax.annotate(f"GRU, D≈{gru[0][0]} (lowest line) to D≈{gru[-1][0]}",
                xy=(0, ctt["gru_by_d"][gru[-1][0]]), xycoords=("axes fraction", "data"), xytext=(2, 1.5),
                textcoords="offset points", color=fs.OKABE_ITO["blue"], va="bottom", fontsize=6)
    ax.set_xticks(xs, r4.RUNG_TICK)
    ax.set_xlim(-0.4, len(xs) - 0.2)
    ax.set_ylim(0.700, 0.730)
    ax.set_ylabel(f"Held-out likelihood\n({ctt['n_subjects']} subjects, trial-weighted)")
    ax.legend(loc="lower right", frameon=False)
    fs.save_figure(fig, FIG_BOTH)
    plt.close(fig)


def render(r: dict, hattori: dict) -> str:
    """The result-1 block."""
    lines = [f"![both ladders](../{FIG_BOTH.name})", "",
             "*Both ladders, rung by rung, on the same subjects and sessions; dashed lines, the GRU H128 "
             "curve (study 01 r1).*", "",
             "| rung | HB-CTT | step | HB-Hattori (r4) | step |", "|---|---|---|---|---|"]
    prev_c = prev_h = None
    for rung in r4.RUNGS:
        c, h = r["pooled"][rung], hattori["pooled"][rung]
        sc = "" if prev_c is None else f"{c - prev_c:+.4f}"
        sh = "" if prev_h is None else f"{h - prev_h:+.4f}"
        lines.append(f"| {rung} | {c:.4f} | {sc} | {h:.4f} | {sh} |")
        prev_c, prev_h = c, h
    lines += ["", f"MLE CTT pooled from study 08's per-subject table: {r['pooled']['MLE']:.4f} "
              f"(study 01 r8 publishes {r['mle_published']:.4f}).", "",
              f"![steps](../{FIG_STEPS.name})", "",
              "*Each HB-CTT step, subject by subject (paired), with the share of subjects that gain.*", "",
              "| HB-CTT step, per subject | median | subjects with a gain |", "|---|---|---|"]
    for name, st in r["steps"].items():
        lines.append(f"| {name} | {st['median']:+.4f} | {st['frac_positive']:.0%} "
                     f"({round(st['frac_positive'] * st['n'])}/{st['n']}) |")
    lines += ["", f"![ladder](../{FIG_LADDER.name})", "",
              "*The HB-CTT ladder alone, in r4's layout.*"]
    return "\n".join(lines)


def main() -> None:
    """Compute, plot, rewrite the result block, and write the JSON."""
    fs.apply_paper_style()
    result = _ctt_ladder()
    hattori = json.loads(HATTORI_LADDER.read_text())
    r4.plot_ladder(result)
    r4.plot_steps(result)
    plot_both(result, hattori)
    for st in result["steps"].values():
        st.pop("_values")
    begin, end = "<!-- BEGIN result-1 -->", "<!-- END result-1 -->"
    head, rest = REPORT.read_text().split(begin)
    _, tail = rest.split(end)
    REPORT.write_text(f"{head}{begin}\n{render(result, hattori)}\n{end}{tail}")
    meta = build_meta("analysis/r6_ctt_ladder.py", WANDB_GROUPS, study_root=HERE.parent,
                      code=[__file__, r4.__file__],
                      inputs=[RUNS, PLUGIN, MLE_SOURCE, HATTORI_LADDER, r4.GRU_SOURCE,
                              cross_study.RL_SOURCE, cross_study.GRU_CURVE_SOURCE])
    write_artifact(HERE / "r6_ctt_ladder.json", {"_meta": meta, **result},
                   outputs=[FIG_LADDER, FIG_STEPS, FIG_BOTH, HERE / "reports" / "r6-ctt-ladder.md#result-1"])


if __name__ == "__main__":
    main()
