"""Producer for r2: per-subject held-out likelihood, HB-Hattori2019 two-level vs GRU vs per-subject MLE.

Offline. HB per subject is each run's ``heldout/eval_likelihood`` (session-marginal, trial-weighted
within the subject) from the frozen ``hattori_final_runs.json``. The GRU (H128, v1; the curve r1 quotes)
and the per-subject MLE are read from study 01's committed JSON (declared as inputs and upstream;
not via ``cross_study`` so that adding a reader there does not stale other studies' artifacts).
The GRU per subject is the mean over its 3 seeds at each D. Comparisons are paired by subject, so
between-subject spread, which dwarfs the gaps between models, cancels out.
"""
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
from _meta import build_meta, write_artifact  # noqa: E402
import figure_style as fs  # noqa: E402

S01 = HERE.parents[1] / "01-gru-scaling-law" / "analysis"
GRU_PER_SUBJECT_SOURCE = S01 / "per_subject_rows.json"   # study 01 r2: per-held-out-subject GRU rows
RL_SOURCE = S01 / "rl_baseline.json"                     # study 01 r8: per-mouse RL baselines
RATIO_TO_D = {0.016: 10, 0.049: 30, 0.163: 100, 0.489: 300, 1.0: 614}   # study 01's subject_ratio dial


def _gru_per_subject(variant):
    """Study 01's per-subject GRU likelihoods: ``{D: {subject: [(likelihood, n_trials) per seed]}}``."""
    out = {}
    for row in json.loads(GRU_PER_SUBJECT_SOURCE.read_text())["rows"]:
        if row["variant"] == variant:
            out.setdefault(RATIO_TO_D[row["ratio"]], {}).setdefault(str(row["subject"]), []).append(
                (row["ll"], row["n_trials"]))
    return out


def _rl_per_subject(model):
    """Study 01 r8's per-subject MLE likelihood for one forager family."""
    table = json.loads(RL_SOURCE.read_text())["models"][model]["per_subject_likelihood"]
    return {str(subject): float(value) for subject, value in table.items()}

WANDB_GROUPS = ["hattori@20261027-181430", "hattori@20261027-201810"]   # final set (ADR 0004), via the freeze
RUNS = HERE / "hattori_final_runs.json"   # production fits, k̂-rescued (ADR 0004)
REPORT = HERE / "reports" / "r2-per-subject.md"
FIG = HERE / "fig_r2_per_subject.svg"
GRU_D = (30, 614)


def _arms() -> tuple[dict[str, dict[str, float]], list[str]]:
    """Per-subject likelihood for every arm, plus subjects whose HB and GRU trial counts differ."""
    hb_rows = [r for r in json.loads(RUNS.read_text())["rows"] if r["state"] == "finished"]
    hb = {r["subject_id"]: r["heldout/eval_likelihood"] for r in hb_rows}
    hb_trials = {r["subject_id"]: r["heldout/num_eval_trials"] for r in hb_rows}
    gru_all = _gru_per_subject("v1")
    arms = {"HB two-level": hb}
    mismatch = set()
    for d in GRU_D:
        arms[f"GRU D≈{d}"] = {s: statistics.fmean(v for v, _ in seeds)
                              for s, seeds in gru_all[d].items()}
        mismatch |= {s for s, seeds in gru_all[d].items() if seeds[0][1] != hb_trials.get(s)}
    arms["MLE Hattori"] = _rl_per_subject("hattori")
    arms["MLE CTT"] = _rl_per_subject("ctt")
    common = set.intersection(*(set(a) for a in arms.values()))
    # Study 01's GRU held-out cohort and its per-subject MLE cohort differ by 3 subjects each way;
    # HB uses the MLE's. Pair over the subjects every arm scored, and name the rest.
    assert len(common) >= 146, f"only {len(common)} subjects common to every arm"
    unshared = {name: sorted(set(a) - common) for name, a in arms.items()}
    trial_mismatch = sorted(mismatch & common)
    return ({name: {s: a[s] for s in sorted(common)} for name, a in arms.items()},
            {"unshared": unshared, "trial_mismatch": trial_mismatch})


def _summary(diffs: list[float]) -> dict:
    """Median and mean paired difference, and the share of subjects where HB is higher."""
    return {"median": statistics.median(diffs), "mean": statistics.fmean(diffs),
            "frac_hb_higher": sum(d > 0 for d in diffs) / len(diffs), "n": len(diffs)}


def compute() -> tuple[dict, dict]:
    """Per-arm distributions and paired HB-minus-arm summaries."""
    arms, cohort = _arms()
    subjects = list(arms["HB two-level"])
    paired = {name: [arms["HB two-level"][s] - a[s] for s in subjects]
              for name, a in arms.items() if name != "HB two-level"}
    result = {
        "n_subjects": len(subjects),
        "subjects_not_in_every_arm": cohort["unshared"],
        "subjects_trial_count_differs_from_gru": cohort["trial_mismatch"],
        "per_arm_median": {name: statistics.median(a.values()) for name, a in arms.items()},
        "hb_minus": {name: _summary(d) for name, d in paired.items()},
    }
    return result, {"arms": arms, "paired": paired}


ARM_COLOR = {"HB two-level": fs.OKABE_ITO["vermillion"], "GRU D≈30": fs.OKABE_ITO["sky"],
             "GRU D≈614": fs.OKABE_ITO["blue"], "MLE Hattori": "0.45", "MLE CTT": "0.68"}
ARM_TICK = {"HB two-level": "HB\ntwo-level", "GRU D≈30": "GRU\nD≈30", "GRU D≈614": "GRU\nD≈614",
            "MLE Hattori": "MLE\nHattori", "MLE CTT": "MLE\nCTT"}


def plot(data: dict, result: dict) -> None:
    """a: per-subject likelihood per arm. b: per-subject HB minus each arm, share of subjects HB wins."""
    fs.apply_paper_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=fs.mm(fs.fit_width(groups=9, axes=2), 55), width_ratios=[5, 4])
    names = list(data["arms"])
    fs.violins(a, [list(data["arms"][n].values()) for n in names], [ARM_COLOR[n] for n in names],
               [ARM_TICK[n] for n in names])
    a.set_ylabel("Held-out likelihood")
    a.text(1, 1, f"n = {result['n_subjects']} subjects", transform=a.transAxes, ha="right", va="top")
    fs.panel_label(a, "a")
    diff_names = list(data["paired"])
    fs.violins(b, [data["paired"][n] for n in diff_names], [ARM_COLOR[n] for n in diff_names],
               ["vs " + ARM_TICK[n] for n in diff_names])
    b.axhline(0.0, color="0.5", lw=0.5, ls=(0, (3, 2)), zorder=0)
    b.set_ylabel("Paired difference\n(HB − model)")
    for x, n in enumerate(diff_names, start=1):
        b.annotate(f"{result['hb_minus'][n]['frac_hb_higher']:.0%}", xy=(x, 1), xycoords=("data", "axes fraction"),
                   ha="center", va="bottom", fontsize=6)
    b.annotate("HB higher in", xy=(0, 1), xycoords="axes fraction", xytext=(0, 9), textcoords="offset points",
               ha="left", va="bottom", fontsize=6, color="0.35")
    fs.panel_label(b, "b")
    fs.save_figure(fig, FIG)
    plt.close(fig)


def render(r: dict) -> str:
    """The result-1 block."""
    lines = [f"![per-subject comparison](../{FIG.name})", "",
             "| HB − arm | median | mean | subjects where HB is higher |", "|---|---|---|---|"]
    for name, s in r["hb_minus"].items():
        lines.append(f"| {name} | {s['median']:+.4f} | {s['mean']:+.4f} | "
                     f"{s['frac_hb_higher']:.0%} ({round(s['frac_hb_higher'] * s['n'])}/{s['n']}) |")
    lines += ["", "Per-subject medians: " + ", ".join(
        f"{n} {v:.4f}" for n, v in r["per_arm_median"].items()) + ".",
        f"Paired over the {r['n_subjects']} subjects every arm scored. Study 01's GRU held-out cohort "
        "differs from its per-subject MLE cohort (which HB shares) by three subjects each way, left "
        "out here: " + "; ".join(f"{n} only: {', '.join(v)}" for n, v in
                                 r["subjects_not_in_every_arm"].items() if v) + ".",
        f"Shared subjects whose scored-trial count differs between HB and GRU: "
        f"{', '.join(r['subjects_trial_count_differs_from_gru']) or 'none'}."]
    return "\n".join(lines)


def main() -> None:
    """Compute, plot, rewrite the result block, and write the JSON."""
    result, data = compute()
    plot(data, result)
    begin, end = "<!-- BEGIN result-1 -->", "<!-- END result-1 -->"
    head, rest = REPORT.read_text().split(begin)
    _, tail = rest.split(end)
    REPORT.write_text(f"{head}{begin}\n{render(result)}\n{end}{tail}")
    meta = build_meta("analysis/r2_per_subject.py", WANDB_GROUPS, study_root=HERE.parent,
                      code=[__file__], inputs=[RUNS, GRU_PER_SUBJECT_SOURCE, RL_SOURCE])
    write_artifact(HERE / "r2_per_subject.json", {"_meta": meta, **result},
                   outputs=[FIG, HERE / "reports" / "r2-per-subject.md#result-1"])


if __name__ == "__main__":
    main()
