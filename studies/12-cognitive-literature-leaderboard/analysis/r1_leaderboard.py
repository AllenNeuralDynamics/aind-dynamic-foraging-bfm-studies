"""Producer for r1: the leaderboard of literature models on the held-out subjects.

Reads the frozen per-subject fits (``per_subject_fits.csv``, written by ``freeze_fits.py``) and,
through ``studies/util/cross_study.py`` and ``cross_study_literature.py``, the reference values of other studies: HB-CTT and
HB-Hattori2019 (study 13 r5), the H128 GRU's curve (study 01 r1) and per-subject values (study 01
r2), and study 01 r8's per-subject MLE fits of Hattori2019, Bari2019 and CompareToThreshold.

Scores pool over subjects: exp(Σ log p / Σ n) over the odd (eval) sessions. A subject's own
likelihood is exp(log p / n). Fits are single-seed, so there are no run-to-run error bars.

- result-1: the ranked leaderboard on the 149 held-out subjects and its paired gap to the GRU
  on the 146 subjects also on the GRU's held-out list (figure + table).
- result-2: the re-fit Hattori2019, Bari2019 and CompareToThreshold against study 01 r8.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
import cross_study as cs  # noqa: E402
import cross_study_literature as csl  # noqa: E402
import figure_style as fs  # noqa: E402
from _meta import build_meta, write_artifact  # noqa: E402

FITS = csl.LITERATURE_SOURCE                       # analysis/per_subject_fits.csv
RUNS = HERE / "per_subject_fits.json"             # its run table and source digests
WANDB_GROUPS = json.loads(RUNS.read_text())["_meta"]["wandb_groups"]   # pinned in the freeze
REPORT = HERE / "reports" / "r1-leaderboard.md"
FIG = HERE / "fig_r1_leaderboard.svg"
GRU_D = 614
R8 = {"Hattori2019": "hattori", "Bari2019": "bari", "CompareToThreshold": "ctt"}
TOL = 1e-4   # per-subject |Δ likelihood| that counts as reproducing study 01 r8
AXIS_FLOOR = 0.69   # the leaderboard axis starts here; models below it are drawn off scale
# Fitted and frozen but not ranked: a strawman with no lapse term, so its deterministic policy
# scores every wrong call at the probability clip (Han, 2026-11-01).
NOT_RANKED = {"Win-Stay-Lose-Shift": "strawman: deterministic policy, no lapse term"}


def _pooled(scores: dict[str, tuple[float, int]], subjects) -> float:
    return math.exp(sum(scores[s][0] for s in subjects) / sum(scores[s][1] for s in subjects))


def _own(score: tuple[float, int]) -> float:
    return math.exp(score[0] / score[1])


def compute() -> dict:
    heldout = csl.literature_per_subject("heldout")
    train = csl.literature_per_subject("train")
    every = csl.literature_per_subject()            # all 763, for the GRU's own test list
    with FITS.open(newline="") as handle:
        n_params = {r["literature_model"]: int(r["n_free_params"]) for r in csv.DictReader(handle)}
    subjects = sorted(next(iter(heldout.values())))
    gru = csl.gru_per_subject(GRU_D)
    shared = sorted(set(subjects) & set(gru))
    gru_own = {s: statistics.fmean(v for v, _ in gru[s]) for s in shared}
    n_seeds = len(next(iter(gru.values())))
    gru_list = csl.gru_split()["test"]
    assert set(gru_list) == set(gru), "study 01's per-subject GRU rows are not its test list"

    def gru_pooled(over):
        return statistics.fmean(
            math.exp(sum(gru[s][k][1] * math.log(gru[s][k][0]) for s in over)
                     / sum(gru[s][k][1] for s in over)) for k in range(n_seeds))
    any_model = next(iter(heldout.values()))
    trials_differ = sorted(s for s in shared if gru[s][0][1] != any_model[s][1])

    rows = []
    for model, scores in heldout.items():
        deltas = [_own(scores[s]) - gru_own[s] for s in shared]
        rows.append({"literature_model": model, "n_free_params": n_params[model],
                     "pooled_heldout": _pooled(scores, subjects),
                     "median_heldout": statistics.median(_own(scores[s]) for s in subjects),
                     "pooled_train": _pooled(train[model], sorted(train[model])),
                     "pooled_shared": _pooled(scores, shared),
                     "pooled_gru_list": _pooled(every[model], gru_list),
                     "minus_gru_median": statistics.median(deltas),
                     "frac_gru_higher": sum(d < 0 for d in deltas) / len(deltas),
                     "_deltas": deltas})
    rows.sort(key=lambda r: -r["pooled_heldout"])
    not_ranked = [r for r in rows if r["literature_model"] in NOT_RANKED]
    rows = [r for r in rows if r["literature_model"] not in NOT_RANKED]

    repro = {}
    r8 = json.loads(cs.RL_SOURCE.read_text())["models"]
    for model, key in R8.items():
        if model not in heldout:
            continue
        old = {str(s): float(v) for s, v in r8[key]["per_subject_likelihood"].items()}
        common = sorted(set(old) & set(heldout[model]))
        diff = [abs(_own(heldout[model][s]) - old[s]) for s in common]
        repro[model] = {"n_subjects": len(common), "max_abs_delta": max(diff),
                        "n_over_tolerance": sum(d > TOL for d in diff),
                        "pooled_new": round(_pooled(heldout[model], subjects), 4),
                        "pooled_r8": cs.rl_baseline(key)}
    return {"n_subjects_heldout": len(subjects), "n_subjects_shared": len(shared),
            "n_subjects_gru_list": len(gru_list),
            "n_subjects_train": len(next(iter(train.values()))),
            "n_trials_heldout": sum(any_model[s][1] for s in subjects),
            "shared_trials_differ_from_gru": trials_differ,
            "reference": {"hb_ctt": csl.hb_pooled("ctt"), "hb_hattori": csl.hb_pooled("hattori"),
                          "gru_d614_own_list": cs.gru_h128_curve()[GRU_D],
                          "gru_d614_shared": gru_pooled(shared),
                          "gru_d614_gru_list": gru_pooled(gru_list)},
            "tolerance": TOL, "rows": rows, "reproduction": repro,
            "not_ranked": [{"literature_model": r["literature_model"],
                            "reason": NOT_RANKED[r["literature_model"]],
                            "pooled_heldout": r["pooled_heldout"]} for r in not_ranked]}


def figure(r: dict) -> None:
    rows = r["rows"][::-1]                       # best at the top
    y = list(range(len(rows)))
    fig, (a, b) = plt.subplots(1, 2, sharey=True, figsize=fs.mm(150, 18 + 4.2 * len(rows)),
                               gridspec_kw={"wspace": 0.08})
    on = [(i, row) for i, row in zip(y, rows) if row["pooled_heldout"] >= AXIS_FLOOR]
    off = [(i, row) for i, row in zip(y, rows) if row["pooled_heldout"] < AXIS_FLOOR]
    a.scatter([row["pooled_heldout"] for _, row in on], [i for i, _ in on], s=10, color="k", zorder=3)
    ref = r["reference"]
    lo = min([row["pooled_heldout"] for _, row in on] + [ref["hb_hattori"]]) - 0.002
    hi = max(ref["gru_d614_own_list"], max(row["pooled_heldout"] for _, row in on)) + 0.003
    a.set_xlim(lo, hi)
    for i, row in off:   # off scale: a marker at the left edge with the value beside it
        a.scatter([lo], [i], s=14, marker="<", color="0.5", clip_on=False, zorder=3)
        a.text(lo + 0.0006, i, f"{row['pooled_heldout']:.4f}", va="center", color="0.4",
               in_layout=False)
    for value, name, color in [(ref["hb_hattori"], "HB-Hattori (study 13)", fs.OKABE_ITO["orange"]),
                               (ref["hb_ctt"], "HB-CTT (study 13)", fs.OKABE_ITO["vermillion"]),
                               (ref["gru_d614_own_list"], "GRU D≈614 (study 01)", fs.OKABE_ITO["blue"])]:
        a.axvline(value, color=color, lw=0.8, ls="--", zorder=1, label=name)
    fig.legend(*a.get_legend_handles_labels(), loc="outside upper center", ncol=3, frameon=False,
               handlelength=1.5)
    a.set_yticks(y, [f"{row['literature_model']} ({row['n_free_params']})" for row in rows])
    a.set_xlabel(f"Pooled held-out likelihood ({r['n_subjects_heldout']} subjects)")
    fs.panel_label(a, "a", dx_pt=-130)
    b.boxplot([row["_deltas"] if row["pooled_heldout"] >= AXIS_FLOOR else [] for row in rows],
              positions=y, vert=False, widths=0.6, showfliers=False, medianprops={"color": "k"},
              manage_ticks=False)
    for i, row in off:
        b.text(0.02, i, f"median {fs.num(row['minus_gru_median'], '+.3f')}",
               transform=b.get_yaxis_transform(), va="center", color="0.4", in_layout=False)
    b.axvline(0, color=fs.OKABE_ITO["blue"], lw=0.8, ls="--")
    b.set_xlim(right=0.01)
    b.xaxis.set_major_locator(plt.MaxNLocator(4))
    b.set_xlabel(f"Model − GRU per subject (n = {r['n_subjects_shared']})")
    fs.panel_label(b, "b")
    fs.save_figure(fig, FIG)


def render(r: dict) -> str:
    ref, best = r["reference"], r["rows"][0]
    lines = [
        f"![Leaderboard](../{FIG.name})",
        "",
        f"**a**, pooled held-out likelihood of each literature model (free parameters in brackets) "
        f"on the {r['n_subjects_heldout']} held-out subjects ({r['n_trials_heldout']:,} scored "
        f"trials); dashed lines: HB-CTT and HB-Hattori (study 13 r5, same subjects) and the GRU at "
        f"D≈614 (study 01 r1, on its own held-out list). **b**, per-subject likelihood minus the "
        f"GRU's (3-seed mean) on the {r['n_subjects_shared']} subjects on both lists; boxes are "
        f"quartiles, whiskers 1.5 IQR. Models below {AXIS_FLOOR} are drawn off scale (grey) with "
        f"their value. One fit per subject and model, so no error bars.",
        "",
        f"| rank | literature model | params | held-out ({r['n_subjects_heldout']}) | median subject "
        f"| training pool ({r['n_subjects_train']}) | shared ({r['n_subjects_shared']}) "
        f"| GRU's list ({r['n_subjects_gru_list']}) | median − GRU | GRU higher |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(r["rows"], 1):
        lines.append(
            f"| {i} | {row['literature_model']} | {row['n_free_params']} | {row['pooled_heldout']:.4f} "
            f"| {row['median_heldout']:.4f} | {row['pooled_train']:.4f} | {row['pooled_shared']:.4f} "
            f"| {row['pooled_gru_list']:.4f} "
            f"| {fs.num(row['minus_gru_median'], '+.4f')} | {row['frac_gru_higher']:.0%} |")
    lines += [
        f"| | *HB-CTT (study 13)* | | {ref['hb_ctt']:.4f} | | | | | | |",
        f"| | *HB-Hattori (study 13)* | | {ref['hb_hattori']:.4f} | | | | | | |",
        f"| | *GRU D≈614 (study 01)* | | {ref['gru_d614_own_list']:.4f}† | | | "
        f"{ref['gru_d614_shared']:.4f} | {ref['gru_d614_gru_list']:.4f} | | |",
        "",
        f"† On the GRU's own held-out list, which shares {r['n_subjects_shared']} of the "
        f"{r['n_subjects_heldout']} subjects. Shared subjects whose scored-trial count differs from "
        f"the GRU's: {len(r['shared_trials_differ_from_gru'])}. \"GRU's list\" pools each literature "
        f"model over the GRU's own {r['n_subjects_gru_list']} test subjects; the 3 of them in our "
        f"training pool are fit per subject like the rest, so the pool does not change their score.",
        "",
        f"Best literature model: **{best['literature_model']}**, "
        f"{best['pooled_heldout']:.4f}, which is {fs.num(best['pooled_heldout'] - ref['hb_ctt'], '+.4f')} "
        f"from HB-CTT and {fs.num(best['pooled_shared'] - ref['gru_d614_shared'], '+.4f')} from the "
        f"GRU on the shared subjects.",
    ]
    for nr in r["not_ranked"]:
        lines += ["", f"Fitted but not ranked: {nr['literature_model']} ({nr['reason']}); pooled "
                      f"held-out likelihood {nr['pooled_heldout']:.4f}. Its fits are in the freeze."]
    return "\n".join(lines)


def render_repro(r: dict) -> str:
    lines = [f"Per-subject held-out likelihood of the re-fit against study 01 r8's fit of the same "
             f"model (seed 0, same data and protocol). Tolerance: |Δ| ≤ {r['tolerance']:g} per subject.",
             "",
             "| literature model | subjects | max abs Δ | over tolerance | pooled, re-fit | pooled, r8 |",
             "|---|---:|---:|---:|---:|---:|"]
    for model, v in r["reproduction"].items():
        lines.append(f"| {model} | {v['n_subjects']} | {v['max_abs_delta']:.2e} | "
                     f"{v['n_over_tolerance']} | {v['pooled_new']:.4f} | {v['pooled_r8']:.4f} |")
    return "\n".join(lines)


def _rewrite(text: str, block: str, body: str) -> str:
    begin, end = f"<!-- BEGIN {block} -->", f"<!-- END {block} -->"
    head, rest = text.split(begin)
    _, tail = rest.split(end)
    return f"{head}{begin}\n{body}\n{end}{tail}"


def main() -> None:
    fs.apply_paper_style()
    result = compute()
    figure(result)
    text = _rewrite(REPORT.read_text(), "result-1", render(result))
    REPORT.write_text(_rewrite(text, "result-2", render_repro(result)))
    for row in result["rows"]:
        del row["_deltas"]
    # Stable JSON pointers for readers (the atlas): /by_model/<literature model>/pooled_heldout
    result["by_model"] = {row["literature_model"]: {"rank": i, **row}
                          for i, row in enumerate(result["rows"], 1)}
    meta = build_meta("analysis/r1_leaderboard.py", WANDB_GROUPS, study_root=HERE.parent,
                      code=[__file__], inputs=[FITS, RUNS, csl.HB_CTT_SOURCE, cs.GRU_CURVE_SOURCE,
                                               csl.GRU_PER_SUBJECT_SOURCE, cs.RL_SOURCE,
                                               csl.GRU_SPLIT_SOURCE])
    write_artifact(HERE / "r1_leaderboard.json", {"_meta": meta, **result},
                   outputs=[FIG, f"{REPORT}#result-1", f"{REPORT}#result-2"])


if __name__ == "__main__":
    main()
