"""Producer for r3: the literature models as 2-level EM, against their per-subject fits (r1).

Reads the frozen 2-level EM rows (``two_level_em_fits.csv``, written by ``freeze_two_level_em.py``)
and the frozen per-subject fits (``per_subject_fits.csv``), r1's ranking and model families, and,
through ``studies/util``, the 2-level HB references (study 13 r5) and the GRU (study 01).

Scores pool over subjects, exp(Σ log p / Σ n), on the odd (score) sessions. Every (model, subject)
row must have the same fit and scored trial counts in both freezes, so "2-level EM minus per
subject" compares the same trials. Because 17 of the 19 models have no 2-level HB reference, each
model also gets three checks: the median weight ESS of its session marginals, the share of subjects
whose EM stopped at the iteration cap, and its score minus the per-subject score.

- result-1: the 2-level EM leaderboard on the 149 held-out subjects (figure + table).
- result-2: the per-model checks, on all 763 subjects.
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

EM = HERE / "two_level_em_fits.csv"
EM_RUNS = HERE / "two_level_em_fits.json"
PS = csl.LITERATURE_SOURCE                       # per_subject_fits.csv (r1's freeze)
R1 = HERE / "r1_leaderboard.json"                # ranking and model families
WANDB_GROUPS: list[str] = []                     # no W&B runs (see the freeze)
REPORT = HERE / "reports" / "r3-2level-em.md"
FIG = HERE / "fig_r3_2level_em.svg"
GRU_D = 614
AXIS_FLOOR = 0.69   # the axis starts here; models below it are written, not drawn
TWO_LEVEL_EQUATIONS = r"""

**2-level EM layer** (regime 2; the per-session model above is unchanged):

$$x_{sk} \sim \mathcal{N}\big(\mu_s,\ \operatorname{diag}(\sigma_s^2)\big), \qquad \theta_{sk} = \ell + (h - \ell)\,\Phi(x_{sk}), \qquad k = \text{fit session of subject } s$$

Each bounded parameter $\theta$ with fit bounds $[\ell, h]$ lives on a probit scale $x$ ($\Phi$ the standard normal CDF, so $x \sim \mathcal{N}(0,1)$ is uniform on the bounds). Study 13's fixed hyperprior: $\mu_s \sim \mathcal{N}(0, 1)$, $\log \sigma_s \sim \mathcal{N}(-1, 1)$. Fit by Laplace-EM (Huys et al. 2011), which integrates each session's parameters out: the E-step finds each fit session's best $x_{sk}$ and a Gaussian approximation of its uncertainty; the M-step updates $\mu_s, \sigma_s$; EM runs to convergence (at most 100 iterations). A held-out session $j$ is scored by the session marginal $p(c_j) = \mathbb{E}_{x \sim \mathcal{N}(\mu_s, \sigma_s^2)}\, p(c_j \mid \theta(x))$ over 2,048 draws. Implementation: MEvolve `mevolve.score` (ADR 0004)."""


def _pooled(rows) -> float:
    return math.exp(sum(r[0] for r in rows) / sum(r[1] for r in rows))


def _read(path: Path) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    with path.open(newline="") as handle:
        for r in csv.DictReader(handle):
            out.setdefault(r["literature_model"], {})[r["subject_id"]] = r
    return out


def compute() -> dict:
    em, ps = _read(EM), _read(PS)
    r1 = json.loads(R1.read_text())
    ranked = [row["literature_model"] for row in r1["rows"]]
    family = {m: r1["by_model"][m]["family"] for m in ranked}
    assert set(ranked) <= set(em), f"no 2-level EM rows for {sorted(set(ranked) - set(em))}"

    # The same trials in both freezes, for every subject of every model, or stop.
    for m in ranked:
        assert set(em[m]) == set(ps[m]), f"{m}: subject sets differ"
        for s, e in em[m].items():
            p = ps[m][s]
            assert int(e["n_trials"]) == int(p["eval_total_trials"]), f"{m} {s}: scored trials"
            assert int(e["n_fit_trials"]) == int(p["train_total_trials"]), f"{m} {s}: fit trials"
            assert e["pool"] == p["cohort"], f"{m} {s}: pool {e['pool']} vs cohort {p['cohort']}"

    gru = csl.gru_per_subject(GRU_D)
    gru_list = csl.gru_split()["test"]
    n_seeds = len(next(iter(gru.values())))
    gru_on_list = statistics.fmean(
        math.exp(sum(gru[s][k][1] * math.log(gru[s][k][0]) for s in gru_list)
                 / sum(gru[s][k][1] for s in gru_list)) for k in range(n_seeds))
    runs = json.loads(EM_RUNS.read_text())

    rows = []
    for m in ranked:
        every = list(em[m])
        held = [s for s in every if em[m][s]["pool"] == "heldout"]
        e_ll = {s: (float(em[m][s]["log_lik"]), int(em[m][s]["n_trials"])) for s in every}
        p_ll = {s: (float(ps[m][s]["eval_total_log_likelihood"]), int(ps[m][s]["eval_total_trials"]))
                for s in every}
        ess = [float(em[m][s]["ess_median"]) for s in every if em[m][s]["ess_median"] != ""]
        row = {"literature_model": m, "label": f"EM-{m}", "family": family[m],
               "pooled_heldout": _pooled([e_ll[s] for s in held]),
               "pooled_heldout_per_subject": _pooled([p_ll[s] for s in held]),
               "pooled_gru_list": _pooled([e_ll[s] for s in gru_list]),
               "pooled_all": _pooled([e_ll[s] for s in every]),
               "pooled_all_per_subject": _pooled([p_ll[s] for s in every]),
               "median_ess": statistics.median(ess) if ess else None,
               "share_at_em_cap": sum(em[m][s]["converged"] != "True" for s in every) / len(every),
               "share_below_per_subject": sum(e_ll[s][0] < p_ll[s][0] for s in every) / len(every),
               "n_subjects": len(every), "n_heldout": len(held)}
        row["gain_heldout"] = row["pooled_heldout"] - row["pooled_heldout_per_subject"]
        row["gain_all"] = row["pooled_all"] - row["pooled_all_per_subject"]
        rows.append(row)
    # The headline ranking (Han, 2026-11-02): 2-level EM, except that a model whose 2-level fit failed
    # (it scores below its own per-subject fit) is placed by its per-subject score. em_rank keeps the
    # pure 2-level EM order.
    rows.sort(key=lambda r: -r["pooled_heldout"])
    for i, row in enumerate(rows, 1):
        failed = row["gain_all"] < 0
        row["em_rank"] = i
        row["headline_score"] = row["pooled_heldout_per_subject"] if failed else row["pooled_heldout"]
        row["headline_source"] = "per subject (2-level fit failed)" if failed else "2-level EM"
    rows.sort(key=lambda r: -r["headline_score"])
    return {"n_models": len(rows), "n_subjects": rows[0]["n_subjects"],
            "n_heldout": rows[0]["n_heldout"], "n_gru_list": len(gru_list),
            "settings": runs["settings"], "mevolve_git_sha": runs["mevolve_git_sha"],
            "reference": {"hb_ctt": csl.hb_pooled("ctt"), "hb_hattori": csl.hb_pooled("hattori"),
                          "gru_d614_own_list": cs.gru_h128_curve()[GRU_D],
                          "gru_d614_gru_list": gru_on_list},
            "rows": rows}


def figure(r: dict) -> None:
    rows = r["rows"][::-1]                       # best at the top
    y = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=fs.mm(140, 22 + 4.2 * len(rows)))
    ref = r["reference"]
    on = [row for row in rows if row["pooled_heldout_per_subject"] >= AXIS_FLOOR]
    lo = min([row["pooled_heldout_per_subject"] for row in on] + [ref["hb_hattori"]]) - 0.002
    hi = max(ref["gru_d614_own_list"], max(row["pooled_heldout"] for row in rows)) + 0.003
    ax.set_xlim(lo, hi)
    first = True
    for i, row in zip(y, rows):
        a, b = row["pooled_heldout_per_subject"], row["pooled_heldout"]
        if a < AXIS_FLOOR:   # off scale: both values written in grey
            ax.text(lo + 0.0006, i, f"{a:.4f} → {b:.4f}", va="center", color="0.4", in_layout=False)
            continue
        ax.plot([a, b], [i, i], color="0.6", lw=0.8, zorder=1)
        ax.scatter([a], [i], s=10, facecolors="white", edgecolors="k", zorder=2,
                   label="per subject (r1)" if first else None)
        ax.scatter([b], [i], s=10, color="k", zorder=3, label="2-level EM" if first else None)
        first = False
    for value, name, color in [(ref["hb_hattori"], "2-level HB Hattori (study 13)", fs.OKABE_ITO["orange"]),
                               (ref["hb_ctt"], "2-level HB CTT (study 13)", fs.OKABE_ITO["vermillion"]),
                               (ref["gru_d614_own_list"], "GRU D≈614 (study 01)", fs.OKABE_ITO["blue"])]:
        ax.axvline(value, color=color, lw=0.8, ls="--", zorder=0, label=name)
    ax.set_ylim(-0.8, len(rows) - 0.4)
    ax.set_yticks(y, [row["label"] + (" *" if row["gain_all"] < 0 else "") for row in rows])
    ax.set_xlabel(f"Pooled held-out likelihood ({r['n_heldout']} subjects)")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=2, frameon=False,
               handlelength=1.5)
    fs.save_figure(fig, FIG)


def render(r: dict) -> str:
    ref, s = r["reference"], r["settings"]
    lines = [
        f"![2-level EM](../{FIG.name})",
        "",
        f"Each line runs from a literature model's per-subject score (open circle, r1) to its 2-level EM "
        f"score (filled) on the {r['n_heldout']} held-out subjects. 2-level EM: Laplace-EM within each "
        f"subject under study 13's hyperprior, EM to convergence (max_em = {s['max_em']}), held-out "
        f"sessions scored by the session marginal over {s['n_draws']} draws (MEvolve "
        f"`{r['mevolve_git_sha'][:7]}`). Dashed lines: 2-level HB (study 13) and the GRU at D≈614 "
        f"(on its own held-out list). Models whose per-subject score is below {AXIS_FLOOR} are written "
        f"in grey, not drawn. * marks a model whose 2-level EM score falls below its per-subject score over "
        f"all subjects (a failed Laplace fit; see the checks). One fit per subject and model, so no error bars.",
        "",
        f"Headline ranking: by the 2-level EM score, except that the {sum(row['headline_source'] != '2-level EM' for row in r['rows'])} "
        f"models whose 2-level fit failed (*) are placed by their per-subject score.",
        "",
        f"| rank | literature model | family | headline ({r['n_heldout']}) | 2-level EM | per subject | gain "
        f"| 2-level EM, GRU's list ({r['n_gru_list']}) |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(r["rows"], 1):
        star = " *" if row["headline_source"] != "2-level EM" else ""
        lines.append(f"| {i} | {row['literature_model']}{star} | {row['family']} | {row['headline_score']:.4f} "
                     f"| {row['pooled_heldout']:.4f} | {row['pooled_heldout_per_subject']:.4f} "
                     f"| {fs.num(row['gain_heldout'], '+.4f')} | {row['pooled_gru_list']:.4f} |")
    lines += [
        f"| | *2-level HB CTT (study 13)* | | {ref['hb_ctt']:.4f} | | | | |",
        f"| | *2-level HB Hattori (study 13)* | | {ref['hb_hattori']:.4f} | | | | |",
        f"| | *GRU D≈614 (study 01)* | | {ref['gru_d614_own_list']:.4f}† | | | | "
        f"{ref['gru_d614_gru_list']:.4f} |",
        "",
        "† On the GRU's own held-out list. Every subject's fit and scored trial counts equal r1's, so the "
        "gain compares the same trials.",
    ]
    return "\n".join(lines)


def render_checks(r: dict) -> str:
    lines = [f"On all {r['n_subjects']} subjects. Only CompareToThreshold and Hattori2019 have a 2-level HB "
             "reference (gate 1c); for the others these checks stand in. A 2-level EM score below the "
             "per-subject score would mark a failed Laplace fit.",
             "",
             f"| 2-level EM model | median weight ESS | subjects at EM cap | 2-level EM − per subject "
             f"({r['n_subjects']}) | subjects scoring below per subject |",
             "|---|---:|---:|---:|---:|"]
    for row in sorted(r["rows"], key=lambda x: x["literature_model"]):
        ess = f"{row['median_ess']:.0f}" if row["median_ess"] is not None else "—"
        lines.append(f"| {row['label']} | {ess} | {row['share_at_em_cap']:.1%} "
                     f"| {fs.num(row['gain_all'], '+.4f')} | {row['share_below_per_subject']:.1%} |")
    lines += ["", "EM-Rescorla-Wagner: its epsilon-greedy likelihood is flat or stepped in some parameters, "
              "and the per-session Laplace step uses gradients, so it can report convergence while poorly "
              "fitted. Read its row with that caveat."]
    return "\n".join(lines)


def _fit_flag(row: dict) -> dict | None:
    """Whether to distrust a model's 2-level EM score, and why (the atlas shows it as is).

    laplace_failed: pooled over all subjects, 2-level EM scores below the same model's per-subject
    fit, which adds flexibility and so should not lose; caveat: a likelihood the gradient-based
    per-session step cannot fit reliably (Rescorla-Wagner's epsilon-greedy choice).
    """
    if row["gain_all"] < 0:
        return {"kind": "laplace_failed",
                "reason": f"2-level EM scores {row['gain_all']:+.4f} below the per-subject fit over all "
                          f"subjects ({row['share_below_per_subject']:.0%} of subjects lower)"}
    if row["literature_model"] == "Rescorla-Wagner":
        return {"kind": "caveat", "reason": "epsilon-greedy likelihood is stepped, so the gradient-based "
                                            "per-session step can report convergence while poorly fitted"}
    return None


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
    REPORT.write_text(_rewrite(text, "result-2", render_checks(result)))
    # Atlas leaderboard fields (atlas PR 43, a twin overlay on r1's panel): family and summary
    # required, rank an int (the headline rank; em_rank is the pure 2-level EM order), headline_score
    # and headline_source, pooled_heldout (the 2-level EM score) finite, literature_model equal to a ranked r1 by_model
    # key (a missing twin fails the build), fit_flag null or {kind, reason}; plus the report's
    # result-1 block. Renaming any of these fails the atlas build; tell the atlas session first.
    cards = json.loads(R1.read_text())["by_model"]
    result["by_model"] = {}
    for i, row in enumerate(result["rows"], 1):
        card = cards[row["literature_model"]]   # every EM- model has its r1 twin (keyed by name)
        result["by_model"][row["label"]] = {
            "rank": i, **row,
            "summary": (f"{row['literature_model']} with session-to-session variation: each session's "
                        "parameters vary around the subject's typical values (2-level EM). "
                        + card["summary"]),
            "equations": card["equations"] + TWO_LEVEL_EQUATIONS,
            "paper": card["paper"], "code": card["code"],
            "family_note": card.get("family_note", ""),
            "fit_flag": _fit_flag(row)}
    meta = build_meta("analysis/r3_2level_em.py", WANDB_GROUPS, study_root=HERE.parent,
                      code=[__file__], inputs=[EM, EM_RUNS, PS, R1, csl.HB_CTT_SOURCE,
                                               cs.GRU_CURVE_SOURCE, csl.GRU_PER_SUBJECT_SOURCE,
                                               csl.GRU_SPLIT_SOURCE])
    write_artifact(HERE / "r3_2level_em.json", {"_meta": meta, **result},
                   outputs=[FIG, f"{REPORT}#result-1", f"{REPORT}#result-2"])


if __name__ == "__main__":
    main()
