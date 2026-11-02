"""Producer for r2: CompareToThreshold in stay-credit coordinates against the original.

Reads the frozen fits of the ctt-credit launch (``ctt_credit_fits.csv``, written by
``freeze_ctt_credit.py``): the credit form and the original CompareToThreshold, both refit per
subject on one foraging-models commit. r1's CompareToThreshold fit (``per_subject_fits.csv``) is a
second reference. The two forms share one likelihood (c_win = β(1 − θ), c_loss = −βθ), so they
differ only where CompareToThreshold's bound θ ≥ −1 binds.

A subject's fit score is log p / n over its even (fit) sessions; its held-out score is the same
over its odd (eval) sessions. Pooled scores are exp(Σ log p / Σ n) over subjects. Fits are
single-seed, so there are no run-to-run error bars.

- result-1: pooled scores, the bound / not-bound split, the equivalence checks, figure.
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
import figure_style as fs  # noqa: E402
from _meta import build_meta, write_artifact  # noqa: E402

FITS = HERE / "ctt_credit_fits.csv"
RUNS = HERE / "ctt_credit_fits.json"
R1_FITS = HERE / "per_subject_fits.csv"
WANDB_GROUPS = json.loads(RUNS.read_text())["_meta"]["wandb_groups"]   # pinned in the freeze
REPORT = HERE / "reports" / "r2-ctt-credit.md"
FIG = HERE / "fig_r2_ctt_credit.svg"
CREDIT, CTT, CTT_R1 = "CompareToThreshold-credit", "CompareToThreshold", "CompareToThreshold (r1)"
THETA_MIN, CREDIT_BOUND = -1.0, 20.0   # CompareToThreshold's lower θ bound; the credits' |bound|
BOUND_TOL = 1e-6   # θ within this of −1: the optimizer stopped on the bound
TOL = 1e-4         # agreement that counts as the same fit (per trial, and per parameter)


def _load(path: Path, models: dict[str, str]) -> dict:
    out = {}
    with path.open(newline="") as handle:
        for r in csv.DictReader(handle):
            if r["literature_model"] in models:
                out[(models[r["literature_model"]], r["subject_id"])] = {
                    "cohort": r["cohort"], "params": json.loads(r["fitted_params"]),
                    "fit": (float(r["train_total_log_likelihood"]), int(float(r["train_total_trials"]))),
                    "eval": (float(r["eval_total_log_likelihood"]), int(float(r["eval_total_trials"])))}
    return out


def _pooled(rows: list[dict], key: str) -> float:
    return math.exp(sum(r[key][0] for r in rows) / sum(r[key][1] for r in rows))


def _per_trial(score: tuple[float, int]) -> float:
    return score[0] / score[1]


def _summary(values: list[float]) -> dict:
    q1, med, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return {"median": med, "q1": q1, "q3": q3, "n_up": sum(v > 0 for v in values),
            "n_down": sum(v < 0 for v in values)}


def _p99(values: list[float]) -> float:
    return statistics.quantiles([abs(v) for v in values], n=100, method="inclusive")[98]


def compute() -> dict:
    fits = _load(FITS, {CREDIT: CREDIT, CTT: CTT})
    fits.update(_load(R1_FITS, {CTT: CTT_R1}))
    subjects = sorted({s for _, s in fits})
    per = []
    for s in subjects:
        cr, ct, r1 = fits[(CREDIT, s)], fits[(CTT, s)], fits[(CTT_R1, s)]
        assert cr["cohort"] == ct["cohort"] == r1["cohort"], s
        assert cr["fit"][1] == ct["fit"][1] == r1["fit"][1] and cr["eval"][1] == ct["eval"][1], s
        p, q = cr["params"], ct["params"]
        beta = p["stay_credit_win"] - p["stay_credit_loss"]
        per.append({
            "subject_id": s, "cohort": ct["cohort"], "theta": q["threshold"],
            "at_bound": abs(q["threshold"] - THETA_MIN) < BOUND_TOL,
            "at_bound_r1": abs(r1["params"]["threshold"] - THETA_MIN) < BOUND_TOL,
            "implied_theta": -p["stay_credit_loss"] / beta, "implied_beta": beta,
            "credit_at_bound": max(abs(p["stay_credit_win"]), abs(p["stay_credit_loss"])) >= CREDIT_BOUND - 1e-3,
            "d_fit": _per_trial(cr["fit"]) - _per_trial(ct["fit"]),
            "d_eval": _per_trial(cr["eval"]) - _per_trial(ct["eval"]),
            "d_eval_r1": _per_trial(ct["eval"]) - _per_trial(r1["eval"]),
            "d_win": p["stay_credit_win"] - q["softmax_inverse_temperature"] * (1 - q["threshold"]),
            "d_loss": p["stay_credit_loss"] + q["softmax_inverse_temperature"] * q["threshold"],
            "d_lr": p["learn_rate"] - q["learn_rate"], "d_bias": p["biasL"] - q["biasL"]})

    table = []
    for model in (CREDIT, CTT, CTT_R1):
        rows = {c: [fits[(model, r["subject_id"])] for r in per if r["cohort"] == c]
                for c in ("heldout", "train")}
        table.append({"model": model, "pooled_heldout": _pooled(rows["heldout"], "eval"),
                      "pooled_train_pool_eval": _pooled(rows["train"], "eval"),
                      "pooled_fit_all": _pooled(rows["heldout"] + rows["train"], "fit")})
    split = {}
    for name, keep in (("at_bound", True), ("not_at_bound", False)):
        group = [r for r in per if r["at_bound"] == keep]
        split[name] = {"n": len(group), "n_heldout": sum(r["cohort"] == "heldout" for r in group),
                       "d_fit": _summary([r["d_fit"] for r in group]),
                       "d_eval": _summary([r["d_eval"] for r in group]),
                       "n_fit_gain": sum(r["d_fit"] > TOL for r in group)}
    interior = [r for r in per if not r["at_bound"]]
    bound = [r for r in per if r["at_bound"]]
    flips = [{"subject_id": r["subject_id"], "theta": r["theta"], "d_eval": r["d_eval_r1"]}
             for r in per if r["at_bound"] != r["at_bound_r1"]]
    return {
        "n_subjects": len(per), "n_heldout": sum(r["cohort"] == "heldout" for r in per),
        "bound_tol": BOUND_TOL, "tol": TOL, "table": table, "split": split,
        "equivalence_not_at_bound": {
            "p99_abs_d_fit": _p99([r["d_fit"] for r in interior]),
            "n_d_fit_above_tol": [{"subject_id": r["subject_id"], "theta": r["theta"],
                                   "d_fit": r["d_fit"], "d_eval": r["d_eval"]}
                                  for r in interior if abs(r["d_fit"]) > TOL],
            "p99_abs_d_credit_win": _p99([r["d_win"] for r in interior]),
            "p99_abs_d_credit_loss": _p99([r["d_loss"] for r in interior]),
            "p99_abs_d_learn_rate": _p99([r["d_lr"] for r in interior]),
            "p99_abs_d_biasL": _p99([r["d_bias"] for r in interior])},
        "n_fit_worse": sum(r["d_fit"] < -TOL for r in per),
        "implied_theta_at_bound": {"median": statistics.median(r["implied_theta"] for r in bound),
                                   "min": min(r["implied_theta"] for r in bound),
                                   "max": max(r["implied_theta"] for r in bound)},
        "n_theta_upper_bound": sum(abs(r["theta"] - 1) < BOUND_TOL for r in per),
        "n_credit_at_bound": sum(r["credit_at_bound"] for r in per),
        "n_negative_beta": sum(r["implied_beta"] < 0 for r in per),
        "r1_vs_same_launch": {"max_abs_d_eval": max(abs(r["d_eval_r1"]) for r in per),
                              "n_above_tol": sum(abs(r["d_eval_r1"]) > TOL for r in per),
                              "bound_flips": flips},
        "_per": per}


def figure(r: dict) -> None:
    per = r["_per"]
    fig, axes = plt.subplots(1, 3, figsize=fs.mm(fs.fit_width(square_panels=3, axes=3), 48),
                             layout="constrained")
    colors = {True: fs.OKABE_ITO["vermillion"], False: "0.6"}
    groups = [(True, f"θ at −1 bound (n = {r['split']['at_bound']['n']})"),
              (False, f"θ inside (n = {r['split']['not_at_bound']['n']})")]
    a, b, c = axes
    for keep, label in groups:
        g = [x for x in per if x["at_bound"] == keep]
        a.scatter([x["theta"] for x in g], [1e3 * x["d_fit"] for x in g], s=3, lw=0,
                  color=colors[keep], label=label)
        b.scatter([1e3 * x["d_fit"] for x in g], [1e3 * x["d_eval"] for x in g], s=3, lw=0,
                  color=colors[keep])
        c.scatter([x["theta"] for x in g], [x["implied_theta"] for x in g], s=3, lw=0,
                  color=colors[keep])
    a.axhline(0, color="k", lw=0.5)
    a.set_xlabel("CTT threshold θ")
    a.set_ylabel("Fit gain (mnat/trial)")
    b.axhline(0, color="k", lw=0.5)
    b.axvline(0, color="k", lw=0.5)
    b.set_xlabel("Fit gain (mnat/trial)")
    b.set_ylabel("Held-out gain (mnat/trial)")
    hi = max(x["theta"] for x in per) + 0.05
    c.plot([THETA_MIN, hi], [THETA_MIN, hi], color="k", lw=0.5)
    c.axhline(THETA_MIN, color=fs.OKABE_ITO["vermillion"], lw=0.5, ls="--")
    c.set_xlim(THETA_MIN - 0.05, hi)
    c.set_xlabel("CTT threshold θ")
    c.set_ylabel("θ implied by credit fit")
    fig.legend(*a.get_legend_handles_labels(), loc="outside upper center", ncol=2, frameon=False,
               markerscale=2.5)
    for ax, letter in zip(axes, "abc"):
        fs.panel_label(ax, letter)
    fs.save_figure(fig, FIG)


def render(r: dict) -> str:
    t = {row["model"]: row for row in r["table"]}
    sb, sn = r["split"]["at_bound"], r["split"]["not_at_bound"]
    eq, rv, th = r["equivalence_not_at_bound"], r["r1_vs_same_launch"], r["implied_theta_at_bound"]

    def mn(x: float) -> str:
        return fs.num(1e3 * x, "+.3f")

    def iqr(s: dict) -> str:
        return f"{mn(s['median'])} [{mn(s['q1'])}, {mn(s['q3'])}]"

    lines = [
        "![Credit form vs CompareToThreshold, per subject](../fig_r2_ctt_credit.svg)",
        "",
        f"*Figure.* One dot per subject (n = {r['n_subjects']}); gains are credit form − "
        "CompareToThreshold from the same launch, in millinats per trial. **a**, Fit gain against "
        "CTT's fitted θ. **b**, Held-out gain against fit gain. **c**, The θ the credit fit implies, "
        "−c_loss / (c_win − c_loss), against CTT's θ; the dashed line is CTT's bound θ = −1. One seed per fit, so there are no "
        "error bars.",
        "",
        "| model | pooled held-out, 149 test subjects | pooled odd sessions, 614 training-pool subjects | pooled fit sessions, all 763 |",
        "|---|---|---|---|"]
    for model in (CREDIT, CTT, CTT_R1):
        row = t[model]
        lines.append(f"| {model} | {row['pooled_heldout']:.5f} | {row['pooled_train_pool_eval']:.5f} "
                     f"| {row['pooled_fit_all']:.5f} |")
    lines += [
        "",
        "| subjects | n (of them test) | fit gain, median [IQR] | fit gain > 1e-4 | held-out gain, median [IQR] | held-out up / down |",
        "|---|---|---|---|---|---|"]
    for name, s in (("θ at −1 bound", sb), ("θ inside", sn)):
        lines.append(f"| {name} | {s['n']} ({s['n_heldout']}) | {iqr(s['d_fit'])} | {s['n_fit_gain']} "
                     f"| {iqr(s['d_eval'])} | {s['d_eval']['n_up']} / {s['d_eval']['n_down']} |")
    lines += [
        "",
        f"**Checks.** At the bound means θ within {r['bound_tol']:g} of −1. Inside the bound "
        f"({sn['n']} subjects) the two forms reach the same likelihood: the 99th percentile of "
        f"|fit difference| is {eq['p99_abs_d_fit']:.1e} per trial. {len(eq['n_d_fit_above_tol'])} "
        f"subjects differ by more than {r['tol']:g}, both near the bound and both in the credit "
        "form's favour: " + ", ".join(f"{x['subject_id']} (θ = {fs.num(x['theta'], '.3f')}, fit "
                                      f"{mn(x['d_fit'])}, held-out {mn(x['d_eval'])} mnat/trial)"
                                      for x in eq["n_d_fit_above_tol"]) + ". "
        f"The parameters agree less tightly, along flat directions of the likelihood: 99th "
        f"percentiles of |c_win − β(1 − θ)| {eq['p99_abs_d_credit_win']:.3f}, |c_loss + βθ| "
        f"{eq['p99_abs_d_credit_loss']:.3f}, learning rate {eq['p99_abs_d_learn_rate']:.1e}, bias "
        f"{eq['p99_abs_d_biasL']:.1e}. The credit form fits worse than CTT by more than "
        f"{r['tol']:g} per trial on {r['n_fit_worse']} subjects. No credit reaches its ±20 bound "
        f"({r['n_credit_at_bound']} subjects), no CTT fit reaches θ = +1 "
        f"({r['n_theta_upper_bound']}), and no credit fit implies β < 0 ({r['n_negative_beta']}). "
        f"At the bound, the credit fits imply θ of median {fs.num(th['median'], '.2f')} "
        f"(range {fs.num(th['min'], '.2f')} to {fs.num(th['max'], '.2f')}).",
        "",
        f"**Same-launch CTT (foraging-models v0.29.0) against r1's CTT (v0.26.0).** The largest "
        f"per-subject held-out difference is {rv['max_abs_d_eval']:.1e} per trial "
        f"({rv['n_above_tol']} subjects above {r['tol']:g}); the pooled scores agree to five "
        f"decimals (table). {len(rv['bound_flips'])} subjects sit on the bound in one fit and just "
        "inside it in the other, the search stopping at slightly different points: "
        + ", ".join(f"{f['subject_id']} (θ = {fs.num(f['theta'], '.3f')})" for f in rv["bound_flips"])
        + "."]
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
    REPORT.write_text(_rewrite(REPORT.read_text(), "result-1", render(result)))
    del result["_per"]
    meta = build_meta("analysis/r2_ctt_credit.py", WANDB_GROUPS, study_root=HERE.parent,
                      code=[__file__], inputs=[FITS, RUNS, R1_FITS])
    write_artifact(HERE / "r2_ctt_credit.json", {"_meta": meta, **result},
                   outputs=[FIG, f"{REPORT}#result-1"])


if __name__ == "__main__":
    main()
