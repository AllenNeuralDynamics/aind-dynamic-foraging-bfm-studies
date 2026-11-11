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
R1_RUNS = HERE / "per_subject_fits.json"
# Pinned in the freezes: the ctt-credit group, and the r1 group that fit the reference CTT row.
WANDB_GROUPS = json.loads(RUNS.read_text())["_meta"]["wandb_groups"] + sorted(
    {run["wandb_group"] for run in json.loads(R1_RUNS.read_text())["runs"]
     if run["literature_model"] == "CompareToThreshold"})
REPORT = HERE / "reports" / "r2-ctt-credit.md"
FIG = HERE / "fig_r2_ctt_credit.svg"
CREDIT, CTT, CTT_R1 = "CompareToThreshold-credit", "CompareToThreshold", "CompareToThreshold (r1)"
THETA_MIN, CREDIT_BOUND = -1.0, 20.0   # CompareToThreshold's lower θ bound; the credits' |bound|
BOUND_TOL = 1e-6   # θ within this of −1: the optimizer stopped on the bound
NEAR_BOUND = -0.95  # θ below this is "near the bound" in the text
TOL = 1e-4         # nat per trial (0.1 mnat): agreement that counts as the same fit


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
            "c_win": p["stay_credit_win"], "c_loss": p["stay_credit_loss"],
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
    # The report calls the interior subjects that differ "near the bound, in the credit form's favour".
    assert all(r["theta"] < NEAR_BOUND and r["d_fit"] > 0 for r in interior if abs(r["d_fit"]) > TOL)
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
        # Right-side limits: σ(c), without biasL (which both forms add after a Left choice).
        "stay_after_losses_median": {
            name: statistics.median(1 / (1 + math.exp(-r["c_loss"])) for r in group)
            for name, group in (("at_bound", bound), ("not_at_bound", interior))},
        "stay_after_wins_median": {
            name: statistics.median(1 / (1 + math.exp(-r["c_win"])) for r in group)
            for name, group in (("at_bound", bound), ("not_at_bound", interior))},
        "n_theta_upper_bound": sum(abs(r["theta"] - 1) < BOUND_TOL for r in per),
        "n_credit_at_bound": sum(r["credit_at_bound"] for r in per),
        "n_negative_beta": sum(r["implied_beta"] < 0 for r in per),
        "r1_vs_same_launch": {"max_abs_d_eval": max(abs(r["d_eval_r1"]) for r in per),
                              "n_above_tol": sum(abs(r["d_eval_r1"]) > TOL for r in per),
                              "bound_flips": flips},
        "_per": per}


def figure(r: dict) -> None:
    per = r["_per"]
    fig, (a, b) = plt.subplots(1, 2, figsize=fs.mm(140, 62), layout="constrained",
                               gridspec_kw={"width_ratios": [1, 1.3]})
    bound = [x for x in per if x["at_bound"]]
    inside = [x for x in per if not x["at_bound"]]
    red = fs.OKABE_ITO["vermillion"]

    # a: every subject's credit fit in the (c_win, c_loss) plane. Each θ is a line through the
    # origin, c_loss = −θ / (1 − θ) · c_win; CTT reaches only the wedge below θ = −1.
    x_hi = 1.08 * max(x["c_win"] for x in per)
    y_lo, y_hi = -1.0, 1.25 * max(x["c_loss"] for x in per)
    xs = [0, x_hi]
    a.fill_between(xs, [y_lo, y_lo], [0, x_hi / 2], color="0.93", lw=0, zorder=0)
    a.text(0.97 * x_hi, y_lo / 2, "CTT's range (−1 ≤ θ ≤ 1)", ha="right", va="center",
           color="0.45", in_layout=False)
    for theta in (-1, -0.5, 0):   # labelled in the right margin, where each line leaves the axes
        slope = -theta / (1 - theta)
        color = red if theta == THETA_MIN else "0.45"
        a.plot(xs, [0, slope * x_hi], lw=0.5, color=red if theta == THETA_MIN else "0.65",
               ls="-" if theta == THETA_MIN else ":", zorder=1)
        a.text(1.01 * x_hi, slope * x_hi, f"θ = {fs.num(theta, 'g')}", ha="left", va="center",
               color=color, clip_on=False)
    a.scatter([x["c_win"] for x in inside], [x["c_loss"] for x in inside], s=3, lw=0,
              color="0.55", zorder=2, label=f"CTT's θ inside (n = {len(inside)})")
    a.scatter([x["c_win"] for x in bound], [x["c_loss"] for x in bound], s=4, lw=0, color=red,
              zorder=3, label=f"CTT stopped at θ = −1 (n = {len(bound)})")
    a.set_xlim(0, x_hi)
    a.set_ylim(y_lo, y_hi)
    a.set_xlabel("c_win, stay logit after wins")
    a.set_ylabel("c_loss, stay logit after losses")
    a.legend(loc="upper left", frameon=False, handletextpad=0.2, markerscale=2)
    fs.panel_label(a, "a")

    # b: the bound subjects, ranked by fit gain; held-out gain on top.
    ranked = sorted(bound, key=lambda x: -x["d_fit"])
    idx = range(1, len(ranked) + 1)
    b.bar(idx, [1e3 * x["d_fit"] for x in ranked], width=1.0, color="0.78", lw=0,
          label="fit sessions")
    b.scatter(idx, [1e3 * x["d_eval"] for x in ranked], s=5, lw=0,
              color=fs.OKABE_ITO["blue"], zorder=3, label="held-out sessions")
    b.axhline(0, color="k", lw=0.5)
    b.set_xlim(0, len(ranked) + 1)
    b.set_xlabel(f"The {len(ranked)} subjects at θ = −1, ranked by fit gain")
    b.set_ylabel("Credit − CTT (mnat/trial)")
    b.legend(loc="upper right", bbox_to_anchor=(0.95, 1), frameon=False, handletextpad=0.2)
    fs.panel_label(b, "b")
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
        f"*Figure.* **a**, Each subject's credit fit (n = {r['n_subjects']}). c_win and c_loss "
        "are the stay logits a subject approaches after a run of wins or of losses, before the "
        "side bias (both forms add the same biasL after a Left choice). A threshold θ "
        "is a line through the origin, c_loss = −θ / (1 − θ) · c_win; CTT can reach only the "
        "shaded wedge below θ = −1. Red: subjects whose CTT fit stopped on that line. **b**, Those "
        "subjects ranked by how much better the credit form fits them (bars, fit sessions). Dots: "
        "the same comparison on held-out sessions. Gains are credit form − CTT from the same launch, "
        "in millinats per trial. One seed per fit, so there are no error bars.",
        "",
        f"After a run of losses on the Right side, the subjects at the bound stay with probability "
        f"{r['stay_after_losses_median']['at_bound']:.2f} (median of σ(c_loss)), against "
        f"{r['stay_after_losses_median']['not_at_bound']:.2f} for the others. After wins, "
        f"{r['stay_after_wins_median']['at_bound']:.2f} and "
        f"{r['stay_after_wins_median']['not_at_bound']:.2f} (σ(c_win)). On the Left side the same "
        "limits are σ(c_loss + biasL) and σ(c_win + biasL).",
        "",
        "| model | pooled held-out, 149 test subjects | pooled odd sessions, 614 training-pool subjects | pooled fit sessions, all 763 |",
        "|---|---|---|---|"]
    for model in (CREDIT, CTT, CTT_R1):
        row = t[model]
        lines.append(f"| {model} | {row['pooled_heldout']:.5f} | {row['pooled_train_pool_eval']:.5f} "
                     f"| {row['pooled_fit_all']:.5f} |")
    lines += [
        "",
        f"| subjects | n (of them test) | fit gain (mnat/trial), median [IQR] | fit gain > {1e3 * r['tol']:g} mnat/trial | held-out gain (mnat/trial), median [IQR] | held-out up / down |",
        "|---|---|---|---|---|---|"]
    for name, s in (("θ at −1 bound", sb), ("θ inside", sn)):
        lines.append(f"| {name} | {s['n']} ({s['n_heldout']}) | {iqr(s['d_fit'])} | {s['n_fit_gain']} "
                     f"| {iqr(s['d_eval'])} | {s['d_eval']['n_up']} / {s['d_eval']['n_down']} |")
    lines += [
        "",
        f"**Checks.** At the bound means θ within {r['bound_tol']:g} of −1. Inside the bound "
        f"({sn['n']} subjects) the two forms reach the same likelihood: the 99th percentile of "
        f"|fit difference| is {eq['p99_abs_d_fit']:.1e} nat per trial. {len(eq['n_d_fit_above_tol'])} "
        f"subjects differ by more than {1e3 * r['tol']:g} mnat/trial, all with θ below {fs.num(NEAR_BOUND, 'g')} and all "
        "in the credit form's favour: " + ", ".join(f"{x['subject_id']} (θ = {fs.num(x['theta'], '.3f')}, fit "
                                      f"{mn(x['d_fit'])}, held-out {mn(x['d_eval'])} mnat/trial)"
                                      for x in eq["n_d_fit_above_tol"]) + ". "
        f"The parameters agree less tightly, along flat directions of the likelihood: 99th "
        f"percentiles of |c_win − β(1 − θ)| {eq['p99_abs_d_credit_win']:.3f}, |c_loss + βθ| "
        f"{eq['p99_abs_d_credit_loss']:.3f}, learning rate {eq['p99_abs_d_learn_rate']:.1e}, bias "
        f"{eq['p99_abs_d_biasL']:.1e}. The credit form fits worse than CTT by more than "
        f"{1e3 * r['tol']:g} mnat per trial on {r['n_fit_worse']} subjects. No credit reaches its ±20 bound "
        f"({r['n_credit_at_bound']} subjects), no CTT fit reaches θ = +1 "
        f"({r['n_theta_upper_bound']}), and no credit fit implies β < 0 ({r['n_negative_beta']}). "
        f"At the bound, the credit fits imply θ of median {fs.num(th['median'], '.2f')} "
        f"(range {fs.num(th['min'], '.2f')} to {fs.num(th['max'], '.2f')}).",
        "",
        f"**Same-launch CTT (foraging-models v0.29.0) against r1's CTT (v0.26.0).** The largest "
        f"per-subject held-out difference is {rv['max_abs_d_eval']:.1e} nat per trial "
        f"({rv['n_above_tol']} subjects above {1e3 * r['tol']:g} mnat); the pooled scores agree to five "
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
                      code=[__file__], inputs=[FITS, RUNS, R1_FITS, R1_RUNS])
    write_artifact(HERE / "r2_ctt_credit.json", {"_meta": meta, **result},
                   outputs=[FIG, f"{REPORT}#result-1"])


if __name__ == "__main__":
    main()
