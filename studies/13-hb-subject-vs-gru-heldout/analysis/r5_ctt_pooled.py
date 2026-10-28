"""Producer for r5: pool the per-subject HB-CTT runs and compare them with HB-Hattori, the
per-subject MLE and the GRU.

Offline. Reads the frozen CTT run tables (``ctt_final_runs.json`` and production ``ctt_runs.json``,
``make freeze-ctt``), this study's final
HB-Hattori run table (``hattori_final_runs.json``) and the comparators from study 01's committed JSON
through ``studies/util/cross_study.py``. Pooling is as in r1: ``exp(sum of log_lik_sum / sum of
trials)`` over subjects, Monte Carlo errors in quadrature across runs.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
from _meta import build_meta, write_artifact  # noqa: E402
import cross_study  # noqa: E402
import figure_style as fs  # noqa: E402

WANDB_GROUPS = ["ctt@20261028-003745", "ctt@20261028-020002",          # via the ctt freezes
                "hattori@20261027-181430", "hattori@20261027-223626"]   # via hattori_final_runs.json
RUNS = HERE / "ctt_final_runs.json"          # refit for the gate-failing subjects, production else
PRODUCTION = HERE / "ctt_runs.json"
HATTORI = HERE / "hattori_final_runs.json"
REPORT = HERE / "reports" / "r5-ctt-pooled.md"
FIG = HERE / "fig_r5_ctt.svg"
N_SUBJECTS = 149


def _finished(path):
    """Finished rows of a frozen run table."""
    return [r for r in json.loads(path.read_text())["rows"] if r["state"] == "finished"]


def _pool(rows, key, se_key=None):
    """Trial-weighted pooled likelihood over runs, with its standard error if ``se_key`` is given."""
    trials = sum(r["heldout/num_eval_trials"] for r in rows)
    pooled = math.exp(sum(r[key] for r in rows) / trials)
    if se_key is None:
        return pooled, None
    return pooled, pooled * math.sqrt(sum(r[se_key] ** 2 for r in rows)) / trials


def _per_subject(rows):
    """Each subject's trial-weighted likelihood."""
    return {r["subject_id"]: math.exp(r["heldout/eval_log_lik_sum"] / r["heldout/num_eval_trials"])
            for r in rows}


def compute() -> dict:
    """Pool the CTT rows and line them up against HB-Hattori and the comparators."""
    ctt, hattori, production = _finished(RUNS), _finished(HATTORI), _finished(PRODUCTION)
    assert {r["subject_id"] for r in ctt} == {r["subject_id"] for r in hattori}, "subject sets differ"
    by_subject = {r["subject_id"]: r["heldout/num_eval_trials"] for r in hattori}
    assert all(r["heldout/num_eval_trials"] == by_subject[r["subject_id"]] for r in ctt), \
        "scored trials differ from HB-Hattori"
    converged = [r for r in ctt if r["hb/n_subjects_converged"] == 1 and not r["hb/divergences"]]
    headline, headline_se = _pool(ctt, "heldout/eval_log_lik_sum", "heldout/eval_log_lik_sum_se")
    rl = json.loads(cross_study.RL_SOURCE.read_text())["models"]
    mle_ctt = {str(k): v for k, v in rl["ctt"]["per_subject_likelihood"].items()}
    hb_ctt, hb_hat = _per_subject(ctt), _per_subject(hattori)
    subjects = sorted(hb_ctt)
    vs_hattori = np.array([hb_ctt[s] - hb_hat[s] for s in subjects])
    vs_mle = np.array([hb_ctt[s] - mle_ctt[s] for s in subjects])
    return {
        "n_subjects": len(ctt),
        "n_subjects_expected": N_SUBJECTS,
        "n_trials": sum(r["heldout/num_eval_trials"] for r in ctt),
        "n_trials_comparators": rl["ctt"]["n_trials"],
        "hb_ctt": headline,
        "hb_ctt_se": headline_se,
        "hb_ctt_plain": _pool(ctt, "heldout/eval_log_lik_sum_plain")[0],
        "hb_ctt_production": _pool(production, "heldout/eval_log_lik_sum")[0],
        "n_subjects_converged_production": sum(
            1 for r in production if r["hb/n_subjects_converged"] == 1 and not r["hb/divergences"]),
        "divergences_production": sum(r["hb/divergences"] or 0 for r in production),
        "n_subjects_refit": sum(1 for r in ctt if r["wandb_group"] == "ctt@20261028-020002"),
        "hb_ctt_trial_marginal": _pool(ctt, "heldout/eval_log_lik_sum_trial_marginal")[0],
        "hb_ctt_converged_only": _pool(converged, "heldout/eval_log_lik_sum")[0] if converged else None,
        "n_subjects_converged": len(converged),
        "divergences": sum(r["hb/divergences"] or 0 for r in ctt),
        "max_r_hat": max(r["hb/max_r_hat"] for r in ctt),
        "min_ess_bulk": min(r["hb/min_ess_bulk"] for r in ctt),
        "n_sessions": sum(r["heldout/n_sessions"] for r in ctt),
        "n_sessions_importance": sum(r["heldout/n_sessions_importance"] for r in ctt),
        "n_sessions_pareto_k_fail": sum(r["heldout/n_sessions_pareto_k_fail"] for r in ctt),
        "hb_hattori": _pool(hattori, "heldout/eval_log_lik_sum")[0],
        "mle_ctt": cross_study.rl_baseline("ctt"),
        "mle_hattori": cross_study.rl_baseline("hattori"),
        "gru_by_d": {str(d): v for d, v in cross_study.gru_h128_curve().items()},
        "per_subject": {
            "hb_ctt_minus_hb_hattori_median": float(np.median(vs_hattori)),
            "hb_ctt_beats_hb_hattori": int(np.sum(vs_hattori > 0)),
            "hb_ctt_minus_mle_ctt_median": float(np.median(vs_mle)),
            "hb_ctt_beats_mle_ctt": int(np.sum(vs_mle > 0)),
            "_vs_hattori": vs_hattori.tolist(),
            "_vs_mle": vs_mle.tolist(),
        },
    }


def figure(r: dict) -> None:
    """(a) Pooled held-out likelihood per model; (b) per-subject gain of HB-CTT."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=fs.mm(fs.fit_width(groups=6, axes=2), 60),
                                 gridspec_kw={"width_ratios": [1.6, 1]})
    arms = [("MLE\nHattori", r["mle_hattori"], "0.45"), ("MLE\nCTT", r["mle_ctt"], "0.2"),
            ("HB\nHattori", r["hb_hattori"], fs.OKABE_ITO["sky"]),
            ("HB\nCTT", r["hb_ctt"], fs.OKABE_ITO["vermillion"])]
    for i, (_, v, c) in enumerate(arms):
        ax.plot(i, v, "o", color=c, ms=4)
        ax.annotate(fs.num(v, ".4f"), xy=(i, v), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=5)
    gru = r["gru_by_d"]
    for k, (d, v) in enumerate(gru.items()):
        ax.axhline(v, color=fs.OKABE_ITO["blue"], lw=0.5, ls=(0, (3, 2)),
                   alpha=0.3 + 0.7 * k / max(len(gru) - 1, 1))
        ax.annotate(f"GRU D≈{d}", xy=(len(arms) - 0.5, v), xytext=(2, 0), textcoords="offset points",
                    color=fs.OKABE_ITO["blue"], va="center", fontsize=5)
    ax.set_xticks(range(len(arms)), [a[0] for a in arms])
    ax.set_xlim(-0.5, len(arms) + 0.6)
    ax.set_ylabel("held-out likelihood (trial-weighted)")
    fs.panel_label(ax, "a")
    fs.violins(bx, [r["per_subject"]["_vs_mle"], r["per_subject"]["_vs_hattori"]],
               [fs.OKABE_ITO["vermillion"]] * 2, ["HB-CTT\n− MLE CTT", "HB-CTT\n− HB-Hattori"])
    bx.axhline(0, color="0.5", lw=0.5)
    bx.set_ylabel("per-subject difference")
    fs.panel_label(bx, "b")
    fig.tight_layout()
    fs.save_figure(fig, FIG)
    plt.close(fig)


def render(r: dict) -> str:
    """The result-1 block."""
    gru = ", ".join(f"D≈{d} {v:.4f}" for d, v in r["gru_by_d"].items())
    p = r["per_subject"]
    lines = [] if r["n_subjects"] == r["n_subjects_expected"] else [
        f"**Incomplete: {r['n_subjects']} of {r['n_subjects_expected']} subjects; not a result.**", ""]
    conv = r["hb_ctt_converged_only"]
    lines += [
        "![HB-CTT against the other models](../fig_r5_ctt.svg)",
        "",
        "*(a) Pooled held-out likelihood. Dashed lines: the GRU H128 curve (study 01 r1); darker "
        "means more training subjects. (b) Per-subject difference of HB-CTT from the per-subject "
        "MLE CTT and from HB-Hattori.*",
        "",
        "| model | held-out likelihood (trial-weighted) |",
        "|---|---|",
        f"| HB-CTT two-level, session-marginal (headline) | **{r['hb_ctt']:.4f}** ± "
        f"{r['hb_ctt_se']:.1e} (MC) |",
        f"| … production, before the refit | {r['hb_ctt_production']:.4f} |",
        f"| HB-Hattori2019 two-level (r1) | {r['hb_hattori']:.4f} |",
        f"| per-subject MLE CTT (study 01 r8) | {r['mle_ctt']:.4f} |",
        f"| per-subject MLE Hattori (study 01 r8) | {r['mle_hattori']:.4f} |",
        f"| GRU H128 (study 01 r1) | {gru} |",
        "",
        f"- Per subject: HB-CTT beats the MLE CTT for {p['hb_ctt_beats_mle_ctt']} of "
        f"{r['n_subjects']} subjects (median {fs.num(p['hb_ctt_minus_mle_ctt_median'], '+.4f')}), and "
        f"HB-Hattori for {p['hb_ctt_beats_hb_hattori']} (median "
        f"{fs.num(p['hb_ctt_minus_hb_hattori_median'], '+.4f')}).",
        f"- Subjects {r['n_subjects']}; scored trials {r['n_trials']:,} (comparators: "
        f"{r['n_trials_comparators']:,}).",
        f"- Convergence gates (R-hat < 1.01, bulk ESS ≥ 400, no divergences): "
        f"{r['n_subjects_converged']} of {r['n_subjects']} pass; worst R-hat {r['max_r_hat']:.3f}, "
        f"lowest ESS {r['min_ess_bulk']:.0f}; {r['divergences']} divergences. Converged only: "
        + (f"{conv:.4f}." if conv else "n/a."),
        f"- Refit: the {r['n_subjects_refit']} subjects failing a gate in production were refit as in "
        f"hattori (ADR 0004). Production: {r['n_subjects_converged_production']} of "
        f"{r['n_subjects']} passing every gate, {r['divergences_production']:,} divergences.",
        f"- Pareto k̂: {r['n_sessions_importance']} of {r['n_sessions']} sessions failed on the plain "
        f"draw-average and were importance-sampled; {r['n_sessions_pareto_k_fail']} still fail. "
        f"All-plain {r['hb_ctt_plain']:.6f}; headline {r['hb_ctt']:.6f}.",
        f"- Trial-marginal (study 08's score): {r['hb_ctt_trial_marginal']:.4f}.",
    ]
    return "\n".join(lines)


def main() -> None:
    """Compute, draw, rewrite the result block, and write the JSON."""
    fs.apply_paper_style()
    result = compute()
    figure(result)
    begin, end = "<!-- BEGIN result-1 -->", "<!-- END result-1 -->"
    head, rest = REPORT.read_text().split(begin)
    _, tail = rest.split(end)
    REPORT.write_text(f"{head}{begin}\n{render(result)}\n{end}{tail}")
    result["per_subject"] = {k: v for k, v in result["per_subject"].items() if not k.startswith("_")}
    meta = build_meta("analysis/r5_ctt_pooled.py", WANDB_GROUPS, study_root=HERE.parent,
                      code=[__file__], inputs=[RUNS, PRODUCTION, HATTORI, cross_study.RL_SOURCE,
                                               cross_study.GRU_CURVE_SOURCE])
    write_artifact(HERE / "r5_ctt_pooled.json", {"_meta": meta, **result},
                   outputs=[FIG, HERE / "reports" / "r5-ctt-pooled.md#result-1"])


if __name__ == "__main__":
    main()
