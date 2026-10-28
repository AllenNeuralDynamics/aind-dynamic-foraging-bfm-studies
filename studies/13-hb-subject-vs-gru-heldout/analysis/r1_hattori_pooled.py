"""Producer for r1: pool the per-subject HB-Hattori2019 runs and compare them with the per-subject
MLE and the GRU.

Offline. Reads the frozen run table (``hattori_runs.json``, ``make freeze-hattori``) and the
comparators from study 01's committed JSON through ``studies/util/cross_study.py``. Each run fitted
one held-out subject alone, so the pooled trial-weighted likelihood is
``exp(sum of log_lik_sum / sum of trials)``, and Monte Carlo errors add in quadrature across runs
(separate runs share no posterior draws, and each run's key folds in its subject).
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
from _meta import build_meta, write_artifact  # noqa: E402
import cross_study  # noqa: E402

# The final set (ADR 0004: production fits, k̂-rescued) carries the headline; production is shown beside it.
WANDB_GROUPS = ["hattori@20261026-002131", "hattori@20261027-181430", "hattori@20261027-223626",
                "hattori@20261027-181415"]
RUNS = HERE / "hattori_final_runs.json"
PRODUCTION = HERE / "hattori_runs.json"
REFIT = HERE / "hattori_refit_runs.json"   # the rejected centred refit (ADR 0004)
PLUGIN = HERE / "hattori_plugin.json"   # freeze_plugin.py: one fixed parameter set per subject
REPORT = HERE / "reports" / "r1-hattori-pooled.md"
N_SUBJECTS = 149


def _pool(rows, key, se_key=None):
    """Trial-weighted pooled likelihood over runs, with its standard error if ``se_key`` is given."""
    total = sum(r[key] for r in rows)
    trials = sum(r["heldout/num_eval_trials"] for r in rows)
    pooled = math.exp(total / trials)
    if se_key is None:
        return pooled, None
    total_se = math.sqrt(sum(r[se_key] ** 2 for r in rows))
    return pooled, pooled * total_se / trials


def compute() -> dict:
    """Pool the frozen rows and line them up against the comparators."""
    rows = [r for r in json.loads(RUNS.read_text())["rows"] if r["state"] == "finished"]
    assert len(rows) == len({r["subject_id"] for r in rows}), "a subject was fitted twice"
    production = [r for r in json.loads(PRODUCTION.read_text())["rows"] if r["state"] == "finished"]
    refit = [r for r in json.loads(REFIT.read_text())["rows"] if r["state"] == "finished"]
    before = {r["subject_id"]: r for r in production}
    before = [before[r["subject_id"]] for r in refit]   # the same subjects, production fits
    final_by = {r["subject_id"]: r for r in rows}
    refit_ids = [r["subject_id"] for r in refit]
    # PLAN.md's gates: R-hat < 1.01 and bulk ESS >= 400 (the run's own flag), and no divergences.
    converged = [r for r in rows if r["hb/n_subjects_converged"] == 1 and not r["hb/divergences"]]
    headline, headline_se = _pool(rows, "heldout/eval_log_lik_sum", "heldout/eval_log_lik_sum_se")
    rl = json.loads(cross_study.RL_SOURCE.read_text())["models"]
    plugin = {r["subject_id"]: r for r in json.loads(PLUGIN.read_text())["rows"]}
    assert set(plugin) == {r["subject_id"] for r in rows}, "plug-in must cover the same subjects"
    plugin_trials = sum(r["n_trials"] for r in plugin.values())
    return {
        "n_subjects": len(rows),
        "n_subjects_expected": N_SUBJECTS,
        "n_trials": sum(r["heldout/num_eval_trials"] for r in rows),
        "n_trials_comparators": rl["hattori"]["n_trials"],
        "hb_session_marginal": headline,
        "hb_session_marginal_se": headline_se,
        "hb_session_marginal_plain": _pool(rows, "heldout/eval_log_lik_sum_plain")[0],
        "hb_session_marginal_production": _pool(production, "heldout/eval_log_lik_sum")[0],
        "n_sessions_importance": sum(r["heldout/n_sessions_importance"] for r in rows),
        "n_sessions_pareto_k_fail_plain": sum(r["heldout/n_sessions_pareto_k_fail_plain"]
                                              for r in rows),
        "n_sessions_pareto_k_fail_production": sum(r["heldout/n_sessions_pareto_k_fail"]
                                                   for r in production),
        "refit_noncentred": {
            "n_subjects": len(refit),
            "divergences_production": sum(r["hb/divergences"] for r in before),
            "divergences_refit": sum(final_by[s]["hb/divergences"] for s in refit_ids),
            "pass_rhat_ess_production": sum(r["hb/n_subjects_converged"] == 1 for r in before),
            "pass_rhat_ess_refit": sum(final_by[s]["hb/n_subjects_converged"] == 1
                                       for s in refit_ids),
            "pass_all_refit": sum(final_by[s]["hb/n_subjects_converged"] == 1
                                  and not final_by[s]["hb/divergences"] for s in refit_ids),
        },
        "refit": {
            "n_subjects": len(refit),
            "divergences_production": sum(r["hb/divergences"] for r in before),
            "divergences_refit": sum(r["hb/divergences"] for r in refit),
            "pass_rhat_ess_production": sum(r["hb/n_subjects_converged"] == 1 for r in before),
            "pass_rhat_ess_refit": sum(r["hb/n_subjects_converged"] == 1 for r in refit),
            "pass_all_refit": sum(r["hb/n_subjects_converged"] == 1 and not r["hb/divergences"]
                                  for r in refit),
        },
        "n_subjects_converged_production": sum(
            1 for r in production if r["hb/n_subjects_converged"] == 1 and not r["hb/divergences"]),
        "hb_trial_marginal": _pool(rows, "heldout/eval_log_lik_sum_trial_marginal")[0],
        "hb_plugin": math.exp(sum(r["plugin_log_lik_sum"] for r in plugin.values()) / plugin_trials),
        "hb_zero_spread": math.exp(sum(r["zero_spread_log_lik_sum"] for r in plugin.values())
                                   / plugin_trials),
        "hb_session_marginal_converged_only": _pool(converged, "heldout/eval_log_lik_sum")[0]
        if converged else None,
        "n_subjects_converged": len(converged),
        "n_sessions": sum(r["heldout/n_sessions"] for r in rows),
        "n_sessions_pareto_k_fail": sum(r["heldout/n_sessions_pareto_k_fail"] for r in rows),
        "divergences": sum(r["hb/divergences"] or 0 for r in rows),
        "max_r_hat": max(r["hb/max_r_hat"] for r in rows),
        "min_ess_bulk": min(r["hb/min_ess_bulk"] for r in rows),
        "mle_hattori": cross_study.rl_baseline("hattori"),
        "mle_ctt": cross_study.rl_baseline("ctt"),
        "gru_by_d": {str(d): v for d, v in cross_study.gru_h128_curve().items()},
    }


def render(r: dict) -> str:
    """The result-1 block."""
    gru = ", ".join(f"D≈{d} {v:.4f}" for d, v in r["gru_by_d"].items())
    conv = r["hb_session_marginal_converged_only"]
    lines = [] if r["n_subjects"] == r["n_subjects_expected"] else [
        f"**Incomplete: {r['n_subjects']} of {r['n_subjects_expected']} subjects finished; "
        "not a result.**", ""]
    lines += [
        "| arm | held-out likelihood (trial-weighted) |",
        "|---|---|",
        f"| HB-Hattori2019 two-level, session-marginal (headline: non-centred refit of the gate-failing "
        f"subjects, k̂-failing sessions importance-sampled; ADR 0004) | **{r['hb_session_marginal']:.4f}** ± {r['hb_session_marginal_se']:.1e} (MC); "
        f"{r['hb_session_marginal']:.6f} |",
        f"| … same fits, plain draw-average on every session | "
        f"{r['hb_session_marginal_plain']:.6f} |",
        f"| … production (before the refit and the k̂ rescue) | "
        f"{r['hb_session_marginal_production']:.6f} |",
        f"| HB-Hattori2019 two-level, plug-in: one fixed parameter set per subject (typical-session "
        f"posterior median), no within-session inference | {r['hb_plugin']:.4f} |",
        f"| HB-Hattori2019 two-level, zero-spread: every posterior draw of the typical session, no "
        f"between-session spread, session-marginal | {r['hb_zero_spread']:.4f} |",
        f"| HB-Hattori2019 two-level, trial-marginal | {r['hb_trial_marginal']:.4f} |",
        f"| per-subject MLE Hattori (study 01 r8) | {r['mle_hattori']:.4f} |",
        f"| per-subject MLE CTT (study 01 r8) | {r['mle_ctt']:.4f} |",
        f"| GRU H128 (study 01 r1) | {gru} |",
        "",
        f"- Subjects {r['n_subjects']}; scored trials {r['n_trials']:,} (comparators: "
        f"{r['n_trials_comparators']:,}).",
        f"- Convergence gates (R-hat < 1.01, bulk ESS ≥ 400, no divergences): {r['n_subjects_converged']} of "
        f"{r['n_subjects']} subjects pass; worst R-hat {r['max_r_hat']:.3f}, lowest ESS "
        f"{r['min_ess_bulk']:.0f}; {r['divergences']} divergences. Converged subjects only: "
        + (f"{conv:.4f}." if conv else "n/a."),
        f"- Pareto k̂ gate: {r['n_sessions_pareto_k_fail_plain']} of {r['n_sessions']} scored sessions "
        f"fail on the plain draw-average (production: {r['n_sessions_pareto_k_fail_production']}); "
        f"{r['n_sessions_importance']} were rescored by importance sampling, after which "
        f"{r['n_sessions_pareto_k_fail']} still fail (k̂ ≥ 0.7 or unjudged).",
        f"- Refit (ADR 0004) of the {r['refit']['n_subjects']} subjects failing a gate in production. "
        f"Centred (rejected): divergences {r['refit']['divergences_production']:,} → "
        f"{r['refit']['divergences_refit']:,}, R-hat/ESS passes "
        f"{r['refit']['pass_rhat_ess_production']} → {r['refit']['pass_rhat_ess_refit']}. "
        f"Non-centred at target acceptance 0.95 (used): divergences "
        f"{r['refit_noncentred']['divergences_production']:,} → "
        f"{r['refit_noncentred']['divergences_refit']:,}, R-hat/ESS passes "
        f"{r['refit_noncentred']['pass_rhat_ess_production']} → "
        f"{r['refit_noncentred']['pass_rhat_ess_refit']}, "
        f"{r['refit_noncentred']['pass_all_refit']} of {r['refit_noncentred']['n_subjects']} "
        "passing every gate.",
    ]
    return "\n".join(lines)


def main() -> None:
    """Compute, rewrite the result block, and write the JSON."""
    result = compute()
    begin, end = "<!-- BEGIN result-1 -->", "<!-- END result-1 -->"
    head, rest = REPORT.read_text().split(begin)
    _, tail = rest.split(end)
    REPORT.write_text(f"{head}{begin}\n{render(result)}\n{end}{tail}")
    meta = build_meta("analysis/r1_hattori_pooled.py", WANDB_GROUPS, study_root=HERE.parent,
                      code=[__file__], inputs=[RUNS, PRODUCTION, REFIT, PLUGIN, cross_study.RL_SOURCE,
                                               cross_study.GRU_CURVE_SOURCE])
    write_artifact(HERE / "r1_hattori_pooled.json", {"_meta": meta, **result},
                   outputs=[HERE / "reports" / "r1-hattori-pooled.md#result-1"])


if __name__ == "__main__":
    main()
