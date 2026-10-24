"""Adaptation-budget curves for Grossman (mouse) and Chen (mouse) (dispatcher #166). Offline.

Reads the frozen ``adaptation_budget_results.json`` (freeze_adaptation_budget.py) and the
pre-registered ``adaptation_budget_cohort.json``. The unit is the subject: each subject's
test log likelihood per trial, in bits, with the GRU averaged over its three source seeds.
Every number is over a fixed cohort, the same subjects at every K:

  * ``primary`` (>= 4 adaptation sessions): K in {0, 1, 2, 4}. Chen (mouse) has exactly 4
    adaptation sessions per subject, so its K=4 is its full half.
  * ``k8`` (Grossman (mouse), >= 8 adaptation sessions): K in {0, 1, 2, 4, 8, half}.

Uncertainty is a subject bootstrap (10,000 resamples, fixed seed) of the cohort mean and of
the paired GRU - Q difference. Q has no K=0. Writes adaptation_budget_summary.json,
fig_adaptation_budget.png and the result block of reports/r7-adaptation-budget.md.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta  # noqa: E402
from plot_style import apply_presentation_style  # noqa: E402

ANALYSIS = STUDY / "analysis"
RESULTS = ANALYSIS / "adaptation_budget_results.json"
COHORT = ANALYSIS / "adaptation_budget_cohort.json"
SUMMARY = ANALYSIS / "adaptation_budget_summary.json"
FIGURE = ANALYSIS / "fig_adaptation_budget.png"
REPORT = ANALYSIS / "reports" / "r7-adaptation-budget.md"
LABELS = {"grossman": "Grossman (mouse)", "chen": "Chen (mouse)"}
# (cohort, dataset) -> the budgets on its curve; a value is the results key to read
CURVES = {
    ("primary", "grossman"): {"0": "0", "1": "1", "2": "2", "4": "4"},
    ("primary", "chen"): {"0": "0", "1": "1", "2": "2", "4": "half"},
    ("k8", "grossman"): {"0": "0", "1": "1", "2": "2", "4": "4", "8": "8", "half": "half"},
}
N_BOOT, SEED = 10_000, 0
LN2 = math.log(2)


def _bits(record: dict, subjects: list[str]) -> np.ndarray:
    per_subject = record["metrics"]["per_subject_mean_log_likelihood_nats"]
    return np.array([per_subject[s] / LN2 for s in subjects])


def _boot_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _summarize(results: dict, cohorts: dict) -> list[dict]:
    rng = np.random.default_rng(SEED)
    rows = []
    for (cohort, dataset), curve in CURVES.items():
        subjects = cohorts[dataset]["cohorts"][cohort]
        for k, key in curve.items():
            gru = np.mean([_bits(r, subjects) for r in results["gru"][dataset][key].values()], axis=0)
            row = {"cohort": cohort, "dataset": dataset, "K": k, "results_key": key,
                   "n_subjects": len(subjects),
                   "gru_bits": float(gru.mean()), "gru_bits_ci95": _boot_ci(gru, rng)}
            if key in results["q"][dataset]:
                q = _bits(results["q"][dataset][key], subjects)
                diff = gru - q
                row.update(q_bits=float(q.mean()), q_bits_ci95=_boot_ci(q, rng),
                           gru_minus_q_bits=float(diff.mean()),
                           gru_minus_q_bits_ci95=_boot_ci(diff, rng),
                           frac_subjects_gru_better=float((diff > 0).mean()))
            rows.append(row)
    return rows


def _figure(rows: list[dict]) -> None:
    apply_presentation_style()
    panels = list(CURVES)
    fig, axes = plt.subplots(2, len(panels), figsize=(6 * len(panels), 9), sharex="col")
    for col, (cohort, dataset) in enumerate(panels):
        sub = [r for r in rows if r["cohort"] == cohort and r["dataset"] == dataset]
        x = np.arange(len(sub))
        for model, color, offset in (("gru", "#08519c", -0.06), ("q", "#636363", 0.06)):
            pts = [(i, r) for i, r in enumerate(sub) if f"{model}_bits" in r]
            xs = [i + offset for i, _ in pts]
            ys = [r[f"{model}_bits"] for _, r in pts]
            err = np.array([[y - r[f"{model}_bits_ci95"][0], r[f"{model}_bits_ci95"][1] - y]
                            for y, (_, r) in zip(ys, pts)]).T
            axes[0, col].errorbar(xs, ys, yerr=err, marker="o", capsize=4, color=color,
                                  label="frozen-core GRU" if model == "gru" else "Bari2019 Q")
        d = [(i, r) for i, r in enumerate(sub) if "gru_minus_q_bits" in r]
        dy = [r["gru_minus_q_bits"] for _, r in d]
        derr = np.array([[y - r["gru_minus_q_bits_ci95"][0], r["gru_minus_q_bits_ci95"][1] - y]
                         for y, (_, r) in zip(dy, d)]).T
        axes[1, col].errorbar([i for i, _ in d], dy, yerr=derr, marker="s", capsize=4, color="k")
        axes[1, col].axhline(0, color="0.6", ls="--", lw=1)
        axes[0, col].set_title(f"{LABELS[dataset]} -- {cohort} cohort (n={sub[0]['n_subjects']})")
        axes[1, col].set_xticks(x, [r["K"] for r in sub])
        axes[1, col].set_xlabel("adaptation sessions per subject (K)")
        axes[0, col].set_ylabel("test log2 likelihood (bits/trial)")
        axes[1, col].set_ylabel("GRU - Q (bits/trial)")
    axes[0, 0].legend(frameon=False)
    fig.text(0.5, 0.005, "Points: subject-mean (GRU averaged over 3 source seeds); bars: 95% "
             "subject-bootstrap CI. Same subjects and test trials at every K.",
             ha="center", va="bottom", fontsize=12, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)


def _table(rows: list[dict]) -> str:
    out = ["| cohort | dataset | K | n | GRU bits/trial [95% CI] | Q bits/trial [95% CI] | "
           "GRU - Q [95% CI] | subjects GRU better |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        def ci(v, c):
            return f"{v:.4f} [{c[0]:.4f}, {c[1]:.4f}]"
        q = ci(r["q_bits"], r["q_bits_ci95"]) if "q_bits" in r else "--"
        d = (f"{r['gru_minus_q_bits']:+.4f} [{r['gru_minus_q_bits_ci95'][0]:+.4f}, "
             f"{r['gru_minus_q_bits_ci95'][1]:+.4f}]") if "q_bits" in r else "--"
        w = f"{100 * r['frac_subjects_gru_better']:.0f}%" if "q_bits" in r else "--"
        out.append(f"| {r['cohort']} | {LABELS[r['dataset']]} | {r['K']} | {r['n_subjects']} | "
                   f"{ci(r['gru_bits'], r['gru_bits_ci95'])} | {q} | {d} | {w} |")
    return "\n".join(out)


def _write_report(rows: list[dict], groups: list[str]) -> None:
    body = ("[regenerated by `analysis/report_adaptation_budget.py` -- do not edit by hand]\n\n"
            "![adaptation budget](../fig_adaptation_budget.png)\n\n" + _table(rows) + "\n\n"
            "Source W&B groups: " + ", ".join(f"`{g}`" for g in groups) + ".")
    text = REPORT.read_text()
    pattern = re.compile(r"(<!-- BEGIN result-1 -->).*?(<!-- END result-1 -->)", re.DOTALL)
    if not pattern.search(text):
        raise AssertionError(f"{REPORT} has no result-1 markers")
    REPORT.write_text(pattern.sub(lambda m: m.group(1) + "\n" + body + "\n" + m.group(2), text))


def main() -> None:
    results = json.loads(RESULTS.read_text())
    cohorts = json.loads(COHORT.read_text())["datasets"]
    rows = _summarize(results, cohorts)
    groups = results["_meta"]["wandb_groups"]
    SUMMARY.write_text(json.dumps({
        "_meta": build_meta("analysis/report_adaptation_budget.py", groups, study_root=STUDY),
        "unit": "subject; test log2 likelihood per trial (bits), GRU averaged over 3 source seeds",
        "bootstrap": {"n": N_BOOT, "seed": SEED, "over": "subjects"},
        "rows": rows,
    }, indent=1) + "\n")
    _figure(rows)
    _write_report(rows, groups)
    print(f"wrote {SUMMARY.name}, {FIGURE.name}, {REPORT.name}")


if __name__ == "__main__":
    main()
