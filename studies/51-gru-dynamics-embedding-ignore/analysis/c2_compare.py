"""C2: side-choice likelihood of C3, C and the RL models on the same engaged trials.

    python analysis/c2_compare.py

Shared engaged trials of C and C3 (``_pk02.engaged_comparison_table``). Every model is
scored on the animal's L/R choice: C3 by P(right | engaged), C by P(right), the RL fits by
their P(right). C and the RL models never see ignored trials; C3 does. So the question is
whether seeing them helps the side prediction, and where: trials are split by ``gap``, the
number of ignored trials just before (0 = the previous trial was engaged).

Panels: a. pooled likelihood per model, overall and per task; b. per-mouse C3 vs C;
c. likelihood by gap per model; d. per-trial log-likelihood advantage of C3 over C (and of
C over CTT, for scale) by gap, mean ± 1.96 SEM across mice.
Writes ``fig_C2_likelihood.png`` and ``c2_compare.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk02 as s13  # noqa: E402

MODELS = ["C3", "C", "ctt", "bari", "hattori"]
COLORS = {"C3": s13.ENGAGE_COLOR, "C": s13.MODEL_COLOR, "ctt": "#55A868", "bari": "#DD8452", "hattori": "#937860"}
GAP_BINS = [(0, 0), (1, 1), (2, 2), (3, 5), (6, 10), (11, 10_000)]
MIN_TRIALS_MOUSE = 20


def gap_label(lo: int, hi: int) -> str:
    return str(lo) if lo == hi else (f"{lo}+" if hi >= 10_000 else f"{lo}–{hi}")


def main() -> None:
    s13.apply_presentation_style()
    t = s13.engaged_comparison_table()
    right = t["action"].to_numpy() == 1
    for m in MODELS:
        p = np.clip(t[f"p_{m}"].to_numpy(float), 1e-10, 1 - 1e-10)
        t[f"ll_{m}"] = np.log(np.where(right, p, 1 - p))
    t["gap_bin"] = pd.cut(t["gap"], [lo - 0.5 for lo, _ in GAP_BINS] + [np.inf],
                          labels=[gap_label(*b) for b in GAP_BINS])

    def lik(df: pd.DataFrame) -> dict:
        return {m: float(np.exp(df[f"ll_{m}"].mean())) for m in MODELS}

    pooled = {"all": {"n": int(len(t)), **lik(t)}}
    for task in s13.TASKS:
        d = t[t["task"] == task]
        pooled[task] = {"n": int(len(d)), **lik(d)}
    per_mouse = t.groupby("subject_id")[[f"ll_{m}" for m in MODELS]].mean().pipe(np.exp)
    per_mouse.columns = MODELS
    diff_mouse = per_mouse["C3"] - per_mouse["C"]

    by_gap = {}
    for label, d in t.groupby("gap_bin", observed=True):
        mouse = d.groupby("subject_id").agg(n=("ll_C3", "size"), adv=("ll_C3", "mean"), c=("ll_C", "mean"),
                                            ctt=("ll_ctt", "mean"))
        mouse = mouse[mouse["n"] >= MIN_TRIALS_MOUSE]
        adv = mouse["adv"] - mouse["c"]
        ref = mouse["c"] - mouse["ctt"]
        by_gap[str(label)] = {
            "n_trials": int(len(d)), **lik(d),
            "n_mice": int(len(mouse)),
            "C3_minus_C_per_trial_mean_over_mice": float(adv.mean()),
            "C3_minus_C_sem": float(adv.std(ddof=1) / np.sqrt(len(adv))),
            "C3_minus_C_share_mice_positive": float((adv > 0).mean()),
            "C_minus_CTT_per_trial_mean_over_mice": float(ref.mean()),
            "C_minus_CTT_sem": float(ref.std(ddof=1) / np.sqrt(len(ref))),
            "C3_minus_C_pooled_per_trial": float((d["ll_C3"] - d["ll_C"]).mean()),
        }

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    ax = axes[0, 0]
    groups = ["all", *s13.TASKS]
    width = 0.16
    for i, m in enumerate(MODELS):
        ax.bar(np.arange(len(groups)) + (i - 2) * width, [pooled[g][m] for g in groups], width, color=COLORS[m], label=m)
    ax.set_xticks(range(len(groups)), [g.replace(" ", "\n", 1) for g in groups], fontsize=11)
    lo = min(pooled[g][m] for g in groups for m in MODELS)
    ax.set_ylim(lo - 0.02, max(pooled[g][m] for g in groups for m in MODELS) + 0.01)
    ax.set_ylabel("L/R likelihood (geometric mean)")
    ax.set_title(f"a. Pooled, {len(t):,} shared engaged trials", fontsize=14)
    ax.legend(frameon=False, ncol=5, fontsize=11)

    ax = axes[0, 1]
    ax.scatter(per_mouse["C"], per_mouse["C3"], s=14, color=s13.ENGAGE_COLOR, alpha=0.6, linewidths=0)
    lims = [per_mouse[["C", "C3"]].min().min() - 0.01, per_mouse[["C", "C3"]].max().max() + 0.01]
    ax.plot(lims, lims, color="0.5", lw=1, ls=":")
    ax.set(xlim=lims, ylim=lims, xlabel="C (2-way) per-mouse likelihood", ylabel="C3 (3-way) per-mouse likelihood",
           title=f"b. Per mouse: C3 > C in {(diff_mouse > 0).mean():.0%} of {len(diff_mouse)} mice "
                 f"(median Δ {diff_mouse.median():+.4f})")

    ax = axes[1, 0]
    labels = list(by_gap)
    for m in MODELS:
        ax.plot(labels, [by_gap[g][m] for g in labels], color=COLORS[m], marker="o", lw=2, label=m)
    ax.set(xlabel="ignored trials just before this engaged trial (gap)", ylabel="L/R likelihood",
           title="c. Likelihood by gap")
    for i, g in enumerate(labels):
        ax.text(i, ax.get_ylim()[0], f"n={by_gap[g]['n_trials']:,}", ha="center", va="bottom", fontsize=9, color="0.4")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    x = np.arange(len(labels))
    for key, color, name in (("C3_minus_C", s13.ENGAGE_COLOR, "C3 − C"), ("C_minus_CTT", COLORS["ctt"], "C − CTT (scale)")):
        mean = [by_gap[g][f"{key}_per_trial_mean_over_mice"] for g in labels]
        sem = [by_gap[g][f"{key}_sem"] for g in labels]
        ax.errorbar(x, mean, yerr=1.96 * np.asarray(sem), color=color, marker="o", lw=2, capsize=4, label=name)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xticks(x, labels)
    ax.set(xlabel="ignored trials just before (gap)", ylabel="log-likelihood advantage per trial (nats)",
           title=f"d. Advantage by gap (mean over mice with ≥{MIN_TRIALS_MOUSE} trials, ±1.96 SEM)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(s13.figure_path("fig_C2_likelihood.png"))
    plt.close(fig)

    s13.write_json("c2_compare.json", {
        "trials": "shared engaged trials of C and C3; L/R choice scored by every model",
        "pooled_likelihood": pooled,
        "per_mouse": {"n_mice": int(len(per_mouse)), "C3_minus_C_median": float(diff_mouse.median()),
                      "C3_minus_C_share_positive": float((diff_mouse > 0).mean()),
                      "C3_minus_C_quantiles": diff_mouse.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict(),
                      "C3_minus_best_rl_share_positive": float(
                          (per_mouse["C3"] > per_mouse[["ctt", "bari", "hattori"]].max(axis=1)).mean())},
        "by_gap": by_gap,
        "min_trials_per_mouse_in_gap_bin": MIN_TRIALS_MOUSE,
    }, "analysis/c2_compare.py", extra_groups=s13.twin_wandb_groups())


if __name__ == "__main__":
    main()
