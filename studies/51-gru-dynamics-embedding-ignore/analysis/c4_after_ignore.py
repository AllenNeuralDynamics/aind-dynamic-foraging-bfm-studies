"""C4: what the animals do after a bout of ignores, and what each model predicts.

    python analysis/c4_after_ignore.py

Shared engaged trials of C and C3. For each engaged trial with a previous engaged trial in
the session: did the animal repeat that previous engaged choice ("stay")? Split by how many
ignored trials came in between (``gap``) and by whether that previous choice was rewarded.
The model's P(stay) is its P(previous side): C3 via P(right | engaged), C and CTT via
P(right). C and CTT never see the ignores, so a gap only reaches them through the
history before it.

Also the model's confidence |P(right) − 0.5| by gap, and the share of trials where the
animal's choice falls on the side the model favours.
Writes ``fig_C4_after_ignore.png`` and ``c4_after_ignore.json``.
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

MODELS = [("C3", s13.ENGAGE_COLOR), ("C", s13.MODEL_COLOR), ("ctt", "#55A868")]
GAP_BINS = [(0, 0), (1, 1), (2, 2), (3, 5), (6, 10), (11, 10_000)]


def gap_label(lo: int, hi: int) -> str:
    return str(lo) if lo == hi else (f"{lo}+" if hi >= 10_000 else f"{lo}–{hi}")


def main() -> None:
    s13.apply_presentation_style()
    t = s13.engaged_comparison_table()
    g = t.groupby("ses_idx", sort=False)
    t["prev_action"] = g["action"].shift(1)
    t["prev_reward"] = g["reward"].shift(1)
    t = t.dropna(subset=["prev_action"]).copy()
    prev_right = t["prev_action"].to_numpy() == 1
    t["stay"] = (t["action"].to_numpy() == t["prev_action"].to_numpy()).astype(float)
    for m, _ in MODELS:
        p = t[f"p_{m}"].to_numpy(float)
        t[f"pstay_{m}"] = np.where(prev_right, p, 1 - p)
        t[f"conf_{m}"] = np.abs(p - 0.5)
    t["prev_rewarded"] = t["prev_reward"].to_numpy() > 0
    t["gap_bin"] = pd.cut(t["gap"], [lo - 0.5 for lo, _ in GAP_BINS] + [np.inf],
                          labels=[gap_label(*b) for b in GAP_BINS])

    summary = {}
    for outcome, d_out in (("rewarded", t[t["prev_rewarded"]]), ("unrewarded", t[~t["prev_rewarded"]]), ("all", t)):
        rows = {}
        for label, d in d_out.groupby("gap_bin", observed=True):
            mouse = d.groupby("subject_id")["stay"].agg(["mean", "size"])
            mouse = mouse[mouse["size"] >= 20]
            rows[str(label)] = {
                "n": int(len(d)), "animal": float(d["stay"].mean()),
                "animal_sem_over_mice": float(mouse["mean"].std(ddof=1) / np.sqrt(len(mouse))) if len(mouse) > 1 else None,
                **{m: float(d[f"pstay_{m}"].mean()) for m, _ in MODELS},
                **{f"confidence_{m}": float(d[f"conf_{m}"].mean()) for m, _ in MODELS},
            }
        summary[outcome] = rows

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.6))
    for ax, outcome in zip(axes[:2], ("rewarded", "unrewarded")):
        rows = summary[outcome]
        labels = list(rows)
        animal = np.array([rows[k]["animal"] for k in labels])
        sem = np.array([rows[k]["animal_sem_over_mice"] or 0 for k in labels])
        ax.errorbar(labels, animal, yerr=1.96 * sem, color="k", marker="x", ms=9, lw=1.8, capsize=4, label="animal")
        for m, color in MODELS:
            ax.plot(labels, [rows[k][m] for k in labels], color=color, marker="o", lw=2, label=m)
        ax.axhline(0.5, color="0.85", lw=0.8)
        ax.set(xlabel="ignored trials in between (gap)", ylabel="P(repeat previous engaged choice)",
               title=f"previous engaged choice {outcome}", ylim=(0.3, 1.0))
        for i, k in enumerate(labels):
            ax.text(i, 0.31, f"n={rows[k]['n']:,}", ha="center", fontsize=8.5, color="0.4")
        ax.legend(frameon=False)
    ax = axes[2]
    rows = summary["all"]
    labels = list(rows)
    for m, color in MODELS:
        ax.plot(labels, [rows[k][f"confidence_{m}"] for k in labels], color=color, marker="o", lw=2, label=m)
    ax.set(xlabel="ignored trials in between (gap)", ylabel="model confidence |P(right) − 0.5|",
           title="how sure each model is after a gap")
    ax.legend(frameon=False)
    fig.suptitle("After a bout of ignores (shared engaged trials; error bars 1.96 SEM across mice)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(s13.figure_path("fig_C4_after_ignore.png"))
    plt.close(fig)
    s13.write_json("c4_after_ignore.json", {
        "trials": "shared engaged trials with a previous engaged trial in the session",
        "stay": "animal repeats the previous engaged choice; model P(stay) = its P(that side)",
        "by_previous_outcome_and_gap": summary,
    }, "analysis/c4_after_ignore.py", extra_groups=s13.twin_wandb_groups())


if __name__ == "__main__":
    main()
