"""C1b: logistic-regression history kernels — and how they change after a bout of ignores.

    python analysis/c1_kernels.py

Study 50's design (Su2022: rewarded choice, unrewarded choice, N_BACK engaged trials back,
``prepare_logistic_design_matrix`` per session on the engaged-trial sequence), on the shared
engaged trials of C and C3. Targets: the animal's choice; C3's P(right | engaged), C's
P(right) and CTT's P(right) as soft labels (each row twice, weights p and 1 − p).

New here: the trials are split by ``gap``, the number of ignored trials just before trial t
(between engaged trials t−1 and t), and a separate kernel is fitted per gap group. If the
animals' dependence on history fades over an ignore bout, their kernel shrinks with the
gap; C3 sees the ignores, C and CTT do not. Ignores further back are not modelled.
Writes ``c1_kernels.json`` and ``fig_C1_kernels.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from aind_dynamic_foraging_models.logistic_regression import MODEL_MAPPER, prepare_logistic_design_matrix  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk02 as s13  # noqa: E402

DESIGN = "Su2022"
N_BACK = 10
C = 1.0
SOURCES = {"animal": ("action", False), "C3": ("p_C3", True), "C": ("p_C", True), "ctt": ("p_ctt", True)}
COLORS = {"animal": "k", "C3": s13.ENGAGE_COLOR, "C": s13.MODEL_COLOR, "ctt": "#55A868"}
GAP_GROUPS = {"0": (0, 0), "≥1": (1, 10_000), "1": (1, 1), "2": (2, 2), "3–5": (3, 5), "6+": (6, 10_000)}
SHOWN = ["0", "1", "2", "3–5", "6+"]


def design() -> tuple[np.ndarray, "pd.DataFrame"]:
    t = s13.engaged_comparison_table()
    blocks, keep = [], []
    for rows in t.groupby("ses_idx", sort=False).indices.values():
        if len(rows) < N_BACK + 2:
            continue
        d = prepare_logistic_design_matrix(t["action"].to_numpy()[rows], (t["reward"].to_numpy()[rows] > 0).astype(int),
                                           logistic_model=DESIGN, n_trial_back=N_BACK)
        blocks.append(d["X"].to_numpy(dtype=np.float32))
        keep.append(rows[d.index.to_numpy()])
    return np.vstack(blocks), t.iloc[np.concatenate(keep)].reset_index(drop=True)


def fit_rows(x: np.ndarray, y: np.ndarray, rows: np.ndarray, soft: bool) -> np.ndarray:
    model = LogisticRegression(C=C, max_iter=2000)
    xs, ys = x[rows], y[rows]
    if soft:
        model.fit(np.vstack([xs, xs]), np.r_[np.ones(len(ys)), np.zeros(len(ys))], sample_weight=np.r_[ys, 1 - ys])
    else:
        model.fit(xs, ys)
    return np.r_[model.coef_[0], model.intercept_[0]]


def as_kernels(coef: np.ndarray) -> dict:
    out = {term: coef[i * N_BACK:(i + 1) * N_BACK][::-1] for i, term in enumerate(MODEL_MAPPER[DESIGN])}
    out["bias"] = coef[-1]
    out["reward_effect"] = out["RewC"] - out["UnrC"]
    out["persistence"] = (out["RewC"] + out["UnrC"]) / 2
    return out


def main() -> None:
    s13.apply_presentation_style()
    x, t = design()
    gap = t["gap"].to_numpy()
    groups = {name: np.flatnonzero((gap >= lo) & (gap <= hi)) for name, (lo, hi) in GAP_GROUPS.items()}
    jobs = [(g, s) for g in GAP_GROUPS for s in SOURCES]
    coefs = Parallel(n_jobs=12)(
        delayed(fit_rows)(x, t[SOURCES[s][0]].to_numpy(float), groups[g], SOURCES[s][1]) for g, s in jobs)
    kernels = {}
    for (g, s), c in zip(jobs, coefs):
        kernels.setdefault(g, {})[s] = as_kernels(c)

    # Shrinkage of the lag-1 weights relative to gap 0, per source.
    shrink = {s: {g: {term: float(kernels[g][s][term][0] / kernels["0"][s][term][0]) for term in ("RewC", "UnrC")}
                  for g in GAP_GROUPS} for s in SOURCES}

    lags = np.arange(1, N_BACK + 1)
    fig, axes = plt.subplots(2, len(SHOWN), figsize=(5 * len(SHOWN), 10), sharex=True, sharey="row")
    for col, g in enumerate(SHOWN):
        for row, term in enumerate(("RewC", "UnrC")):
            ax = axes[row, col]
            for s in SOURCES:
                ax.plot(lags, kernels[g][s][term], color=COLORS[s], marker="x" if s == "animal" else "o",
                        ms=8 if s == "animal" else 4, lw=2.2 if s == "animal" else 1.6, label=s)
            ax.axhline(0, color="0.8", lw=0.8)
            if row == 0:
                ax.set_title(f"gap = {g} ignored just before\n(n={len(groups[g]):,})", fontsize=13)
            if col == 0:
                ax.set_ylabel({"RewC": "rewarded choice k back", "UnrC": "unrewarded choice k back"}[term]
                              + "\n(logistic weight, toward that side)")
            if row == 1:
                ax.set_xlabel("engaged trials back (k)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("History kernels by the number of ignored trials just before the choice "
                 "(Su2022; animal choices, model P(right) as soft labels)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(s13.figure_path("fig_C1_kernels.png"))
    plt.close(fig)

    s13.write_json("c1_kernels.json", {
        "design": DESIGN, "terms": MODEL_MAPPER[DESIGN], "n_back": N_BACK, "C": C, "n_trials": int(len(t)),
        "trials": "shared engaged trials with N_BACK engaged trials of history",
        "gap_groups": {g: {"range": r, "n": int(len(groups[g]))} for g, r in GAP_GROUPS.items()},
        "kernels": kernels,
        "lag1_ratio_to_gap0": shrink,
    }, "analysis/c1_kernels.py", extra_groups=s13.twin_wandb_groups())


if __name__ == "__main__":
    main()
