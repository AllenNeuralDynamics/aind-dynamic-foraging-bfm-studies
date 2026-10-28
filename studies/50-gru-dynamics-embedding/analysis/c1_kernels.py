"""C1b: logistic-regression history kernels of the animals, the GRU and the RL baselines.

    python analysis/c1_kernels.py

One design for everyone: Su2022 (rewarded choice, unrewarded choice; N_BACK trials back),
built per session from the *animal's* history with
``aind_dynamic_foraging_models.logistic_regression.prepare_logistic_design_matrix``.
Targets differ: the animal's choice, or a model's P(right) as a soft label (each row twice,
y=1 weight p and y=0 weight 1−p — the model's kernel without sampling noise).

Why not Hattori2019: in this encoding its Choice term is exactly RewC + UnrC, so the design
is rank-deficient and the L2 penalty, not the data, splits the weights (the first version
of this script used it; review 2026-10-19). Su2022's two terms are the identified
quantities: the effect of a rewarded and of an unrewarded choice k trials back.

Fits: pooled over all mice for every source, at N_BACK and (animal, GRU) at N_BACK_LONG to
check whether the sign of the late reward effect is a truncation effect; per task for animal
and GRU; per mouse for animal and GRU (spread across mice is the uncertainty shown, and
phase F's input).
Reads cached tables only. Writes ``c1_kernels.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

from aind_dynamic_foraging_models.logistic_regression import MODEL_MAPPER, prepare_logistic_design_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pk01 as s12  # noqa: E402

DESIGN = "Su2022"
N_BACK = 10
N_BACK_LONG = 20
C = 1.0
RL = ("bari", "ctt", "hattori")


def design(n_back: int = N_BACK) -> tuple[np.ndarray, pd.DataFrame]:
    """X (rows = trials with n_back of history) and the matching trial-table rows."""
    t = s12.load_trials(["subject_id", "ses_idx", "trial_idx", "task", "action", "reward", "p_model1_right"])
    for name in RL:
        t[f"p_{name}"] = pd.read_parquet(s12.CACHE / f"rl_trials_{name}_{s12.MODEL_KEY}.parquet",
                                         columns=["p_rl_right"])["p_rl_right"].to_numpy()
    blocks, keep = [], []
    starts = t.groupby("ses_idx", sort=False).indices
    for rows in starts.values():
        if len(rows) < n_back + 2:
            continue
        d = prepare_logistic_design_matrix(t["action"].to_numpy()[rows], (t["reward"].to_numpy()[rows] > 0).astype(int),
                                           logistic_model=DESIGN, n_trial_back=n_back)
        blocks.append(d["X"].to_numpy(dtype=np.float32))
        keep.append(rows[d.index.to_numpy()])
    return np.vstack(blocks), t.iloc[np.concatenate(keep)].reset_index(drop=True)


def fit(x: np.ndarray, y: np.ndarray, soft: bool) -> np.ndarray:
    """Coefficients (design columns, then intercept)."""
    model = LogisticRegression(C=C, max_iter=2000)
    if soft:
        model.fit(np.vstack([x, x]), np.r_[np.ones(len(y)), np.zeros(len(y))], sample_weight=np.r_[y, 1 - y])
    else:
        model.fit(x, y)
    return np.r_[model.coef_[0], model.intercept_[0]]


def fit_rows(x: np.ndarray, y: np.ndarray, rows: np.ndarray | None, soft: bool) -> np.ndarray:
    """Top-level so joblib can memory-map x and y once instead of pickling them per fit."""
    return fit(x, y, soft) if rows is None else fit(x[rows], y[rows], soft)


def as_kernels(coef: np.ndarray, n_back: int = N_BACK) -> dict:
    terms = MODEL_MAPPER[DESIGN]
    # design columns run trial_back n_back..1 within each term; report lag 1..n_back.
    out = {term: coef[i * n_back:(i + 1) * n_back][::-1] for i, term in enumerate(terms)}
    out["bias"] = coef[-1]
    out["reward_effect"] = out["RewC"] - out["UnrC"]
    out["persistence"] = (out["RewC"] + out["UnrC"]) / 2  # outcome-independent tendency to repeat
    return out


def main() -> None:
    x, t = design()
    targets = {"animal": (t["action"].to_numpy().astype(float), False),
               "gru": (t["p_model1_right"].to_numpy(), True),
               **{name: (t[f"p_{name}"].to_numpy(), True) for name in RL}}

    subjects = t.groupby("subject_id", sort=False).indices
    task_rows = {task: np.flatnonzero((t["task"] == task).to_numpy()) for task in s12.TASKS}
    jobs = [("pooled", source, None, None, None) for source in targets]
    jobs += [("task", source, task, None, task_rows[task]) for source in ("animal", "gru") for task in s12.TASKS]
    jobs += [("mouse", source, None, subject, rows) for source in ("animal", "gru") for subject, rows in subjects.items()]

    coefs = Parallel(n_jobs=24, verbose=0)(
        delayed(fit_rows)(x, targets[source][0], rows, targets[source][1]) for _, source, _, _, rows in jobs)
    results = [(kind, source, task, subject, coef) for (kind, source, task, subject, _), coef in zip(jobs, coefs)]
    pooled = {s: as_kernels(c) for k, s, _, _, c in results if k == "pooled"}
    by_task = {}
    for k, s, task, _, c in results:
        if k == "task":
            by_task.setdefault(task, {})[s] = as_kernels(c)
    per_mouse = {}
    for k, s, _, subject, c in results:
        if k == "mouse":
            per_mouse.setdefault(s, {})[subject] = c
    mouse_kernels = {s: [as_kernels(c) for c in coefs.values()] for s, coefs in per_mouse.items()}
    mouse_summary = {
        s: {term: {"median": np.median([m[term] for m in ks], axis=0),
                   "sem": np.std([m[term] for m in ks], axis=0, ddof=1) / np.sqrt(len(ks))}
            for term in ("RewC", "UnrC", "reward_effect", "persistence")}
        for s, ks in mouse_kernels.items()}

    x_long, t_long = design(N_BACK_LONG)
    long_targets = {"animal": (t_long["action"].to_numpy().astype(float), False), "gru": (t_long["p_model1_right"].to_numpy(), True)}
    long_coefs = Parallel(n_jobs=2)(delayed(fit_rows)(x_long, y, None, soft) for y, soft in long_targets.values())
    pooled_long = {s: as_kernels(c, N_BACK_LONG) for s, c in zip(long_targets, long_coefs)}

    manifest_groups = [m["group"] for m in __import__("json").loads((s12.STUDY / "analysis" / "source_runs.json").read_text())["rl_baselines"]["models"].values()]
    out = {"_meta": s12.build_meta("analysis/c1_kernels.py", s12.wandb_groups() + manifest_groups, study_root=s12.STUDY),
           "design": DESIGN, "terms": MODEL_MAPPER[DESIGN], "n_back": N_BACK, "C": C, "n_trials": int(len(t)),
           "soft_labels": "model P(right) as soft targets; animal as observed choices",
           "pooled": pooled, "pooled_n_back_long": pooled_long, "n_back_long": N_BACK_LONG,
           "by_task": by_task, "per_mouse_summary": mouse_summary,
           "per_mouse_coef": {s: {m: c for m, c in coefs.items()} for s, coefs in per_mouse.items()},
           "coef_layout": "per term, trial_back N..1, then intercept"}
    path = s12.ANALYSIS / "c1_kernels.json"
    import json  # noqa: PLC0415
    path.write_text(json.dumps(out, indent=1, default=s12._json_default) + "\n")
    print("reward effect lag1..: animal", np.round(pooled["animal"]["reward_effect"], 3), "gru", np.round(pooled["gru"]["reward_effect"], 3))


if __name__ == "__main__":
    main()
