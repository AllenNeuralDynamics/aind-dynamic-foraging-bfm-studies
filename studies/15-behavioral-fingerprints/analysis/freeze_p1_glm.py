#!/usr/bin/env python
"""P1: a per-subject GLM (Su2022 design), fit on the even sessions and replayed on the odd ones.

The design is foraging-models' ``Su2022``: the rewarded choice (``RewC``, +1 right, -1 left, 0 if
unrewarded) and the unrewarded choice (``UnrC``) at 1 to 15 trials back, and a bias. One logistic
regression per subject, on all trials of its fit sessions; its L2 strength is chosen from 10
values by log loss over 5 folds that keep each session whole (a fold of random trials would
share a session's history). The history before a session's first trial is zero, in the fit and
in the replay alike, so every trial has a prediction and the replay is the fitted model itself
(foraging-models' ``prepare_logistic_design_matrix`` instead drops each session's first 15 trials
from its fit; a model that replays every trial needs them).

Gate: the replay's p(right), rebuilt from the stored coefficients, equals scikit-learn's own
prediction to 1e-12. Also writes each subject's held-out likelihood (clipped as study 12 clips),
so the GLM can sit beside study 12's scores, and its tallies in ``p1_replay/`` in the format of
``freeze_p1_replay.py`` (parquet).

Needs /allen (the MEvolve freeze) and scikit-learn; on a compute node:

    sbatch -p aind --ntasks=1 --cpus-per-task=56 --mem=64G --time=01:00:00 --wrap \\
      "env PYTHONPATH=<foraging-models>/src:<mevolve>/src <mevolve env>/bin/python \\
       analysis/freeze_p1_glm.py --models-repo <foraging-models> --mevolve-repo <mevolve>"
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
sys.path.insert(0, str(HERE))
from _meta import build_meta, write_artifact  # noqa: E402
from freeze_p0_golden_mevolve import FREEZE, FREEZE_MANIFEST, _git_sha, _sha256  # noqa: E402
from freeze_p1_replay import CLIP, _pred, pred_frame  # noqa: E402

MODEL = "GLM-Su2022"
LAGS = 15
CS, FOLDS, SEED = 10, 5, 0
TOL = 1e-12
GROUPS = HERE / "p1_session_groups.json"
OUT = HERE / "p1_glm.json"
OUT_TALLIES = HERE / "p1_replay" / f"{MODEL}.parquet"


def design(choices: np.ndarray, rewards: np.ndarray) -> np.ndarray:
    """One session's design: RewC at lags 1..LAGS, then UnrC at lags 1..LAGS; zero before trial 0."""
    side = 2.0 * np.asarray(choices, float) - 1.0
    rewarded = np.asarray(rewards, float) > 0
    rewc, unrc = side * rewarded, side * ~rewarded
    x = np.zeros((len(side), 2 * LAGS))
    for lag in range(1, LAGS + 1):
        x[lag:, lag - 1] = rewc[:-lag]
        x[lag:, LAGS + lag - 1] = unrc[:-lag]
    return x


def _subject_job(args) -> dict:
    sid, fit, score, labels = args
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import GroupKFold

    from aind_dynamic_foraging_models.fingerprints import tally_groups

    x = np.vstack([design(s.choices, s.rewards) for s in fit])
    y = np.concatenate([s.choices for s in fit])
    session = np.concatenate([np.full(len(s.choices), i) for i, s in enumerate(fit)])
    folds = list(GroupKFold(n_splits=min(FOLDS, len(fit))).split(x, y, session))
    reg = LogisticRegressionCV(Cs=CS, cv=folds, penalty="l2", solver="lbfgs", scoring="neg_log_loss",
                               max_iter=5000, random_state=SEED).fit(x, y)
    w, b = reg.coef_[0], float(reg.intercept_[0])

    p_right, log_lik, n, worst = [], 0.0, 0, 0.0
    for s in score:
        xs = design(s.choices, s.rewards)
        p = 1.0 / (1.0 + np.exp(-(xs @ w + b)))
        worst = max(worst, float(np.max(np.abs(p - reg.predict_proba(xs)[:, 1]))))
        p_chosen = np.where(s.choices == 1, p, 1.0 - p)
        log_lik += float(np.log(np.clip(p_chosen, CLIP, 1 - CLIP)).sum())
        n += len(s.choices)
        p_right.append(p)
    tallies = tally_groups([s.choices for s in score], [s.rewards for s in score], labels, p_right)
    return {"subject_id": sid, "C": float(reg.C_[0]), "coef_rewc": w[:LAGS].tolist(),
            "coef_unrc": w[LAGS:].tolist(), "bias": b, "n_fit_trials": int(len(y)),
            "log_lik": log_lik, "n_trials": n, "max_abs_diff_sklearn": worst,
            "pred": _pred(tallies)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-repo", type=Path, required=True)
    ap.add_argument("--mevolve-repo", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=56)
    a = ap.parse_args()
    import aind_dynamic_foraging_models
    import sklearn

    lib_path = Path(aind_dynamic_foraging_models.__file__).resolve()
    assert (a.models_repo / "src").resolve() in lib_path.parents, f"library from {lib_path}"
    models_sha = _git_sha(a.models_repo)
    from mevolve.data import Freeze
    from mevolve.workers import WORKER_ENV

    from aind_dynamic_foraging_models.fingerprints import BINS, SPEC_VERSION
    from aind_dynamic_foraging_models.fingerprints.spec import RATE_STATISTICS

    freeze = Freeze(FREEZE)
    assert _sha256(FREEZE / "manifest.json") == FREEZE_MANIFEST, "another data freeze"
    groups = json.loads(GROUPS.read_text())
    assert groups["freeze_manifest_sha256"] == FREEZE_MANIFEST
    subjects = freeze.subjects("heldout")
    jobs = []
    for sid in subjects:
        score = freeze.subject_sessions(sid, "score")
        jobs.append((sid, freeze.subject_sessions(sid, "fit"), score,
                     [groups["groups"][s.session_id] for s in score]))
    os.environ.update(WORKER_ENV)
    OUT_TALLIES.parent.mkdir(exist_ok=True)
    with mp.get_context("spawn").Pool(min(a.workers, len(jobs))) as pool:
        rows = pool.map(_subject_job, jobs, chunksize=1)

    worst = max(r["max_abs_diff_sklearn"] for r in rows)
    n = sum(r["n_trials"] for r in rows)
    meta_args = dict(study_root=HERE.parent, kind="freeze",
                     code=[__file__, HERE / "freeze_p0_golden_mevolve.py", HERE / "freeze_p1_replay.py"],
                     inputs=[GROUPS],
                     sources=[{"type": "file", "id": str(FREEZE / "manifest.json"), "sha256": FREEZE_MANIFEST}])
    pred_frame([{"subject_id": r["subject_id"], "per_subject": {"pred": r["pred"]}} for r in rows],
               ["per_subject"]).to_parquet(OUT_TALLIES, index=False)
    write_artifact(OUT, {
        "_meta": build_meta("analysis/freeze_p1_glm.py", [], **meta_args),
        "fingerprints_library": {"repo": "AllenNeuralDynamics/aind-dynamic-foraging-models",
                                 "git_sha": models_sha},
        "scikit_learn": sklearn.__version__,
        "model": MODEL,
        "spec": SPEC_VERSION,
        "design": {"regressors": ["RewC", "UnrC"], "lags": LAGS, "history_before_session": 0,
                   "fit": "fit (even) sessions, all trials", "Cs": CS, "folds": FOLDS,
                   "fold_unit": "session", "scoring": "neg_log_loss", "penalty": "l2"},
        "tolerance": TOL,
        "max_abs_diff_sklearn": worst,
        "passed": worst <= TOL,
        "pooled_heldout_likelihood": float(np.exp(sum(r["log_lik"] for r in rows) / n)),
        "subjects": [{k: v for k, v in r.items() if k not in ("pred", "max_abs_diff_sklearn")} for r in rows],
    }, outputs=[OUT_TALLIES])
    print(f"{'PASS' if worst <= TOL else 'FAIL'} (max {worst:.1e}); pooled held-out "
          f"{np.exp(sum(r['log_lik'] for r in rows) / n):.4f} -> {OUT}")


if __name__ == "__main__":
    main()
