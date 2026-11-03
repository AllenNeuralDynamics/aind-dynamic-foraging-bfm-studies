#!/usr/bin/env python
"""P0 golden check 2: the fingerprint library reproduces the wrapper's committed rollout numbers.

Study 05's ``generative-rl-baseline`` variant rolled the per-subject compare-to-threshold, Bari2019
and Hattori2019 fits out over all 614 training-pool subjects (18,124 sessions, one rollout each)
and committed, per alias, the wrapper's subject-mean history-pattern rows
(``rl_rollout_summaries/<alias>_history_patterns.json``) and its quantitative summaries
(``<alias>_quantitative_summary.json``). The session histories behind them are still cached on
scratch. From those histories this rebuilds, with fingerprints-v1 tallies only:

- the ``abstract`` history-pattern rows at 1, 2 and 3 trials back (per pattern, the mean over
  subjects with at least 5 trials on both sides of each subject's switch rate, and that count);
- their summaries (rows' MAE, RMSE, bias, correlation, total weight);
- the subject-mean summary of the post-switch statistic by reward x run length (1, >1).

Pass: every number within 1e-12 and every subject count equal. The ``detailed`` encoding keeps
left/right identity, which fingerprints-v1 does not, so it is not reproduced.

Needs /allen (the cached histories); on a compute node:

    srun --ntasks=1 --cpus-per-task=4 --mem=32G -p aind --time=00:30:00 \\
      env PYTHONPATH=<foraging-models>/src <dynamic-foraging-bfm-cpu env>/bin/python \\
      analysis/freeze_p0_golden_wrapper.py --models-repo <foraging-models>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
from _meta import _git_env, build_meta, write_artifact  # noqa: E402

from aind_dynamic_foraging_models.fingerprints import (  # noqa: E402
    BINS,
    OBS,
    N,
    add,
    drop_ignores,
    empty_tally,
    session_tally,
    wrapper_abstract_label,
    wrapper_post_switch,
)

CACHE = Path("/allen/aind/scratch/han.hou/tmp/rlgen")
ALIASES = {"ctt": "ForagingCompareThreshold", "bari": "QLearning_L1F1_CK1_softmax",
           "hattori": "QLearning_L2F1_softmax"}
COMMITTED = HERE.parents[1] / "05-disrnn-scaling-law/variants/generative-rl-baseline/rl_rollout_summaries"
MIN_TRIALS = 5          # the wrapper's subject_min_trials and _SUBJECT_LEVEL_MIN_ANIMAL_N
DEPTH_OFFSETS = {1: 0, 2: 2, 3: 10}
TOL = 1e-12
SUMMARY_KEYS = ("n_rows", "total_weight", "mae", "rmse", "bias", "correlation")
OUT = HERE / "p0_golden_wrapper.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha(repo: Path) -> str:
    env = _git_env(repo)  # ignores a GIT_DIR / GIT_WORK_TREE exported for another checkout
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", "src"],
                           capture_output=True, text=True, check=True, env=env).stdout.strip()
    if dirty:
        raise SystemExit(f"{repo} has uncommitted changes under src/: {dirty}")
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True, env=env).stdout.strip()


def subject_tallies(frame) -> tuple[dict, int]:
    """Per-subject summed tallies, and the count of engaged trials with no reward value."""
    out, n_nan_reward = {}, 0
    for row in frame.itertuples(index=False):
        c, r = drop_ignores(row.choice_history, row.reward_history)
        n_nan_reward += int(np.isnan(r).sum())
        sid = str(row.subject_id)
        out[sid] = add(out.get(sid, empty_tally()), session_tally(c, r))
    return out, n_nan_reward


def _rate(table_row) -> float:
    return float(table_row[OBS] / table_row[N]) if table_row[N] > 0 else math.nan


def summarize(rows: list[dict]) -> dict:
    """The wrapper's ``_summarize_probability_rows`` with ``weight_key="n_subjects"``."""
    rows = [r for r in rows if math.isfinite(r["animal_mean"]) and math.isfinite(r["simulated_mean"])]
    x = np.array([r["animal_mean"] for r in rows])
    y = np.array([r["simulated_mean"] for r in rows])
    d = y - x
    return {"n_rows": len(rows), "total_weight": float(sum(r["n_subjects"] for r in rows)),
            "mae": float(np.mean(np.abs(d))), "rmse": float(np.sqrt(np.mean(d ** 2))),
            "bias": float(np.mean(d)), "correlation": float(np.corrcoef(x, y)[0, 1])}


def subject_mean_rows(animal, simulated, stat, indices, names, animal_only_min=False):
    """Per bin, subject means over subjects passing the wrapper's minimum-count rule."""
    subjects = [s for s in animal if s in simulated]
    rows = []
    for i, name in zip(indices, names):
        a_vals, s_vals = [], []
        for s in subjects:
            a, m = animal[s][stat][i], simulated[s][stat][i]
            ok = a[N] >= MIN_TRIALS and (m[N] > 0 if animal_only_min else m[N] >= MIN_TRIALS)
            if ok:
                a_vals.append(_rate(a))
                s_vals.append(_rate(m))
        rows.append({"pattern": name, "n_subjects": len(a_vals),
                     "animal_mean": float(np.mean(a_vals)) if a_vals else math.nan,
                     "simulated_mean": float(np.mean(s_vals)) if s_vals else math.nan})
    return rows


def _post_switch_tallies(per_subject: dict) -> dict:
    return {s: {"post_switch": wrapper_post_switch(t)} for s, t in per_subject.items()}


def check_alias(alias: str) -> tuple[dict, list]:
    model_dir = CACHE / alias / "analysis" / ALIASES[alias]
    paths = {"animal": CACHE / alias / "analysis" / "animal_session_history.pkl",
             "simulated": model_dir / "simulated_session_history.pkl"}
    frames = {k: pickle.loads(p.read_bytes()) for k, p in paths.items()}
    assert len(frames["animal"]) == len(frames["simulated"]) == 18124, alias
    animal, nan_a = subject_tallies(frames["animal"])
    simulated, nan_s = subject_tallies(frames["simulated"])
    assert nan_a == nan_s == 0, f"{alias}: engaged trials with no reward ({nan_a}, {nan_s})"

    patterns = json.loads((COMMITTED / f"{alias}_history_patterns.json").read_text())
    quant = json.loads((COMMITTED / f"{alias}_quantitative_summary.json").read_text())
    result = {"n_subjects": len(animal), "n_sessions": len(frames["animal"]), "depths": {}}
    worst = 0.0
    for depth, offset in DEPTH_OFFSETS.items():
        idx = list(range(offset, offset + 2 * 4 ** (depth - 1)))
        names = [wrapper_abstract_label(BINS["pattern"][i]) for i in idx]
        ours = {r["pattern"]: r for r in subject_mean_rows(animal, simulated, "pattern", idx, names)}
        block = patterns["subject_aggregate"]["abstract"][str(depth)]
        theirs = {r["pattern"]: r for r in block["rows"]}
        assert set(theirs) <= set(ours), f"{alias} depth {depth}: patterns {set(theirs) - set(ours)}"
        diffs = [abs(ours[p][k] - theirs[p][k]) for p in theirs for k in ("animal_mean", "simulated_mean")]
        count_mismatch = sum(ours[p]["n_subjects"] != theirs[p]["n_subjects"] for p in theirs)
        summary = summarize([ours[p] for p in theirs])
        committed = quant["history_dependent"]["quantitative_summary"]["subject_mean"]["abstract"][str(depth)]
        sdiff = {k: abs(summary[k] - committed[k]) for k in SUMMARY_KEYS}
        worst = max(worst, max(diffs), max(sdiff.values()))
        result["depths"][str(depth)] = {"n_rows": len(theirs), "max_abs_row_diff": max(diffs),
                                        "n_subject_count_mismatches": count_mismatch,
                                        "summary_abs_diff": sdiff}

    rows = subject_mean_rows(_post_switch_tallies(animal), _post_switch_tallies(simulated),
                             "post_switch", range(4), ["r1", "r>1", "u1", "u>1"],
                             animal_only_min=True)
    summary = summarize(rows)
    committed = quant["switch_triggered"]["quantitative_summary"]["subject_mean"][
        "post_switch_by_reward_and_run_length"]
    sdiff = {k: abs(summary[k] - committed[k]) for k in SUMMARY_KEYS}
    worst = max(worst, max(sdiff.values()))
    result["post_switch"] = {"summary_abs_diff": sdiff,
                             "rows": [{k: r[k] for k in ("pattern", "n_subjects", "animal_mean",
                                                         "simulated_mean")} for r in rows]}
    result["max_abs_diff"] = worst
    result["passed"] = worst <= TOL and all(
        d["n_subject_count_mismatches"] == 0 for d in result["depths"].values())
    sources = [{"type": "file", "id": str(p), "sha256": _sha256(p)} for p in paths.values()]
    return result, sources


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-repo", type=Path, required=True)
    a = ap.parse_args()
    models_sha = _git_sha(a.models_repo)
    results, sources = {}, []
    for alias in ALIASES:
        results[alias], src = check_alias(alias)
        sources += src
        print(f"{alias}: max |diff| {results[alias]['max_abs_diff']:.3g}, "
              f"{'PASS' if results[alias]['passed'] else 'FAIL'}", flush=True)
    inputs = [COMMITTED / f"{alias}_{kind}.json" for alias in ALIASES
              for kind in ("history_patterns", "quantitative_summary")]
    meta = build_meta("analysis/freeze_p0_golden_wrapper.py", [], study_root=HERE.parent,
                      kind="freeze", code=[__file__], inputs=inputs, sources=sources)
    write_artifact(OUT, {
        "_meta": meta,
        "fingerprints_library": {"repo": "AllenNeuralDynamics/aind-dynamic-foraging-models",
                                 "git_sha": models_sha},
        "tolerance": TOL,
        "passed": all(r["passed"] for r in results.values()),
        "aliases": results,
    })
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
