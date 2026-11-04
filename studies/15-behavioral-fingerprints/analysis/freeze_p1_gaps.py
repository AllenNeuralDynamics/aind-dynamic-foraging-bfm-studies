#!/usr/bin/env python
"""P1: replay gaps and binomial floors from the committed replay tallies (ADR 0003, 0008).

For every model, regime, subject, session set and rate statistic: the replay gap
(``fingerprints.replay_gap``) and the subject's binomial floor (``fingerprints.binomial_floor``),
then per model, regime, set and statistic the subject-balanced gap, floor and floor ratio
(``fingerprints.floor_ratio``), for every model of ``p1_replay.json`` and ``p1_glm.json``. A set sums the tallies of its session groups
(``p1_session_groups.json``): ``non-warm-up`` (the replay headline; ADR 0004 keeps warm-ups out),
``warm-up``, and each ``{stage_group}|{task}`` group alone. It reads only committed files, but it
needs the fingerprint library, which CI does not install, so it is a freeze and r2 reads its output.

A model and regime enter only if every subject's replay reproduced study 12 (``p1_replay.json``).
The one exception is a 2-level fit that study 12 r3 marks failed (its 2-level score is below its
per-subject score, so r3 places the model by the latter): a cell of such a fit that did not
reproduce is left out and listed in ``excluded``; any other failed cell stops the freeze.

Also writes, for the headline set, each subject's gap and floor (for scatters) and the
subject-mean predicted and observed rate per bin (for the per-bin curves).

Runs anywhere the library imports (no /allen); on a compute node:

    srun --ntasks=1 --cpus-per-task=8 --mem=16G -p aind --time=00:30:00 \\
      env PYTHONPATH=<foraging-models>/src <python> analysis/freeze_p1_gaps.py --models-repo <foraging-models>
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
sys.path.insert(0, str(HERE))
import cross_study_literature as lit  # noqa: E402
from _meta import build_meta, write_artifact  # noqa: E402
from freeze_p0_golden_mevolve import _git_sha  # noqa: E402

from aind_dynamic_foraging_models.fingerprints import (  # noqa: E402
    BINS, MIN_TRIALS, OBS, PRED, SPEC_VERSION, N, add, binomial_floor, empty_tally, floor_ratio,
    kept_bins, replay_gap, values)
from aind_dynamic_foraging_models.fingerprints.spec import RATE_STATISTICS  # noqa: E402

REPLAY = HERE / "p1_replay.json"
GLM = HERE / "p1_glm.json"
TALLIES = HERE / "p1_replay"
GROUPS = HERE / "p1_session_groups.json"
HEADLINE = "non-warm-up"
OUT = HERE / "p1_replay_gaps.json"


def _sets(labels) -> dict[str, set[str]]:
    groups = sorted(set(labels))
    return {HEADLINE: {g for g in groups if g != "warm-up"}, **{g: {g} for g in groups}}


def _animal(rows: dict) -> dict:
    """``{group: tally}`` with N and OBS filled, PRED zero."""
    out = {}
    for g, stats in rows.items():
        t = empty_tally()
        for s, table in stats.items():
            t[s][:, [N, OBS]] = np.asarray(table)
        out[g] = t
    return out


def _with_pred(animal: dict, pred: dict) -> dict:
    out = {}
    for g, t in animal.items():
        m = {s: x.copy() for s, x in t.items()}
        for s in RATE_STATISTICS:
            m[s][:, PRED] = pred[g][s]
        out[g] = m
    return out


def _summed(tallies: dict, groups: set[str]):
    present = [tallies[g] for g in groups if g in tallies]
    if not present:
        return None
    total = empty_tally()
    for t in present:
        total = add(total, t)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-repo", type=Path, required=True)
    a = ap.parse_args()
    models_sha = _git_sha(a.models_repo)
    replay = json.loads(REPLAY.read_text())
    assert replay["spec"] == SPEC_VERSION
    animal_file = TALLIES / "animal.json"
    animal_rows = json.loads(animal_file.read_text())
    assert animal_rows["spec"] == SPEC_VERSION and animal_rows["columns"] == ["N", "OBS"]
    animal = {sid: _animal(rows) for sid, rows in animal_rows["subjects"].items()}
    subjects = sorted(animal)
    sets = _sets(json.loads(GROUPS.read_text())["groups"].values())

    floors = {}   # set -> stat -> subject -> floor
    for name, groups in sets.items():
        floors[name] = {}
        for stat in RATE_STATISTICS:
            floors[name][stat] = {}
            for sid in subjects:
                t = _summed(animal[sid], groups)
                floors[name][stat][sid] = np.nan if t is None else binomial_floor(t, stat)

    glm = json.loads(GLM.read_text())
    assert glm["passed"], "the GLM replay gate failed; fix it before comparing"
    failed_fits = {r["literature_model"] for r in lit.literature_headline_ranking()
                   if r["source"] != "2-level EM"}
    excluded = []
    for model, regimes in replay["models"].items():
        for regime, g in regimes.items():
            if g["passed"]:
                continue
            if regime != "two_level_em" or model not in failed_fits:
                raise SystemExit(f"{model} {regime} does not reproduce study 12; fix it before comparing")
            excluded.append({"model": model, "regime": regime,
                             "n_subjects_replay_differs": g["n_subjects_replay_differs"],
                             "reason": "study 12 r3 marks this 2-level fit failed, and its replay "
                                       "from the stored typical values does not reproduce the "
                                       "stored score"})
    skip = {(e["model"], e["regime"]) for e in excluded}
    inputs, models = [REPLAY, GLM, GROUPS, animal_file, lit.HEADLINE_SOURCE], {}
    for model in list(replay["models"]) + [glm["model"]]:
        path = TALLIES / f"{model}.json"
        inputs.append(path)
        body = json.loads(path.read_text())
        assert body["spec"] == SPEC_VERSION and body["model"] == model
        models[model] = {}
        for regime, r in body["regimes"].items():
            if (model, regime) in skip:
                continue
            flagged = set(r["subjects_replay_differs"])
            tallies = {sid: _with_pred(animal[sid], r["pred"][sid]) for sid in subjects}
            cells = {}
            for name, groups in sets.items():
                cells[name] = {}
                for stat in RATE_STATISTICS:
                    gaps, fl = [], []
                    for sid in subjects:
                        t = _summed(tallies[sid], groups)
                        gaps.append(np.nan if t is None else replay_gap(t, stat))
                        fl.append(floors[name][stat][sid])
                    g, f = np.asarray(gaps), np.asarray(fl)
                    ok = np.isfinite(g) & np.isfinite(f)
                    cell = {"n_subjects": int(ok.sum()),
                            "n_subjects_replay_differs": int(sum(ok[i] for i, s in enumerate(subjects)
                                                                 if s in flagged)),
                            "gap": float(g[ok].mean()) if ok.any() else None,
                            "floor": float(f[ok].mean()) if ok.any() else None,
                            "floor_ratio": floor_ratio(g, f) if ok.any() else None}
                    if name == HEADLINE:
                        cell["per_subject"] = {"gap": [None if not np.isfinite(x) else float(x) for x in g],
                                               "floor": [None if not np.isfinite(x) else float(x) for x in f]}
                        cell["per_bin"] = _per_bin(tallies, subjects, groups, stat)
                    cells[name][stat] = cell
            models[model][regime] = cells
        print(model, {reg: {s: round(c[HEADLINE][s]["floor_ratio"], 3) for s in RATE_STATISTICS}
                      for reg, c in models[model].items()}, flush=True)

    meta = build_meta("analysis/freeze_p1_gaps.py", [], study_root=HERE.parent, kind="freeze",
                      code=[__file__, HERE / "freeze_p0_golden_mevolve.py"], inputs=inputs)
    write_artifact(OUT, {
        "_meta": meta,
        "fingerprints_library": {"repo": "AllenNeuralDynamics/aind-dynamic-foraging-models",
                                 "git_sha": models_sha},
        "spec": SPEC_VERSION,
        "min_trials": MIN_TRIALS,
        "headline_set": HEADLINE,
        "excluded": excluded,
        "sets": {k: sorted(v) for k, v in sets.items()},
        "subjects": subjects,
        "bins": {s: list(BINS[s]) for s in RATE_STATISTICS},
        "models": models,
    })
    print("->", OUT)


def _per_bin(tallies: dict, subjects: list[str], groups: set[str], stat: str) -> dict:
    """Subject means of the predicted and observed rate per bin, over subjects keeping the bin."""
    pred, obs = [], []
    for sid in subjects:
        t = _summed(tallies[sid], groups)
        if t is None:
            continue
        keep = kept_bins(t, stat)
        pred.append(np.where(keep, values(t, stat, PRED), np.nan))
        obs.append(np.where(keep, values(t, stat, OBS), np.nan))
    pred, obs = np.asarray(pred), np.asarray(obs)
    n = np.isfinite(obs).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # a bin no subject keeps is NaN
        mp, mo = np.nanmean(pred, axis=0), np.nanmean(obs, axis=0)
    return {"n_subjects": n.tolist(),
            "model": [None if k == 0 else float(x) for k, x in zip(n, mp)],
            "subject": [None if k == 0 else float(x) for k, x in zip(n, mo)]}


if __name__ == "__main__":
    main()
