#!/usr/bin/env python
"""P1: replay gaps and binomial floors from the committed replay tallies (ADR 0003, 0008).

For every model, regime, subject, session set and rate statistic: the replay gap and the subject's
binomial floor (``fingerprints.compare_replay``), then per model, regime, set and statistic the
subject-balanced gap, floor and floor ratio, and per set the replay fingerprint score of ADR 0009
(``fingerprints.aggregate_subjects``), for every model of ``p1_replay.json``, ``p1_glm.json`` and
``p1_gru.json`` (the GRU's regimes are its seeds; the subjects its checkpoint was trained on are
left out of its numbers). A set sums the tallies of its session groups
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
    BINS, MIN_TRIALS, OBS, PRED, SPEC_VERSION, N, add, aggregate_subjects, compare_replay, empty_tally,
    kept_bins, values)
from aind_dynamic_foraging_models.fingerprints.spec import RATE_STATISTICS  # noqa: E402

REPLAY = HERE / "p1_replay.json"
GLM = HERE / "p1_glm.json"
GRU = HERE / "p1_gru.json"
TALLIES = HERE / "p1_replay"
GROUPS = HERE / "p1_session_groups.json"
HEADLINE = "non-warm-up"
OUT = HERE / "p1_replay_gaps.json"


def _sets(labels) -> dict[str, set[str]]:
    groups = sorted(set(labels))
    return {HEADLINE: {g for g in groups if g != "warm-up"}, **{g: {g} for g in groups}}


def _bin_index() -> dict:
    return {stat: {lab: i for i, lab in enumerate(BINS[stat])} for stat in BINS}


def _animal(path: Path) -> dict:
    """``{subject: {group: tally}}`` with N and OBS filled, PRED zero, from ``animal.parquet``."""
    import pandas as pd

    index, out = _bin_index(), {}
    for (sid, g), rows in pd.read_parquet(path).groupby(["subject_id", "group"], sort=True):
        t = empty_tally()
        for r in rows.itertuples(index=False):
            t[r.statistic][index[r.statistic][r.bin], [N, OBS]] = (r.n, r.obs)
        out.setdefault(sid, {})[g] = t
    return out


def _preds(path: Path) -> dict:
    """``{regime: {subject: {group: {statistic: PRED per bin}}}}`` from a model's parquet."""
    import pandas as pd

    index, out = _bin_index(), {}
    for (regime, sid, g, stat), rows in pd.read_parquet(path).groupby(
            ["regime", "subject_id", "group", "statistic"], sort=True):
        v = np.zeros(len(BINS[stat]))
        v[[index[stat][b] for b in rows["bin"]]] = rows["pred"].to_numpy()
        out.setdefault(regime, {}).setdefault(sid, {}).setdefault(g, {})[stat] = v
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
    animal_file = TALLIES / "animal.parquet"
    animal = _animal(animal_file)
    subjects = sorted(animal)
    sets = _sets(json.loads(GROUPS.read_text())["groups"].values())

    glm = json.loads(GLM.read_text())
    assert glm["passed"], "the GLM replay gate failed; fix it before comparing"
    assert glm["spec"] == SPEC_VERSION
    gru = json.loads(GRU.read_text())
    assert gru["passed"], "the GRU replay gate failed; fix it before comparing"
    assert gru["spec"] == SPEC_VERSION
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
    # the subjects every model holds out (the GRU's checkpoint trained on 3 of the 149): the
    # per-bin curves drawn side by side use these, so every curve is over the same subjects
    not_held = {sid for g in gru["seeds"].values() for sid in g["subjects_in_training"]}
    common = [sid for sid in subjects if sid not in not_held]
    inputs, models, scores = [REPLAY, GLM, GRU, GROUPS, animal_file, lit.HEADLINE_SOURCE], {}, {}
    for model in list(replay["models"]) + [glm["model"], gru["model"]]:
        path = TALLIES / f"{model}.parquet"
        inputs.append(path)
        gates = (gru["seeds"] if model == gru["model"] else   # one regime per GRU seed
                 replay["models"].get(model, {"per_subject": {"subjects_replay_differs": []}}))
        models[model], scores[model] = {}, {}
        runs = {}
        for regime, preds in _preds(path).items():
            if (model, regime) in skip:
                continue
            flagged = set(gates[regime]["subjects_replay_differs"])
            trained = set(gates[regime].get("subjects_in_training", []))   # not held out for the GRU
            held = [sid for sid in subjects if sid not in trained]
            tallies = {sid: _with_pred(animal[sid], preds[sid]) for sid in subjects}
            # the library's one-call comparison per subject, then its subject-balanced aggregate
            per_subject = [_left_out(sets) if sid in trained else compare_replay(tallies[sid], sets)
                           for sid in subjects]
            runs[regime] = per_subject
            models[model][regime] = _cells(
                per_subject, sets, subjects, flagged,
                lambda groups, stat: {"per_bin": _per_bin(tallies, held, groups, stat),
                                      "per_bin_common": _per_bin(tallies, common, groups, stat)})
            scores[model][regime] = _scores(per_subject, sets)
        if model == gru["model"]:   # the mean over seeds, in the grid's per-subject column
            seeds = [models[model][r] for r in runs]
            per_subject = _seed_mean(list(runs.values()), sets)
            models[model]["per_subject"] = _cells(
                per_subject, sets, subjects, set(),
                lambda groups, stat: {k: _mean_per_bin(seeds, stat, k) for k in ("per_bin", "per_bin_common")})
            scores[model]["per_subject"] = _scores(per_subject, sets)
        print(model, {reg: {s: round(c[HEADLINE][s]["floor_ratio"], 3) for s in RATE_STATISTICS}
                      for reg, c in models[model].items()}, flush=True)

    # Its only external input is the library's code, at the commit it ran (the reason this is a freeze).
    library = {"type": "url", "state": f"git {models_sha}",
               "id": "https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/tree/"
                     f"{models_sha}/src/aind_dynamic_foraging_models/fingerprints"}
    meta = build_meta("analysis/freeze_p1_gaps.py", [], study_root=HERE.parent, kind="freeze",
                      code=[__file__, HERE / "freeze_p0_golden_mevolve.py"], inputs=inputs,
                      sources=[library])
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
        "scores": scores,
    })
    print("->", OUT)


def _left_out(sets: dict) -> dict:
    """A subject a model does not count (NaN everywhere, so the aggregate skips it)."""
    nan = float("nan")
    return {name: {"statistics": {stat: {"gap": nan, "floor": nan, "floor_ratio": nan}
                                  for stat in RATE_STATISTICS}, "score": nan} for name in sets}


def _scores(per_subject: list[dict], sets: dict) -> dict:
    agg = aggregate_subjects(per_subject)
    return {name: (agg[name]["score"] if np.isfinite(agg[name]["score"]) else None) for name in sets}


def _cells(per_subject: list[dict], sets: dict, subjects: list[str], flagged: set[str], per_bin) -> dict:
    """Per set and rate statistic: the subject-balanced aggregate, and for the headline set each
    subject's gap and floor and the per-bin subject means (``per_bin(groups, stat)``: over the
    subjects the model counts, and over the subjects every model holds out)."""
    agg = aggregate_subjects(per_subject)
    cells = {}
    for name, groups in sets.items():
        cells[name] = {}
        for stat in RATE_STATISTICS:
            g = np.array([r[name]["statistics"][stat]["gap"] for r in per_subject])
            f = np.array([r[name]["statistics"][stat]["floor"] for r in per_subject])
            ok = np.isfinite(g) & np.isfinite(f)
            a = agg[name]["statistics"][stat]
            cell = {"n_subjects": a["n_subjects"],
                    "n_subjects_replay_differs": int(sum(ok[i] for i, s in enumerate(subjects)
                                                         if s in flagged)),
                    "gap": a["gap"] if ok.any() else None,
                    "floor": a["floor"] if ok.any() else None,
                    "floor_ratio": a["floor_ratio"] if ok.any() else None}
            if name == HEADLINE:
                cell["per_subject"] = {"gap": [None if not np.isfinite(x) else float(x) for x in g],
                                       "floor": [None if not np.isfinite(x) else float(x) for x in f]}
                cell.update(per_bin(groups, stat))
            cells[name][stat] = cell
    return cells


def _seed_mean(runs: list[list[dict]], sets: dict) -> list[dict]:
    """Per subject, set and statistic: the mean gap over a model's seeds (a typical seed, as study
    01 averages seeds), with the subject's floor, which is the same for every seed."""
    out = []
    for i in range(len(runs[0])):
        out.append({name: {"statistics": {stat: {
            "gap": float(np.mean([run[i][name]["statistics"][stat]["gap"] for run in runs])),
            "floor": runs[0][i][name]["statistics"][stat]["floor"]} for stat in RATE_STATISTICS}}
            for name in sets})
    return out


def _mean_per_bin(seeds: list[dict], stat: str, key: str) -> dict:
    """The headline per-bin curves averaged over seeds (the subjects' own curve is shared)."""
    curves = [cells[HEADLINE][stat][key] for cells in seeds]
    model = np.array([_nan(c["model"]) for c in curves]).mean(axis=0)
    return {"n_subjects": curves[0]["n_subjects"],
            "model": [None if not np.isfinite(x) else float(x) for x in model],
            "subject": curves[0]["subject"]}


def _nan(values) -> list[float]:
    return [np.nan if v is None else v for v in values]


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
