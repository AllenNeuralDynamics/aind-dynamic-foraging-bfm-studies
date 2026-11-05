#!/usr/bin/env python
"""P2: rollout gaps, split-half floors, coverage and scores from the committed rollout tallies.

For every model, regime and subject, the fingerprint library compares the rollout tallies with the
subject's own score sessions (``compare_rollout``): per session set and statistic, the gap
(``rollout_gap``), the split-half floor of the set's real sessions (``noise_floor``, ADR 0003), their
ratio and the coverage, and the rollout score over all five statistics (ADR 0009); then the
subject-balanced aggregate (``aggregate_subjects``). Sets sum session groups:

- ``headline``: every uncoupled group and the final-stage coupled ones (ADR 0006: the simulator has
  neither coupled hold, which the early coupled stages use);
- ``non-warm-up``: every group but the warm-ups (P1's replay headline);
- each group alone.

Also writes, for the headline set, each subject's gap, floor and coverage, and the subject-mean
rollout and observed value per bin over the subjects every model holds out (the GRU's checkpoint
trained on 3 of the 149), for the curves and scatters. Reads only committed files, but needs the
library, which CI does not install, so it is a freeze; r3 reads its output.

    srun --ntasks=1 --cpus-per-task=40 --mem=64G -p aind --time=01:00:00 \\
      env PYTHONPATH=<foraging-models>/src <python> analysis/freeze_p2_gaps.py --models-repo <foraging-models>
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
sys.path.insert(0, str(HERE))
from _meta import build_meta, write_artifact  # noqa: E402
from freeze_p0_golden_mevolve import _git_sha  # noqa: E402

ROLLOUTS, TALLIES, SPECS, GRU = HERE / "p2_rollouts.json", HERE / "p2_rollouts", HERE / "p2_task_specs.json", HERE / "p1_gru.json"
OUT = HERE / "p2_rollout_gaps.json"
HEADLINE = "headline"


def _sets(groups) -> dict[str, set[str]]:
    groups = sorted(set(groups))
    head = {g for g in groups if g.endswith("|Uncoupled Baiting") or g.endswith("|Uncoupled Without Baiting")
            or (g.startswith("final|") and "|Coupled" in g)}
    return {HEADLINE: head, "non-warm-up": {g for g in groups if g != "warm-up"}, **{g: {g} for g in groups}}


def _tables(path: Path) -> dict:
    """``{regime: {subject: {group: tally}}}`` (or ``{subject: {session: (group, tally)}}``)."""
    import pandas as pd

    from aind_dynamic_foraging_models.fingerprints import BINS, OBS, N, empty_tally

    index = {st: {lab: i for i, lab in enumerate(BINS[st])} for st in BINS}
    df = pd.read_parquet(path)
    keys = ["regime", "subject_id", "group"] if "regime" in df else ["subject_id", "session_id", "group"]
    out: dict = {}
    for key, rows in df.groupby(keys, sort=True):
        t = empty_tally()
        for st, part in rows.groupby("statistic"):
            idx = [index[st][b] for b in part["bin"]]
            t[st][idx, N] = part["n"].to_numpy()
            t[st][idx, OBS] = part["obs"].to_numpy()
        a, b, g = key
        out.setdefault(a, {})[b] = (g, t) if "session_id" in keys else {**out.get(a, {}).get(b, {}), g: t}
    return out


def _model_job(args) -> dict:
    """One model: every regime's per-subject comparison, aggregate, and headline per-bin curves."""
    name, sets, subjects, common = args
    from aind_dynamic_foraging_models.fingerprints import (STATISTICS, aggregate_subjects, compare_rollout,
                                                           kept_bins, values)

    animal = _ANIMAL
    tallies = _tables(TALLIES / f"{name}.parquet")
    out = {}
    for regime, by_subject in tallies.items():
        per = {}
        for sid in subjects:
            sessions = animal.get(sid, {})
            if sid not in by_subject or not sessions:
                continue
            per[sid] = compare_rollout(by_subject[sid], [t for _, t in sessions.values()],
                                       [g for g, _ in sessions.values()], sets)
        agg = aggregate_subjects(list(per.values()))
        cells = {}
        for set_name, members in sets.items():
            cells[set_name] = {"score": _f(agg[set_name]["score"]) if set_name in agg else None,
                               "statistics": {}}
            for st in STATISTICS:
                a = agg.get(set_name, {}).get("statistics", {}).get(st, {})
                cell = {k: _f(a.get(k)) for k in ("gap", "floor", "floor_ratio", "coverage")}
                cell["n_subjects"] = int(a.get("n_subjects", 0))
                if set_name == HEADLINE:
                    cell["per_subject"] = {k: [_f(per[s][set_name]["statistics"][st][k]) if s in per else None
                                               for s in subjects] for k in ("gap", "floor", "coverage")}
                    cell["per_bin_common"] = _per_bin(by_subject, animal, common, members, st, kept_bins, values)
                cells[set_name]["statistics"][st] = cell
        out[regime] = cells
    return {"model": name, "regimes": out}


def _f(x):
    return None if x is None or not np.isfinite(x) else float(x)


def _per_bin(by_subject, animal, subjects, members, stat, kept_bins, values) -> dict:
    """Subject means of the rollout's and the subject's value per bin, over subjects keeping it."""
    from aind_dynamic_foraging_models.fingerprints import add, empty_tally

    model, obs = [], []
    for sid in subjects:
        if sid not in by_subject or sid not in animal:
            continue
        own = [t for g, t in animal[sid].values() if g in members]
        rolled = [t for g, t in by_subject[sid].items() if g in members]
        if not own or not rolled:
            continue
        s, m = empty_tally(), empty_tally()
        for t in own:
            s = add(s, t)
        for t in rolled:
            m = add(m, t)
        keep = kept_bins(s, stat)
        obs.append(np.where(keep, values(s, stat), np.nan))
        model.append(np.where(keep, values(m, stat), np.nan))
    obs, model = np.asarray(obs), np.asarray(model)
    n = np.isfinite(obs).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mo, so = np.nanmean(model, axis=0), np.nanmean(obs, axis=0)
    return {"n_subjects": n.tolist(), "model": [None if k == 0 or not np.isfinite(x) else float(x) for k, x in zip(n, mo)],
            "subject": [None if k == 0 else float(x) for k, x in zip(n, so)]}


_ANIMAL: dict = {}


def _init(animal):
    global _ANIMAL
    _ANIMAL = animal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-repo", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=40)
    a = ap.parse_args()
    from aind_dynamic_foraging_models.fingerprints import BINS, MIN_TRIALS, SPEC_VERSION, STATISTICS

    runs = json.loads(ROLLOUTS.read_text())
    assert runs["passed"], "a P2 rollout check failed; fix it before comparing"
    assert runs["spec"] == SPEC_VERSION
    specs = json.loads(SPECS.read_text())
    subjects = sorted({s["subject_id"] for s in specs["sessions"]})
    trained = set(json.loads(GRU.read_text())["seeds"]["seed0"]["subjects_in_training"])
    common = [s for s in subjects if s not in trained]
    sets = _sets(s["group"] for s in specs["sessions"])
    animal = _tables(TALLIES / "animal_sessions.parquet")
    names = list(runs["models"])
    with mp.get_context("spawn").Pool(min(a.workers, len(names)), initializer=_init, initargs=(animal,)) as pool:
        res = pool.map(_model_job, [(n, sets, subjects, common) for n in names], chunksize=1)
    models = {r["model"]: r["regimes"] for r in res}
    for m, regimes in models.items():
        print(m, {reg: (c[HEADLINE]["score"] and round(c[HEADLINE]["score"], 2)) for reg, c in regimes.items()}, flush=True)

    sha = _git_sha(a.models_repo)
    library = {"type": "url", "state": f"git {sha}",
               "id": f"https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/tree/{sha}"
                     "/src/aind_dynamic_foraging_models/fingerprints"}
    inputs = [ROLLOUTS, SPECS, GRU, TALLIES / "animal_sessions.parquet", *(TALLIES / f"{n}.parquet" for n in names)]
    meta = build_meta("analysis/freeze_p2_gaps.py", [], study_root=HERE.parent, kind="freeze",
                      code=[__file__, HERE / "freeze_p0_golden_mevolve.py"], inputs=inputs, sources=[library])
    write_artifact(OUT, {
        "_meta": meta,
        "fingerprints_library": {"repo": "AllenNeuralDynamics/aind-dynamic-foraging-models", "git_sha": sha},
        "spec": SPEC_VERSION, "min_trials": MIN_TRIALS, "headline_set": HEADLINE,
        "sets": {k: sorted(v) for k, v in sets.items()}, "subjects": subjects, "common_subjects": common,
        "bins": {s: list(BINS[s]) for s in STATISTICS}, "models": models,
    })
    print("->", OUT)


if __name__ == "__main__":
    main()
