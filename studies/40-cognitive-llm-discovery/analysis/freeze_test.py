#!/usr/bin/env python
"""One program on a test cohort, scored once (study 40 ADR 0008, "Test scoring").

The same fit, score, replay and rollouts as ``freeze_selection.py`` (its functions, unchanged: MEvolve
146153c, 2-level, ``max_em`` 100, per-subject fallback, ten rollouts per session, paired task seeds), on
``test`` (the 149 held-out subjects) or ``gru_test`` (the GRU's own 149). It keeps each subject's
tallies on three session sets, ``all`` (the gate's) and study 15's headline sets (``non-warm-up`` for
replay, ``headline`` for rollout; its ``freeze_p2_gaps._sets``), so the report can compare with study
15's models on the same subjects and sessions.

Needs /allen and the pinned code (foraging-models 392c504, MEvolve 146153c) on PYTHONPATH; on a compute
node (24 CPUs, 72 GB):

    python analysis/freeze_test.py --cohort test --program <file> --name <name>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
import zlib
from pathlib import Path

import numpy as np

from freeze_selection import (COHORTS, FREEZE, MEVOLVE_SHA, MODELS_SHA, N_DRAWS, N_ROLLOUTS, S15, SEED, STUDY,
                              _clean, _load, agent_rng, mean_curves, task_seed)

HERE = Path(__file__).resolve().parent
SETS = ("all", "non-warm-up", "headline")


def _summed(tallies: dict, members: set, to_json, add, empty_tally):
    """One set's tally: the sum of its groups' tallies (``all`` is the library's own)."""
    t, n = empty_tally(), 0
    for g, x in tallies.items():
        if g in members:
            t, n = add(t, x), n + 1
    return to_json(t) if n else None


def subject_job(args) -> dict:
    """``freeze_selection.subject_job``, keeping the tallies of the sets in ``sets``."""
    path, name, sid, rec, fell, groups, specs, sessions, sets = args
    import aind_behavior_gym.dynamic_foraging.task  # noqa: F401  (imported before isolation is installed)
    import jax
    import jax.numpy as jnp
    from aind_dynamic_foraging_models.fingerprints import ProgramAgent, replay_fingerprints, rollout, \
        rollout_fingerprints, to_json, add, empty_tally
    from mevolve import fitting, scoring
    from p2_task import build_task

    prog, jitted = _load(path)
    assert [s.session_id for s in sessions] == rec["session_ids"], f"{sid}: other sessions"
    batch = fitting.pad(sessions)
    labels = [groups[s.session_id] for s in sessions]
    choices, rewards = [s.choices for s in sessions], [s.rewards for s in sessions]
    if fell:
        p = scoring.point_p_right(prog, fitting.from_params(prog, {k: rec["params"][k] for k in prog.params}), batch)
    else:
        mu = fitting.from_params(prog, rec["typical_params"])
        log_sigma = jnp.asarray(rec["log_sigma"], dtype=jnp.float64)
        key = jax.random.PRNGKey(SEED + zlib.crc32(sid.encode()) % 100_000)
        p = scoring.marginal_p_right(prog, mu, log_sigma, batch, key, n_draws=N_DRAWS)
    p_right = [np.asarray(p[i, : len(s.choices)]) for i, s in enumerate(sessions)]
    rep_full = replay_fingerprints(choices, rewards, labels, p_right, sets={"all": set(labels)})
    keep = {k: (lambda t, m=m: _summed(t, m, to_json, add, empty_tally)) for k, m in sets.items()}
    tallies = {"replay": {k: f(rep_full["tallies"]) for k, f in keep.items()}}
    tallies["replay"]["all"] = to_json(rep_full["tallies"]["all"])

    real = {s.session_id: s for s in sessions}
    if fell:
        point = {k: float(rec["params"][k]) for k in prog.params}
    else:
        mu_np = np.asarray(fitting.from_params(prog, rec["typical_params"]), float)
        sigma = np.exp(np.asarray(rec["log_sigma"], float))
    rc, rr, rg, oc, orw, og = [], [], [], [], [], []
    for spec in specs:
        x = real[spec["session_id"]]
        assert len(x.choices) == spec["n_trials"], f"{spec['session_id']}: engaged trials differ"
        facts = fitting.session_facts(x)
        oc.append(x.choices), orw.append(x.rewards), og.append(spec["group"])
        for k in range(N_ROLLOUTS):
            rng = agent_rng(name, spec["session_id"], k)
            if fell:
                params = point
            else:
                theta = mu_np + sigma * rng.standard_normal(mu_np.shape)
                params = {kk: float(v) for kk, v in fitting.to_params(prog, jnp.asarray(theta)).items()}
            c, r, _ = rollout(ProgramAgent(jitted, params), build_task(spec, task_seed(spec["session_id"], k)),
                              rng, session=facts)
            rc.append(c), rr.append(r), rg.append(spec["group"])
    rol = {}
    if specs:
        rol_full = rollout_fingerprints(rc, rr, rg, oc, orw, og, sets={"all": set(og)})
        rol = rol_full["sets"]
        tallies |= {"rollout": {k: f(rol_full["tallies"]["rollout"]) for k, f in keep.items()},
                    "rollout_subject": {k: f(rol_full["tallies"]["subject"]) for k, f in keep.items()}}
        tallies["rollout"]["all"] = to_json(rol_full["tallies"]["rollout"]["all"])
        tallies["rollout_subject"]["all"] = to_json(rol_full["tallies"]["subject"]["all"])
    return {"subject_id": sid, "replay": rep_full["sets"], "rollout": rol, "n_rollout_sessions": len(specs),
            "tallies": tallies}


def _all(per: list[dict]) -> list[dict]:
    """The ``all`` tallies in ``freeze_selection.mean_curves``'s shape."""
    return [{"tallies": {k: v["all"] for k, v in p["tallies"].items()}} for p in per]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["test", "gru_test"])
    ap.add_argument("--program", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, help="first N subjects only (a smoke test, never frozen)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    import pandas as pd
    from aind_dynamic_foraging_models.fingerprints import SPEC_VERSION, aggregate_subjects
    from mevolve import evaluate
    from mevolve.data import Freeze
    from mevolve.workers import WORKER_ENV

    import selection_cohorts

    t0 = time.time()
    program_path = str(Path(a.program).resolve())
    subjects = selection_cohorts.subjects(a.cohort, pd.read_parquet(FREEZE / "sessions.parquet"))[: a.limit]
    tasks_path = HERE / "selection" / f"tasks_{a.cohort}.json"
    tasks = json.loads(tasks_path.read_text())
    freeze = Freeze(FREEZE)
    data = [evaluate.subject_data(freeze, s) for s in subjects]
    records = evaluate.evaluate(program_path, data, regime="2level", workers=a.workers, max_em=100,
                                n_draws=N_DRAWS, isolate=True)
    score = evaluate.program_score(records)
    fell = set(score["fallback_subjects"])
    by_subject: dict = {}
    for s in tasks["sessions"]:
        by_subject.setdefault(s["subject_id"], []).append(s)
    from freeze_p2_gaps import _sets

    all_groups = set(tasks["groups"].values())
    sets = {k: v for k, v in _sets(all_groups).items() if k in SETS} | {"all": all_groups}
    jobs = [(program_path, a.name, r["subject_id"], r, r["subject_id"] in fell, tasks["groups"],
             by_subject.get(r["subject_id"], []), d.score, sets) for r, d in zip(records, data)]
    os.environ.update(WORKER_ENV)
    with mp.get_context("spawn").Pool(a.workers) as pool:
        per = pool.map(subject_job, jobs, chunksize=1)
    agg_replay = aggregate_subjects([p["replay"] for p in per])
    agg_rollout = aggregate_subjects([p["rollout"] for p in per if p["rollout"]])
    flat = _all(per)
    out = a.out or HERE / "test" / a.cohort / f"{a.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    from _meta import build_meta, write_artifact

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    in_study = Path(program_path).is_relative_to(STUDY)
    meta = build_meta(f"analysis/{Path(__file__).name}", [], study_root=STUDY, kind="freeze",
                      code=[__file__, HERE / "freeze_selection.py", HERE / "selection_cohorts.py", S15 / "p2_task.py",
                            S15 / "freeze_p2_gaps.py"],
                      inputs=[COHORTS, tasks_path] + ([program_path] if in_study else [])
                      + ([selection_cohorts.GRU_ROWS] if a.cohort == "gru_test" else []),
                      sources=([] if in_study else [{"type": "file", "id": program_path, "sha256": sha(program_path)}]) + [
                          {"type": "file", "id": str(FREEZE / "manifest.json"), "sha256": sha(FREEZE / "manifest.json")},
                          {"type": "url", "id": "https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-mevolve/commit/"
                           + MEVOLVE_SHA},
                          {"type": "url", "id": "https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/commit/"
                           + MODELS_SHA}])
    keep = ("subject_id", "log_lik", "per_subject_log_lik", "n_trials", "em_iterations", "converged")
    body = _clean({
        "name": a.name, "program": program_path, "cohort": a.cohort, "n_subjects": len(subjects),
        "sets": {k: sorted(v) for k, v in sets.items()},
        "settings": {"regime": "2level", "max_em": 100, "n_draws": N_DRAWS, "seed": SEED, "n_rollouts": N_ROLLOUTS,
                     "fallback_tol": evaluate.FALLBACK_TOL, "fingerprint_spec": SPEC_VERSION,
                     "mevolve": MEVOLVE_SHA, "foraging_models": MODELS_SHA},
        "score": score,
        "replay": {"score": agg_replay["all"]["score"], "statistics": agg_replay["all"]["statistics"],
                   "curves": mean_curves(flat, "replay", list(agg_replay["all"]["statistics"]))},
        "rollout": {"score": agg_rollout["all"]["score"], "statistics": agg_rollout["all"]["statistics"],
                    "curves": mean_curves(flat, "rollout", list(agg_rollout["all"]["statistics"])),
                    "n_subjects": sum(1 for p in per if p["rollout"]),
                    "n_rollout_sessions": sum(p["n_rollout_sessions"] for p in per)},
        "subjects": [{**{k: r.get(k) for k in keep}, "fell_back": r["subject_id"] in fell,
                      "replay": p["replay"].get("all"), "rollout": p["rollout"].get("all"), "tallies": p["tallies"]}
                     for r, p in zip(records, per)],
        "seconds": time.time() - t0,
    })
    write_artifact(out, {"_meta": meta, **body})
    print(a.name, a.cohort, {"score": round(100 * score["score"], 4), "fallback": score["n_fallback"]},
          f"{time.time() - t0:.0f}s", "->", out)


if __name__ == "__main__":
    main()
