#!/usr/bin/env python
"""One program on one cohort: held-out likelihood, replay and rollout fingerprint scores (study 40 ADR 0008).

The program is fitted per subject with MEvolve's 2-level Laplace-EM (``max_em`` 100, per-mouse fallback,
MEvolve ADR 0004) and scored on the subjects' odd sessions. Then, with study 15's fingerprint library
(spec v1; study 15 ADR 0009):

- **replay**: the session marginal's p(right) on every score session (MEvolve's draws: 2,048, the key
  of ``evaluate``), tallied by the session groups of ``selection/tasks_<cohort>.json``; a mouse whose
  2-level fit fell back is replayed with its per-subject fit, as it is scored;
- **rollout**: ``N_ROLLOUTS`` plays of each score session's gym task (study 15's ``build_task``), each
  drawing its parameters from the subject's 2-level fit (theta = mu + exp(log sigma) * eps on the
  unconstrained scale, study 15 ADR 0005; the per-subject fit for a fallen-back mouse). The task is
  seeded by (session, rollout) only, so every program plays the same tasks; the agent by (program,
  session, rollout).

Per subject, ``replay_fingerprints`` and ``rollout_fingerprints`` on the ``all`` set; then
``aggregate_subjects`` gives each mode's subject-balanced floor ratios and fingerprint score, and
``mean_curves`` the mouse-average curves and their per-bin RMSE (study 15's Figure 5 view). Each
subject's ``all`` tallies are kept, so other statistics can be drawn later without refitting.

Needs /allen and the pinned code (foraging-models 392c504, MEvolve 146153c) on PYTHONPATH; on a compute
node, one job per program (24 CPUs, 72 GB: 12 fitting workers):

    sbatch -p aind --ntasks=1 --cpus-per-task=24 --mem-per-cpu=3000M --time=12:00:00 --wrap \\
      "env PYTHONPATH=<foraging-models>/src:<mevolve>/src <mevolve-search env>/bin/python \\
       analysis/freeze_selection.py --cohort selection --program <file> --name <name>"
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
S15 = STUDY.parent / "15-behavioral-fingerprints" / "analysis"
sys.path.insert(0, str(STUDY.parent / "util"))
sys.path.insert(0, str(S15))

FREEZE = Path("/allen/aind/scratch/han.hou/mevolve/freeze/20260603")
COHORTS = STUDY / "reference" / "cohorts_seed0.json"
N_ROLLOUTS, SEED, N_DRAWS = 10, 0, 2048
MEVOLVE_SHA = "146153c01f6391122414525022e9126f200a1586"   # ADR 0008
MODELS_SHA = "392c504406094b33407321b168173981b8af8291"   # foraging-models with the fingerprint library (ADR 0008)

_PROG: dict = {}


def _load(path: str):
    if path not in _PROG:
        import jax
        from mevolve import isolation, program

        isolation.install(path)          # the candidate's code has no file or network access here either
        prog = program.load(path)

        from types import SimpleNamespace
        _PROG[path] = (prog, SimpleNamespace(init_latents=prog.init_latents, predict=jax.jit(prog.predict),
                                             update=jax.jit(prog.update)))
    return _PROG[path]


def task_seed(session_id: str, k: int) -> int:
    ss = np.random.SeedSequence([SEED, zlib.crc32(session_id.encode()), k])
    return int(ss.generate_state(1)[0])


def agent_rng(name: str, session_id: str, k: int):
    return np.random.default_rng(np.random.SeedSequence([SEED, zlib.crc32(name.encode()),
                                                         zlib.crc32(session_id.encode()), k]))


def subject_job(args) -> dict:
    """Replay and rollout fingerprints of one subject (``all`` set)."""
    path, name, sid, rec, fell, groups, specs, sessions = args
    import aind_behavior_gym.dynamic_foraging.task  # noqa: F401  (imported before isolation is installed)
    import jax
    import jax.numpy as jnp
    from aind_dynamic_foraging_models.fingerprints import ProgramAgent, replay_fingerprints, rollout, \
        rollout_fingerprints, to_json
    from mevolve import fitting, scoring
    from p2_task import build_task

    prog, jitted = _load(path)     # installs isolation: the data come from the parent
    assert [s.session_id for s in sessions] == rec["session_ids"], f"{sid}: other sessions"
    batch = fitting.pad(sessions)
    labels = [groups[s.session_id] for s in sessions]
    choices, rewards = [s.choices for s in sessions], [s.rewards for s in sessions]

    if fell:
        x = fitting.from_params(prog, {k: rec["params"][k] for k in prog.params})
        p = scoring.point_p_right(prog, x, batch)
    else:
        mu = fitting.from_params(prog, rec["typical_params"])
        log_sigma = jnp.asarray(rec["log_sigma"], dtype=jnp.float64)
        key = jax.random.PRNGKey(SEED + zlib.crc32(sid.encode()) % 100_000)   # evaluate's draws
        p = scoring.marginal_p_right(prog, mu, log_sigma, batch, key, n_draws=N_DRAWS)
    p_right = [np.asarray(p[i, : len(s.choices)]) for i, s in enumerate(sessions)]
    sets = {"all": set(labels)}
    rep_full = replay_fingerprints(choices, rewards, labels, p_right, sets=sets)
    rep = rep_full["sets"]
    tallies = {"replay": to_json(rep_full["tallies"]["all"])}

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
    if specs:
        rol_full = rollout_fingerprints(rc, rr, rg, oc, orw, og, sets={"all": set(og)})
        rol = rol_full["sets"]
        tallies |= {"rollout": to_json(rol_full["tallies"]["rollout"]["all"]),
                    "rollout_subject": to_json(rol_full["tallies"]["subject"]["all"])}
    else:
        rol = {}
    return {"subject_id": sid, "replay": rep, "rollout": rol, "n_rollout_sessions": len(specs), "tallies": tallies}


def mean_curves(per: list[dict], mode: str, stats: list[str]) -> dict:
    """The mouse-average curves and their per-bin RMSE (study 15's ``_per_bin`` and Figure 5's RMSE).

    Per statistic: each subject's value per bin where its own tally keeps the bin (``kept_bins``); the
    mean over subjects per bin, for the subjects and for the model (replay: the predicted rate of the
    same tally; rollout: the rate in the subject's pooled rollouts); RMSE over the bins where both means
    are finite, each bin once. ``curve_score`` is the geometric mean of the statistics' RMSEs.
    """
    import warnings

    from aind_dynamic_foraging_models.fingerprints import OBS, PRED, from_json, kept_bins, values

    out = {}
    for st in stats:
        model, obs = [], []
        for p in per:
            t = p["tallies"]
            if mode == "replay":
                own = from_json(t["replay"])
                m = values(own, st, PRED)
            elif "rollout" in t:
                own, rolled = from_json(t["rollout_subject"]), from_json(t["rollout"])
                m = values(rolled, st)
            else:
                continue
            keep = kept_bins(own, st)
            obs.append(np.where(keep, values(own, st, OBS), np.nan))
            model.append(np.where(keep, m, np.nan))
        obs, model = np.asarray(obs), np.asarray(model)
        n = np.isfinite(obs).sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mo, so = np.nanmean(model, axis=0), np.nanmean(obs, axis=0)
        ok = np.isfinite(mo) & np.isfinite(so) & (n > 0)
        out[st] = {"n_subjects": n.tolist(), "model": mo.tolist(), "subject": so.tolist(),
                   "rmse": float(np.sqrt(np.mean((mo[ok] - so[ok]) ** 2)))}
    return {"statistics": out, "curve_score": float(np.exp(np.mean([np.log(v["rmse"]) for v in out.values()])))}


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["generation", "selection", "test", "gru_test"])
    ap.add_argument("--program", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-em", type=int, default=100)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--limit", type=int, help="first N subjects only (a smoke test, never frozen)")
    a = ap.parse_args()
    from aind_dynamic_foraging_models.fingerprints import SPEC_VERSION, aggregate_subjects
    from mevolve import evaluate
    from mevolve.data import Freeze
    from mevolve.workers import WORKER_ENV

    t0 = time.time()
    program_path = str(Path(a.program).resolve())
    import pandas as pd
    import selection_cohorts

    subjects = selection_cohorts.subjects(a.cohort, pd.read_parquet(FREEZE / "sessions.parquet"))[: a.limit]
    tasks = json.loads((HERE / "selection" / f"tasks_{a.cohort}.json").read_text())
    freeze = Freeze(FREEZE)
    data = [evaluate.subject_data(freeze, s) for s in subjects]
    records = evaluate.evaluate(program_path, data, regime="2level", workers=a.workers, max_em=a.max_em,
                                n_draws=N_DRAWS, isolate=True)
    score = evaluate.program_score(records)
    fell = set(score["fallback_subjects"])
    by_subject: dict = {}
    for s in tasks["sessions"]:
        by_subject.setdefault(s["subject_id"], []).append(s)
    jobs = [(program_path, a.name, r["subject_id"], r, r["subject_id"] in fell, tasks["groups"],
             by_subject.get(r["subject_id"], []), d.score) for r, d in zip(records, data)]
    os.environ.update(WORKER_ENV)
    with mp.get_context("spawn").Pool(a.workers) as pool:
        per = pool.map(subject_job, jobs, chunksize=1)
    agg_replay = aggregate_subjects([p["replay"] for p in per])
    agg_rollout = aggregate_subjects([p["rollout"] for p in per if p["rollout"]])
    out = a.out or HERE / "selection" / a.cohort / f"{a.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    from _meta import build_meta, write_artifact
    import hashlib

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    in_study = Path(program_path).is_relative_to(STUDY)
    tasks_path = HERE / "selection" / f"tasks_{a.cohort}.json"
    meta = build_meta(f"analysis/{Path(__file__).name}", [], study_root=STUDY, kind="freeze",
                      code=[__file__, HERE / "selection_cohorts.py", S15 / "p2_task.py"],
                      inputs=[COHORTS, tasks_path] + ([selection_cohorts.GRU_ROWS] if a.cohort == "gru_test" else []) + ([program_path] if in_study else []),
                      sources=([] if in_study else [{"type": "file", "id": program_path, "sha256": sha(program_path)}]) + [
                          {"type": "file", "id": str(FREEZE / "manifest.json"), "sha256": sha(FREEZE / "manifest.json")},
                          {"type": "url", "id": "https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-mevolve/commit/"
                           + MEVOLVE_SHA},
                          {"type": "url", "id": "https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/commit/"
                           + MODELS_SHA}])
    keep = ("subject_id", "log_lik", "per_subject_log_lik", "n_trials", "em_iterations", "converged")
    body = _clean({
        "name": a.name, "program": program_path, "cohort": a.cohort, "n_subjects": len(subjects),
        "settings": {"regime": "2level", "max_em": a.max_em, "n_draws": N_DRAWS, "seed": SEED,
                     "n_rollouts": N_ROLLOUTS, "fallback_tol": evaluate.FALLBACK_TOL, "fingerprint_spec": SPEC_VERSION,
                     "mevolve": MEVOLVE_SHA, "foraging_models": MODELS_SHA},
        "score": {k: v for k, v in score.items()},
        "replay": {"score": agg_replay["all"]["score"], "statistics": agg_replay["all"]["statistics"],
                   "curves": mean_curves(per, "replay", list(agg_replay["all"]["statistics"]))},
        "rollout": {"score": agg_rollout["all"]["score"], "statistics": agg_rollout["all"]["statistics"],
                    "curves": mean_curves(per, "rollout", list(agg_rollout["all"]["statistics"])),
                    "n_subjects": sum(1 for p in per if p["rollout"]),
                    "n_rollout_sessions": sum(p["n_rollout_sessions"] for p in per)},
        "subjects": [{**{k: r.get(k) for k in keep}, "fell_back": r["subject_id"] in fell,
                      "replay": p["replay"].get("all"), "rollout": p["rollout"].get("all"), "tallies": p["tallies"]}
                     for r, p in zip(records, per)],
        "seconds": time.time() - t0,
    })
    write_artifact(out, {"_meta": meta, **body})
    print(a.name, {"score": round(100 * score["score"], 4), "fallback": score["n_fallback"],
                   "replay": agg_replay["all"]["score"], "rollout": agg_rollout["all"]["score"],
                   "replay_rmse": body["replay"]["curves"]["curve_score"],
                   "rollout_rmse": body["rollout"]["curves"]["curve_score"]},
          f"{time.time() - t0:.0f}s", "->", out)


if __name__ == "__main__":
    main()
