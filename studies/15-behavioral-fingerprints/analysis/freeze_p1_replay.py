#!/usr/bin/env python
"""P1: replay fingerprints of the literature models, per subject and as 2-level EM (ADR 0001, 0005).

Each model replays the score (held-out, odd) sessions of study 12's 149 test subjects (ADR 0007),
and the library tallies its predicted switch probability per bin, by the session groups of
``p1_session_groups.json``. Before a subject's tally counts, the replay must reproduce the score
study 12 stored for it, so the model, its parameters and the sessions are the ones study 12 ranked:

- **per subject**: study 12's fitted parameters (``cross_study_literature``), replayed through the
  model's MEvolve program, which matches the foraging-models forager to 1e-6 per trial (MEvolve
  gate 1a). Gate: the
  summed log likelihood equals study 12's (clipped as it clips) to ``TOL_LOG_LIK``.
- **2-level EM**: the session marginal's predicted p(right) (ADR 0005), from each subject's typical
  values and spreads in study 12's 2-level files, with MEvolve's draws (2,048, common random draws
  per subject). Gates: the summed session-marginal log likelihood equals the stored one to
  ``TOL_LOG_LIK``, and MEvolve's ``replay_gap`` rebuilt from the library's tallies equals the stored
  fingerprints to ``TOL_FINGERPRINT`` (P0's golden check, on these fits).

A subject that fails a gate is counted per model and regime (``n_subjects_replay_differs``) and its
tally is still written, flagged, but the freeze fails. (The foraging-models release study 12 fit
with, v0.26.0, chose a random side when a logit passed 700, which a program cannot reproduce; no
test subject's ActorCritic fit gets there.)

Writes ``p1_replay.json`` (gates, and the digest of every tally file) and ``p1_replay/``, in
parquet: ``animal.parquet`` (the subjects' own tallies, N and OBS per bin) and one file per model
with the summed prediction (PRED) per regime, subject, group and bin; a model's full tally is the
animal tally with that PRED column (ADR 0004).

Needs /allen and the mevolve env; on a compute node:

    sbatch -p aind --ntasks=1 --cpus-per-task=56 --mem=400G --time=04:00:00 --wrap \\
      "env PYTHONPATH=<foraging-models>/src:<mevolve>/src <mevolve env>/bin/python \\
       analysis/freeze_p1_replay.py --models-repo <foraging-models> --mevolve-repo <mevolve>"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
sys.path.insert(0, str(HERE))
import cross_study_literature as lit  # noqa: E402
from _meta import build_meta, write_artifact  # noqa: E402
from freeze_p0_golden_mevolve import FREEZE, FREEZE_MANIFEST, _git_sha, _max_diff  # noqa: E402

GROUPS = HERE / "p1_session_groups.json"
TWO_LEVEL_INDEX = lit.REPO / "studies" / "12-cognitive-literature-leaderboard" / "analysis" / "two_level_em_fits.json"
TOL_LOG_LIK, TOL_FINGERPRINT = 1e-6, 1e-9
CLIP = 1e-10
OUT = HERE / "p1_replay.json"
OUT_DIR = HERE / "p1_replay"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _pred(tallies: dict) -> dict:
    """The PRED column of each rate statistic, per group ("all" is their sum, so not stored)."""
    from aind_dynamic_foraging_models.fingerprints import PRED
    from aind_dynamic_foraging_models.fingerprints.spec import RATE_STATISTICS

    return {g: {s: t[s][:, PRED].tolist() for s in RATE_STATISTICS} for g, t in tallies.items() if g != "all"}


def _program(name: str):
    from mevolve import literature, program

    return program.literature(literature.LITERATURE[name][0])


def _subject_job(args) -> dict:
    """One literature model on one subject: both regimes, their gates and their tallies."""
    name, sid, sessions, labels, params, stored_ps, rec, seed, n_draws = args
    import jax
    import jax.numpy as jnp
    from mevolve import fitting, scoring

    from aind_dynamic_foraging_models.fingerprints import mevolve_replay_gaps, tally_groups

    prog = _program(name)
    k = len(sessions)
    choices, rewards = [s.choices for s in sessions], [s.rewards for s in sessions]
    batch = fitting.pad(sessions)

    def per_trial(p):
        return [np.asarray(p[i, : len(s.choices)]) for i, s in enumerate(sessions)]

    missing = set(prog.params) - set(params)
    assert not missing, f"{name} {sid}: study 12 has no {missing}"
    x = fitting.from_params(prog, {key: params[key] for key in prog.params})
    ll, n = scoring.point_scores(prog, x, batch)
    p_right = per_trial(scoring.point_p_right(prog, x, batch))
    ps = {"log_lik_diff": float(ll[:k].sum()) - stored_ps[0],
          "n_trials_equal": int(n[:k].sum()) == stored_ps[1],
          "pred": _pred(tally_groups(choices, rewards, labels, p_right))}

    assert rec["session_ids"] == [s.session_id for s in sessions], f"{name} {sid}: other sessions"
    mu = fitting.from_params(prog, rec["typical_params"])
    log_sigma = jnp.asarray(rec["log_sigma"], dtype=jnp.float64)
    key = jax.random.PRNGKey(seed + zlib.crc32(sid.encode()) % 100_000)   # mevolve.evaluate's draws
    m = scoring.marginal_scores(prog, mu, log_sigma, batch, key, n_draws=n_draws)
    p_right = per_trial(scoring.marginal_p_right(prog, mu, log_sigma, batch, key, n_draws=n_draws))
    mevolve_groups = [f"{s.stage_group}|{s.task}" for s in sessions]
    gaps = mevolve_replay_gaps(tally_groups(choices, rewards, mevolve_groups, p_right))
    fp_diff, fp_unshared = _max_diff(gaps, rec["fingerprints"])
    tl = {"log_lik_diff": float(m["log_lik"][:k].sum()) - rec["log_lik"],
          "n_trials_equal": int(m["n_trials"][:k].sum()) == rec["n_trials"],
          "fingerprint_diff": fp_diff, "fingerprint_unshared": fp_unshared,
          "pred": _pred(tally_groups(choices, rewards, labels, p_right))}
    return {"subject_id": sid, "per_subject": ps, "two_level_em": tl}


def _gate(rows: list[dict], regime: str) -> dict:
    """Per model and regime: how many subjects reproduce study 12, and the largest misses."""
    r = [row[regime] for row in rows]
    differs = [not math.isfinite(x["log_lik_diff"]) or abs(x["log_lik_diff"]) > TOL_LOG_LIK
               or not x["n_trials_equal"] for x in r]   # NaN never counts as reproduced
    out = {"n_subjects": len(r), "n_subjects_replay_differs": int(sum(differs)),
           "max_abs_log_lik_diff": float(max(abs(x["log_lik_diff"]) if math.isfinite(x["log_lik_diff"])
                                             else math.inf for x in r)),
           "subjects_replay_differs": sorted(row["subject_id"] for row, d in zip(rows, differs) if d)}
    if regime == "two_level_em":
        same = [x for x, d in zip(r, differs) if not d]
        out["max_fingerprint_diff"] = float(max((x["fingerprint_diff"] for x in same), default=0.0))
        out["n_fingerprint_unshared"] = int(sum(x["fingerprint_unshared"] for x in same))
    out["passed"] = out["n_subjects_replay_differs"] == 0 and (
        regime != "two_level_em" or (out["max_fingerprint_diff"] <= TOL_FINGERPRINT
                                     and out["n_fingerprint_unshared"] == 0))
    return out


def pred_frame(rows: list[dict], regimes) -> "pd.DataFrame":
    """The PRED tallies as one long table: a row per regime, subject, group, statistic and bin."""
    import pandas as pd
    from aind_dynamic_foraging_models.fingerprints import BINS

    rec = []
    for regime in regimes:
        for row in rows:
            for g, stats in row[regime]["pred"].items():
                for stat, vals in stats.items():
                    rec += [(regime, row["subject_id"], g, stat, lab, float(v)) for lab, v in zip(BINS[stat], vals)]
    return pd.DataFrame(rec, columns=["regime", "subject_id", "group", "statistic", "bin", "pred"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-repo", type=Path, required=True)
    ap.add_argument("--mevolve-repo", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=56)
    ap.add_argument("--models", nargs="*", help="only these (a smoke test; writes nothing)")
    a = ap.parse_args()
    import aind_dynamic_foraging_models

    lib_path = Path(aind_dynamic_foraging_models.__file__).resolve()
    assert (a.models_repo / "src").resolve() in lib_path.parents, f"library from {lib_path}"
    models_sha = _git_sha(a.models_repo)
    mevolve_sha = (a.mevolve_repo / "MEVOLVE_GIT_SHA").read_text().strip()

    from mevolve import literature
    from mevolve.data import Freeze
    from mevolve.workers import WORKER_ENV

    from aind_dynamic_foraging_models.fingerprints import BINS, N, OBS, STATISTICS, SPEC_VERSION, tally_groups

    freeze = Freeze(FREEZE)
    assert _sha256(FREEZE / "manifest.json") == FREEZE_MANIFEST, "another data freeze"
    groups = json.loads(GROUPS.read_text())
    assert groups["freeze_manifest_sha256"] == FREEZE_MANIFEST
    subjects = freeze.subjects("heldout")
    assert len(subjects) == 149
    sessions = {sid: freeze.subject_sessions(sid, "score") for sid in subjects}
    labels = {sid: [groups["groups"][s.session_id] for s in ss] for sid, ss in sessions.items()}
    assert sum(len(v) for v in labels.values()) == groups["n_sessions"]

    ranked = lit.literature_ranking()["ranked"]
    assert set(ranked) == set(literature.LITERATURE), "study 12's ranked models are MEvolve's programs"
    names = a.models or ranked
    scores = lit.literature_per_subject("heldout")
    index = {r["literature_model"]: r for r in json.loads(TWO_LEVEL_INDEX.read_text())["runs"]}

    os.environ.update(WORKER_ENV)
    OUT_DIR.mkdir(exist_ok=True)
    meta_args = dict(study_root=HERE.parent, kind="freeze",
                     code=[__file__, HERE / "freeze_p0_golden_mevolve.py"],
                     inputs=[GROUPS, lit.LITERATURE_SOURCE, lit.LEADERBOARD_SOURCE, lit.TWO_LEVEL_EM_SOURCE,
                             TWO_LEVEL_INDEX])
    base_sources = [{"type": "file", "id": str(FREEZE / "manifest.json"), "sha256": FREEZE_MANIFEST}]
    gates, sources, outputs = {}, list(base_sources), []
    for name in names:
        params = lit.literature_fitted_params(name)
        run = index[name]
        path = Path(run["path"])
        assert _sha256(path) == run["sha256"], f"{path} differs from study 12's freeze"
        d = json.loads(path.read_text())
        prog = _program(name)
        assert hashlib.sha256(prog.source.encode()).hexdigest() == d["program_sha256"], name
        recs = {str(r["subject_id"]): r for r in d["subjects"]}
        jobs = [(name, sid, sessions[sid], labels[sid], params[sid], scores[name][sid], recs[sid],
                 d["seed"], d["n_draws"]) for sid in subjects]
        model_sources = [{"type": "file", "id": str(path), "sha256": run["sha256"]}]
        with mp.get_context("spawn").Pool(min(a.workers, len(jobs))) as pool:   # fresh workers per model
            rows = pool.map(_subject_job, jobs, chunksize=1)
        regimes = {r: _gate(rows, r) for r in ("per_subject", "two_level_em")}
        gates[name] = regimes
        print(name, {r: {k: v for k, v in g.items() if k != "subjects_replay_differs"}
                     for r, g in regimes.items()}, flush=True)
        if a.models:
            continue
        path = OUT_DIR / f"{name}.parquet"
        pred_frame(rows, regimes).to_parquet(path, index=False)
        sources += model_sources
        outputs.append(path)
    if a.models:
        return

    import pandas as pd

    rec = []
    for sid in subjects:
        t = tally_groups([s.choices for s in sessions[sid]], [s.rewards for s in sessions[sid]], labels[sid])
        for g, x in t.items():
            if g == "all":   # the sum of the groups
                continue
            for stat in STATISTICS:
                rec += [(sid, g, stat, lab, float(n), float(o))
                        for lab, n, o in zip(BINS[stat], x[stat][:, N], x[stat][:, OBS])]
    path = OUT_DIR / "animal.parquet"
    pd.DataFrame(rec, columns=["subject_id", "group", "statistic", "bin", "n", "obs"]).to_parquet(path, index=False)
    outputs.append(path)

    passed = all(g["passed"] for regimes in gates.values() for g in regimes.values())
    write_artifact(OUT, {
        "_meta": build_meta("analysis/freeze_p1_replay.py", [], sources=sources, **meta_args),
        "fingerprints_library": {"repo": "AllenNeuralDynamics/aind-dynamic-foraging-models",
                                 "git_sha": models_sha},
        "mevolve_git_sha": mevolve_sha,
        "spec": SPEC_VERSION,
        "tolerance": {"log_lik": TOL_LOG_LIK, "fingerprint": TOL_FINGERPRINT},
        "passed": passed,
        "n_subjects": len(subjects),
        "models": gates,
    }, outputs=outputs)
    print(f"{'PASS' if passed else 'FAIL'} -> {OUT}")


if __name__ == "__main__":
    main()
