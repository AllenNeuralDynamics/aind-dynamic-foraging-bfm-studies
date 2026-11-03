#!/usr/bin/env python
"""P0 golden check 1: the fingerprint library reproduces MEvolve's replay fingerprints.

MEvolve's gate-1b per-subject run fit all 19 literature programs to the 149 test subjects and
stored, per subject, the fitted parameters and the replay fingerprints of the score sessions. For
every program and subject this replays the program at the stored parameters with MEvolve's own
scoring code, then computes the replay fingerprints two ways: with MEvolve's ``fingerprints()``
and with the library's ``mevolve_replay_gaps()`` on fingerprints-v1 tallies (ADR 0002).

Pass: library and MEvolve agree to 1e-12 on every subject, group and statistic (same inputs, so
only summation order may differ), and the library reproduces the stored fingerprints to 1e-9 on
every subject whose replay reproduces MEvolve's stored log likelihood (to 1e-6). A subject whose
replay does not (its stored parameters, clipped by ``from_params`` at a bound, are not the fit's
own vector) has other inputs, so it is counted, not compared.

Needs /allen and the mevolve env; on a compute node:

    srun --ntasks=1 --cpus-per-task=20 --mem=64G -p aind --time=01:00:00 \\
      env PYTHONPATH=<foraging-models>/src:<mevolve>/src <mevolve env>/bin/python \\
      analysis/freeze_p0_golden_mevolve.py --models-repo <foraging-models> --mevolve-repo <mevolve>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

GATE1B = Path("/allen/aind/scratch/han.hou/mevolve/gate1b/final_888c4ea")
FREEZE = Path("/allen/aind/scratch/han.hou/mevolve/freeze/20260603")
FREEZE_MANIFEST = "cfea3d1e21f49ed7baa42f11e1fdda66fd5901c519a47773c08716e062e7eef0"
TOL_LIBRARY, TOL_STORED, TOL_LOG_LIK = 1e-12, 1e-9, 1e-6
OUT = HERE / "p0_golden_mevolve.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha(repo: Path) -> str:
    """HEAD of a clean checkout; refuses a dirty one, whose code no SHA describes."""
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", "src"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"{repo} has uncommitted changes under src/: {dirty}")
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def _max_diff(a: dict, b: dict) -> tuple[float, int]:
    """Largest |a - b| over shared (group, statistic) entries, and the count of unshared ones."""
    keys_a = {(g, s) for g, d in a.items() for s in d}
    keys_b = {(g, s) for g, d in b.items() for s in d}
    shared = keys_a & keys_b
    diff = max((abs(a[g][s] - b[g][s]) for g, s in shared), default=0.0)
    return float(diff), len(keys_a ^ keys_b)


def _one_model(args) -> dict:
    path, sessions_by_subject = args
    from mevolve import fitting, program, scoring
    from mevolve.fingerprints import fingerprints

    from aind_dynamic_foraging_models.fingerprints import mevolve_replay_gaps, tally_groups

    d = json.loads(Path(path).read_text())
    module = Path(path).stem.split("_", 2)[2]
    prog = program.literature(module)
    assert hashlib.sha256(prog.source.encode()).hexdigest() == d["program_sha256"], module
    worst = {"library_vs_mevolve": 0.0, "mevolve_vs_stored": 0.0, "library_vs_stored": 0.0}
    unshared = {k: 0 for k in worst}
    n_values, n_other_inputs = 0, 0
    for rec in d["subjects"]:
        sessions = sessions_by_subject[str(rec["subject_id"])]
        assert [s.session_id for s in sessions] == rec["session_ids"], rec["subject_id"]
        x = fitting.from_params(prog, rec["params"])
        batch = fitting.pad(sessions)
        log_lik, _ = scoring.point_scores(prog, x, batch)
        same_inputs = abs(float(log_lik[: len(sessions)].sum()) - rec["log_lik"]) <= TOL_LOG_LIK
        n_other_inputs += int(not same_inputs)
        p = scoring.point_p_right(prog, x, batch)
        p_right = [np.asarray(p[i, : len(s.choices)]) for i, s in enumerate(sessions)]
        ref = fingerprints(sessions, p_right)
        lib = mevolve_replay_gaps(tally_groups(
            [s.choices for s in sessions], [s.rewards for s in sessions],
            [f"{s.stage_group}|{s.task}" for s in sessions], p_right=p_right))
        stored = rec["fingerprints"]
        pairs = {"library_vs_mevolve": (lib, ref)}
        if same_inputs:
            pairs.update({"mevolve_vs_stored": (ref, stored), "library_vs_stored": (lib, stored)})
        for name, (a, b) in pairs.items():
            diff, n_unshared = _max_diff(a, b)
            worst[name] = max(worst[name], diff)
            unshared[name] += n_unshared
        n_values += sum(len(v) for v in stored.values())
    return {"literature_model": d["model"], "program": module, "n_subjects": len(d["subjects"]),
            "n_values": n_values, "n_subjects_replay_differs": n_other_inputs, "max_abs_diff": worst, "n_unshared_keys": unshared,
            "source": {"type": "file", "id": str(path), "sha256": _sha256(Path(path))}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-repo", type=Path, required=True)
    ap.add_argument("--mevolve-repo", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=19)
    a = ap.parse_args()
    models_sha = _git_sha(a.models_repo)
    mevolve_sha = (a.mevolve_repo / "MEVOLVE_GIT_SHA").read_text().strip()

    from mevolve.data import Freeze
    from mevolve.workers import WORKER_ENV

    freeze = Freeze(FREEZE)
    assert _sha256(FREEZE / "manifest.json") == FREEZE_MANIFEST, "another data freeze"
    sessions_by_subject = {sid: freeze.subject_sessions(sid, "score")
                           for sid in freeze.subjects("heldout")}
    files = sorted(GATE1B.glob("subject_*.json"))
    assert len(files) == 19, f"{len(files)} model files in {GATE1B}"
    os.environ.update(WORKER_ENV)
    with mp.get_context("spawn").Pool(min(a.workers, len(files))) as pool:
        rows = pool.map(_one_model, [(str(f), sessions_by_subject) for f in files], chunksize=1)

    for r in rows:
        assert r["n_subjects"] == 149, r["literature_model"]
    passed = all(r["max_abs_diff"]["library_vs_mevolve"] <= TOL_LIBRARY
                 and r["max_abs_diff"]["library_vs_stored"] <= TOL_STORED
                 and not any(r["n_unshared_keys"].values()) for r in rows)
    sources = [r.pop("source") for r in rows]
    sources.append({"type": "file", "id": str(FREEZE / "manifest.json"), "sha256": FREEZE_MANIFEST})
    meta = build_meta("analysis/freeze_p0_golden_mevolve.py", [], study_root=HERE.parent,
                      kind="freeze", code=[__file__], sources=sources)
    write_artifact(OUT, {
        "_meta": meta,
        "fingerprints_library": {"repo": "AllenNeuralDynamics/aind-dynamic-foraging-models",
                                 "git_sha": models_sha},
        "mevolve_git_sha": mevolve_sha,
        "tolerance": {"library_vs_mevolve": TOL_LIBRARY, "library_vs_stored": TOL_STORED,
                      "replay_log_lik": TOL_LOG_LIK},
        "passed": passed,
        "models": sorted(rows, key=lambda r: r["literature_model"]),
    })
    print(f"{'PASS' if passed else 'FAIL'}: {len(rows)} models -> {OUT}")
    for r in rows:
        print(f"  {r['literature_model']:40s} {r['max_abs_diff']}")


if __name__ == "__main__":
    main()
