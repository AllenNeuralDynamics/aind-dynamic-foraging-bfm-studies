#!/usr/bin/env python
"""Per-session held-out scores of 17 programs on the 100 monitor subjects (study 40 r5, issue #146).

The plan, the program set and the statistics were fixed on #146 before any scoring. Two steps:

- ``prepare`` copies the 17 programs into ``individual/programs/`` (each checked against its recorded
  sha256) and writes ``individual/sessions.json``: per monitor subject and score session, its order,
  stage, task, trials, responses and the ignore rate in each half of the session (the covariates of
  #146). Reads the freeze and the monitor archive on /allen.
- ``score --tag <tag>`` fits one program per subject with MEvolve's 2-level Laplace-EM (``max_em`` 100,
  seed 0, 2,048 draws; settings of ADR 0008, no rollouts) and writes ``individual/scores/<tag>.json``:
  per subject, every score session's 2-level and per-subject held-out log-likelihood
  (``session_per_subject_log_lik``, MEvolve #85), its trials and its group, and whether the subject falls
  back to its per-subject fit (MEvolve ADR 0004).

Needs /allen and MEvolve ``MEVOLVE_SHA`` on PYTHONPATH; on a compute node, one job per program (24 CPUs,
72 GB: 12 fitting workers):

    sbatch -p aind --ntasks=1 --cpus-per-task=24 --mem-per-cpu=3000M --time=4:00:00 --wrap \\
      "env PYTHONPATH=<mevolve>/src <mevolve-search env>/bin/python analysis/freeze_individual.py score --tag <tag>"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

FREEZE = Path("/allen/aind/scratch/han.hou/mevolve/freeze/20260603")
ARCHIVE = Path("/allen/aind/scratch/han.hou/mevolve/monitor/programs")
COHORTS = STUDY / "reference" / "cohorts_seed0_monitor.json"
MONITOR = HERE / "seeded_islands_2" / "monitor.json"
OUT = HERE / "individual"
SEED, N_DRAWS, MAX_EM = 0, 2048, 100
MEVOLVE_URL = "https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-mevolve"
MEVOLVE_SHA = "9d16704f4da03271774d138523bf0d6503a5f800"   # MEvolve #85 (PR #86): session_per_subject_log_lik

# tag -> source: "archive" (the monitor archive, sha256 from monitor.json), an in-study path, or
# "literature:<file>" (MEvolve's literature programs at MEVOLVE_SHA)
PROGRAMS = {
    "arch03": "selection/programs/rank3_e3819885.py",
    "arch01": "archive", "new01": "archive", "arch04": "archive", "arch14": "archive", "arch38": "archive",
    "arch15": "archive", "arch39": "archive", "arch33": "archive", "arch28": "archive", "arch29": "archive",
    "arch08": "archive",
    "s75": "simplify/simplify-75/programs/c1_141c8a13.py",
    "s50": "simplify/simplify-50/programs/c1_cd994c73.py",
    "miller_rhg": "literature:miller_rhg.py",
    "lebedeva_pr": "literature:lebedeva_pr.py",
    "lopez_double_trace": "literature:lopez_double_trace.py",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def program_file(tag: str) -> Path:
    found = sorted((OUT / "programs").glob(f"{tag}_*.py"))
    assert len(found) == 1, (tag, found)
    return found[0]


def monitor_subjects() -> list[str]:
    return json.loads(COHORTS.read_text())["cohorts"]["monitor"]


def prepare() -> None:
    import pandas as pd
    from mevolve import program

    monitor = {p["tag"]: p for p in json.loads(MONITOR.read_text())["programs"]}
    (OUT / "programs").mkdir(parents=True, exist_ok=True)
    manifest, sources, inputs = {}, [], [MONITOR, COHORTS]
    for tag, src in PROGRAMS.items():
        if src == "archive":
            path, want = ARCHIVE / f"{tag}.py", monitor[tag]["code_sha256"]
            sources.append({"type": "file", "id": str(path), "sha256": want})
        elif src.startswith("literature:"):
            path = Path(program.LITERATURE_DIR) / src.split(":", 1)[1]
            want = sha256(path)
            sources.append({"type": "url", "id": f"{MEVOLVE_URL}/blob/{MEVOLVE_SHA}/"
                            f"src/mevolve/programs/literature/{path.name}", "sha256": want})
        else:
            path = HERE / src
            want = sha256(path)
            inputs.append(path)
        got = sha256(path)
        assert got == want, (tag, got, want)
        dest = OUT / "programs" / f"{tag}_{got[:8]}.py"
        shutil.copyfile(path, dest)
        manifest[tag] = {"file": dest.name, "sha256": got, "source": src if src != "archive" else str(path),
                         "name": monitor[tag]["name"] if tag in monitor else tag}
    if "arch03" in monitor:          # the search's best is the monitor archive's arch03
        assert manifest["arch03"]["sha256"] == monitor["arch03"]["code_sha256"]

    subjects = monitor_subjects()
    sessions = pd.read_parquet(FREEZE / "sessions.parquet")
    sessions = sessions[sessions["subject_id"].isin(subjects) & (sessions["role"] == "score")]
    trials = pd.read_parquet(FREEZE / "trials.parquet", columns=["session_id", "trial", "response"])
    trials = trials[trials["session_id"].isin(set(sessions["session_id"]))]
    ignored = {}
    for sid, t in trials.sort_values(["session_id", "trial"]).groupby("session_id"):
        r = t["response"].to_numpy()
        h = len(r) // 2
        ignored[sid] = [float((r[:h] == 2).mean()), float((r[h:] == 2).mean())]
    per: dict = {s: [] for s in subjects}
    for row in sessions.sort_values(["subject_id", "index"]).itertuples(index=False):
        per[row.subject_id].append({"session_id": row.session_id, "index": int(row.index), "task": row.task,
                                    "stage": None if row.stage is None else str(row.stage),
                                    "n_trials": int(row.n_trials), "n_choice": int(row.n_choice),
                                    "ignore_halves": ignored[row.session_id]})
    assert all(len(v) >= 4 for v in per.values()), "a subject has fewer than 4 score sessions"
    files = [OUT / "programs" / m["file"] for m in manifest.values()]
    freeze_manifest = FREEZE / "manifest.json"
    meta = build_meta("analysis/freeze_individual.py", [], study_root=STUDY, kind="freeze", code=[__file__],
                      inputs=inputs, sources=sources + [{"type": "file", "id": str(freeze_manifest),
                                                         "sha256": sha256(freeze_manifest)}])
    write_artifact(OUT / "sessions.json", {"_meta": meta, "programs": manifest, "subjects": per},
                   outputs=files)


def score(tag: str, workers: int, limit: int | None) -> None:
    from mevolve import evaluate
    from mevolve.data import Freeze

    path = program_file(tag)
    plan = json.loads((OUT / "sessions.json").read_text())
    assert plan["programs"][tag]["sha256"] == sha256(path)
    subjects = monitor_subjects()[:limit]
    freeze = Freeze(FREEZE)
    data = [evaluate.subject_data(freeze, s) for s in subjects]
    for d in data:          # the score sessions are the ones prepare recorded, in the same order
        assert [s.session_id for s in d.score] == [s["session_id"] for s in plan["subjects"][d.subject_id]]
    t0 = time.time()
    records = evaluate.evaluate(str(path.resolve()), data, regime="2level", workers=workers, max_em=MAX_EM,
                                n_draws=N_DRAWS, seed=SEED, isolate=True)
    summary = evaluate.program_score(records)
    fell = set(summary.pop("fallback_subjects"))
    keep = ("session_ids", "session_groups", "session_n_trials", "session_log_lik", "session_per_subject_log_lik",
            "log_lik", "per_subject_log_lik", "n_trials", "em_iterations", "converged")
    rows = [{"subject_id": r["subject_id"], "fell_back": r["subject_id"] in fell, **{k: r[k] for k in keep}}
            for r in records]
    out = OUT / "scores" / (f"{tag}.json" if limit is None else f"{tag}_limit{limit}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = build_meta("analysis/freeze_individual.py", [], study_root=STUDY, kind="freeze", code=[__file__],
                      inputs=[OUT / "sessions.json", path],
                      sources=[{"type": "url", "id": f"{MEVOLVE_URL}/tree/{MEVOLVE_SHA}"}])
    write_artifact(out, {"_meta": meta, "tag": tag, "program": path.name,
                         "settings": {"regime": "2level", "max_em": MAX_EM, "n_draws": N_DRAWS, "seed": SEED,
                                      "fallback_tol": evaluate.FALLBACK_TOL, "mevolve": MEVOLVE_SHA},
                         "score": summary, "seconds": time.time() - t0, "subjects": rows}, outputs=[])


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="step", required=True)
    sub.add_parser("prepare")
    s = sub.add_parser("score")
    s.add_argument("--tag", required=True, choices=list(PROGRAMS))
    s.add_argument("--workers", type=int, default=12)
    s.add_argument("--limit", type=int, help="first N subjects only (a smoke test, never frozen)")
    a = ap.parse_args()
    assert MEVOLVE_SHA != "TODO", "pin MEvolve first"
    if a.step == "prepare":
        prepare()
    else:
        score(a.tag, a.workers, a.limit)


if __name__ == "__main__":
    main()
