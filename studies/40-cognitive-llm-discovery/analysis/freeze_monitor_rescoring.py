#!/usr/bin/env python
"""Freeze the monitor rescoring of seeded-islands-2's programs (issue #128) into one committed file.

Two batches were scored on the 48 validation and the 100 monitor subjects (ADR 0003 amendment), each plan
written on #128 before scoring: the best seed and the top 10 at 15:23 PT 2026-11-07 plus the new best
(``programs.json``), and the 40 programs of the final archive (``programs_final_archive.json``). Evaluator:
MEvolve 761cefd (the search's), 2-level, seed 0, ``max_em`` 20, 2,048 draws. This keeps, per program, its
identity and, per cohort and subject, the 2-level and per-subject held-out log-likelihoods and trial
count, so the offline producer can apply the per-mouse fallback (MEvolve ADR 0004) and resample subjects.

Needs /allen (the scoring outputs); reads files only:

    python analysis/freeze_monitor_rescoring.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

SRC = Path("/allen/aind/scratch/han.hou/mevolve/monitor")
COHORTS = STUDY / "reference" / "cohorts_seed0_monitor.json"
OUT = HERE / "seeded_islands_2" / "monitor.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    cohorts = json.loads(COHORTS.read_text())["cohorts"]
    first = json.loads((SRC / "programs.json").read_text())
    final = json.loads((SRC / "programs_final_archive.json").read_text())
    sources = [{"type": "file", "id": str(SRC / n), "sha256": _sha256(SRC / n)}
               for n in ("programs.json", "programs_final_archive.json", "score.py")]
    progs = {}
    for batch, plist in (("first", first), ("final_archive", final)):
        for p in plist:
            entry = progs.setdefault(p["tag"], {k: p.get(k) for k in ("tag", "id", "island", "generation", "name",
                                                                      "code_sha256")}
                                     | {"search_validation_x100": p["validation_score"], "batches": []})
            entry["batches"].append(batch)
            if batch == "final_archive":
                entry["archive_rank"] = p["rank"]
    for tag, entry in progs.items():
        f = SRC / "results" / f"{tag}.json"
        sources.append({"type": "file", "id": str(f), "sha256": _sha256(f)})
        r = json.loads(f.read_text())
        for cohort in ("validation", "monitor"):
            rows = {s["subject_id"]: s for s in r["subjects"] if s["cohort"] == cohort}
            assert set(rows) == set(cohorts[cohort]), f"{tag} {cohort}: other subjects"
            entry[cohort] = [[rows[s]["log_lik"], rows[s]["per_subject_log_lik"], rows[s]["n_trials"]]
                             for s in sorted(rows)]
    write_artifact(OUT, {
        "_meta": build_meta(f"analysis/{Path(__file__).name}", [], study_root=STUDY, kind="freeze",
                            code=[__file__], inputs=[COHORTS], sources=sources),
        "about": "seeded-islands-2 programs rescored on the validation and monitor subjects (#128). Per cohort "
                 "and subject, sorted by subject id: [2-level log-lik, per-subject log-lik, trials].",
        "evaluator": {"mevolve": "761cefd", "regime": "2level", "seed": 0, "max_em": 20, "n_draws": 2048},
        "subjects": {c: sorted(cohorts[c]) for c in ("validation", "monitor")},
        "programs": sorted(progs.values(), key=lambda e: e["tag"]),
    })
    print(f"wrote {len(progs)} programs to {OUT}")


if __name__ == "__main__":
    main()
