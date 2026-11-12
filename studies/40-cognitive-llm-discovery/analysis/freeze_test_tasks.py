#!/usr/bin/env python
"""Session groups and gym task specs for the test cohorts (ADR 0008's test scores).

``freeze_selection_tasks.py``'s rules (study 15's, unchanged), for ``test`` (the 149 held-out subjects;
equal to study 15's committed groups, specs and exclusions) and ``gru_test`` (the GRU's own 149), whose
subjects come from ``selection_cohorts``. Writes ``selection/tasks_<cohort>.json``.

Needs /allen (database snapshot) and duckdb, so the dynamic-foraging-bfm-cpu env; on a compute node:

    python analysis/freeze_test_tasks.py --cohort test
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

import selection_cohorts
from freeze_selection_tasks import (COHORTS, FREEZE, FREEZE_MANIFEST, MAX_BLOCK_TALLY, OUT_DIR, PERSEVERATIVE_LIMIT,
                                    S15, SNAPSHOT, STUDY, build_meta, groups_and_specs, write_artifact)

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["test", "gru_test"])
    a = ap.parse_args()
    manifest = FREEZE / "manifest.json"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == FREEZE_MANIFEST
    sessions = pd.read_parquet(FREEZE / "sessions.parquet")
    subjects = set(selection_cohorts.subjects(a.cohort, sessions))
    mine = sessions[sessions["subject_id"].isin(subjects)].copy()
    assert mine["subject_id"].nunique() == len(subjects)
    groups, specs, excluded, db_path = groups_and_specs(mine)
    out = OUT_DIR / f"tasks_{a.cohort}.json"
    meta = build_meta(f"analysis/{Path(__file__).name}", [], study_root=STUDY, kind="freeze",
                      code=[__file__, HERE / "freeze_selection_tasks.py", HERE / "selection_cohorts.py",
                            S15 / "freeze_p0_task_checks.py", S15 / "freeze_p1_session_groups.py",
                            S15 / "freeze_p2_tasks.py", S15 / "p2_task.py"],
                      inputs=[COHORTS] + ([selection_cohorts.GRU_ROWS] if a.cohort == "gru_test" else []),
                      sources=[{"type": "file", "id": str(manifest), "sha256": FREEZE_MANIFEST},
                               {"type": "file", "id": db_path, "state": f"database snapshot {SNAPSHOT}"}])
    write_artifact(out, {
        "_meta": meta, "cohort": a.cohort, "n_subjects": len(subjects),
        "freeze_manifest_sha256": FREEZE_MANIFEST,
        "rules": {"perseverative_limit": PERSEVERATIVE_LIMIT, "max_block_tally": MAX_BLOCK_TALLY,
                  "source": "study 15 freeze_p1_session_groups and freeze_p2_tasks, applied unchanged"},
        "n_score_sessions": len(groups), "n_specs": len(specs), "n_excluded": len(excluded),
        "n_subjects_with_specs": len({s["subject_id"] for s in specs}), "by_kind": dict(Counter(s["kind"] for s in specs)),
        "groups": dict(sorted(groups.items())), "excluded": excluded, "sessions": specs,
    })
    print({k: v for k, v in json.loads(out.read_text()).items() if k not in ("groups", "excluded", "sessions", "_meta")})


if __name__ == "__main__":
    main()
