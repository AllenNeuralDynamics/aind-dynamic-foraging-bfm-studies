#!/usr/bin/env python
"""Fit/score role of every session (PLAN.md, "Two passes per held-out mouse"), frozen for scoring.

Reads the MEvolve data freeze on /allen (snapshot 20260603), checks that it is the file
``reference/folds.json`` was built on, and writes ``reference/roles.csv`` (subject_id, session_id,
role), the table the wrapper's ``post_training_analysis.heldout_rescore --roles`` reads: pass A tunes
on ``fit`` and scores ``score``; pass B swaps them. Its digest goes in ``reference/roles.json``.
Needs pandas; run on a compute node:

    make -C studies/17-gru-crossfit-predictability freeze-roles PY=<mevolve-search env>/bin/python
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, file_digest, write_artifact  # noqa: E402

FOLDS = STUDY / "reference" / "folds.json"
SESSIONS = Path("/allen/aind/scratch/han.hou/mevolve/freeze/20260603/sessions.parquet")
OUT_CSV = STUDY / "reference" / "roles.csv"
OUT = STUDY / "reference" / "roles.json"


def main() -> None:
    folds = json.loads(FOLDS.read_text())
    recorded = {s["id"]: s["sha256"] for s in folds["_meta"]["sources"]}
    assert file_digest(SESSIONS) == recorded[str(SESSIONS)], "freeze differs from the one folds.json used"
    s = pd.read_parquet(SESSIONS, columns=["subject_id", "session_id", "role"])
    s["subject_id"] = s["subject_id"].astype(str)
    assert set(s["role"]) == {"fit", "score"}
    assert set(s["subject_id"]) == set(folds["fold_of"])
    s.sort_values(["subject_id", "session_id"]).to_csv(OUT_CSV, index=False)

    meta = build_meta("analysis/freeze_roles.py", [], study_root=STUDY, kind="freeze", code=[__file__],
                      inputs=[FOLDS], sources=[{"type": "file", "id": str(SESSIONS), "sha256": file_digest(SESSIONS)}])
    write_artifact(OUT, {
        "_meta": meta,
        "snapshot": "20260603",
        "n_sessions": int(len(s)),
        "n_subjects": int(s["subject_id"].nunique()),
        "roles": s["role"].value_counts().to_dict(),
    }, outputs=[OUT_CSV])


if __name__ == "__main__":
    main()
