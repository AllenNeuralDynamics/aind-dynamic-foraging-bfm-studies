#!/usr/bin/env python
"""Freeze: the stratum of each of the 614 pool mice, as freeze_folds.py dealt the pool folds.

Rebuilds the strata freeze_folds.py used (study 40 tier x curriculum x session-count third) from the
same data freeze and study 40 cohorts file, checks that each stratum count per fold equals
reference/folds.json's `balance`, and writes reference/pool_strata.json (subject -> tier, curriculum,
third). cv_fold_spread.py uses it to re-deal the pool mice within strata (a stratified null).
Needs pandas + pyarrow: the mevolve-search env, on a compute node.

    python analysis/freeze_pool_strata.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, file_digest, write_artifact  # noqa: E402

FREEZE = Path("/allen/aind/scratch/han.hou/mevolve/freeze/20260603")
COHORTS = STUDY.parent / "40-cognitive-llm-discovery" / "reference" / "cohorts_seed0_monitor.json"
FOLDS = STUDY / "reference" / "folds.json"
OUT = STUDY / "reference" / "pool_strata.json"


def main() -> None:
    cohorts = json.loads(COHORTS.read_text())
    manifest = FREEZE / "manifest.json"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == cohorts["freeze_manifest_sha256"]
    folds = json.loads(FOLDS.read_text())
    sessions = pd.read_parquet(FREEZE / "sessions.parquet", columns=["subject_id", "pool", "curriculum"])
    sessions["subject_id"] = sessions["subject_id"].astype(str)
    subjects = sessions.groupby("subject_id").agg(pool=("pool", "first"), curriculum=("curriculum", "first"),
                                                  n_sessions=("pool", "size"))
    pool = subjects[subjects["pool"] == "train"].copy()
    discovery = set().union(*map(set, cohorts["cohorts"].values()))
    excluded = set(cohorts["excluded"])
    pool["tier"] = ["discovery" if s in discovery else "excluded" if s in excluded else "confirmation" for s in pool.index]
    pool["third"] = pd.qcut(pool["n_sessions"].rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str)
    strata = {s: {"tier": r.tier, "curriculum": r.curriculum, "third": r.third} for s, r in pool.iterrows()}
    for k in ("1", "2", "3", "4"):   # the rebuilt strata reproduce the frozen per-fold balance
        ids = folds["folds"][f"pool_{k}"] if f"pool_{k}" in folds["folds"] else folds["folds"][k]
        for col in ("tier", "curriculum", "third"):
            assert dict(Counter(strata[s][col] for s in ids)) == folds["balance"][k][col], (k, col)
    meta = build_meta("analysis/freeze_pool_strata.py", [], study_root=STUDY, kind="freeze", code=[__file__],
                      inputs=[COHORTS, FOLDS], sources=[{"type": "file", "id": str(manifest), "sha256": file_digest(manifest)}])
    write_artifact(OUT, {"_meta": meta, "rule": folds["strata"], "n_subjects": len(strata), "strata": strata})


if __name__ == "__main__":
    main()
