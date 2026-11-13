#!/usr/bin/env python
"""Mouse tiers of study 41 (PLAN.md, "Mouse tiers"; issue #149), frozen before any look.

- discovery: the 296 subjects study 40 used (generation, validation, selection, monitor), read from
  study 40's ``reference/cohorts_seed0_monitor.json``;
- confirmation: every other training-pool subject of the freeze, minus that file's 3 ``excluded``
  subjects (training-pool mice that are the GRU's extra test subjects);
- test: the freeze's 149 held-out subjects.

Writes ``reference/tiers.json``. Reads the MEvolve data freeze on /allen (snapshot 20260603) and checks
its manifest against the cohort file's ``freeze_manifest_sha256``. Needs pandas; run on a compute node:

    make -C studies/41-cognitive-strategy-profiles freeze-tiers PY=<mevolve-search env>/bin/python
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

FREEZE = Path("/allen/aind/scratch/han.hou/mevolve/freeze/20260603")
COHORTS = STUDY.parent / "40-cognitive-llm-discovery" / "reference" / "cohorts_seed0_monitor.json"
OUT = STUDY / "reference" / "tiers.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    cohorts = json.loads(COHORTS.read_text())
    manifest = FREEZE / "manifest.json"
    assert sha256(manifest) == cohorts["freeze_manifest_sha256"], "freeze differs from study 40's"
    sessions = pd.read_parquet(FREEZE / "sessions.parquet", columns=["subject_id", "pool"])
    pools = sessions.astype({"subject_id": str}).groupby("pool").subject_id.unique()
    train, heldout = set(pools["train"]), set(pools["heldout"])

    discovery = {name: sorted(ids) for name, ids in cohorts["cohorts"].items()}
    used = set().union(*discovery.values())
    excluded = set(cohorts["excluded"])
    assert used <= train and excluded <= train and not used & excluded
    confirmation = sorted(train - used - excluded)
    assert (len(train), len(heldout), len(used), len(confirmation)) == (614, 149, 296, 315)

    meta = build_meta("analysis/freeze_tiers.py", [], study_root=STUDY, kind="freeze", code=[__file__],
                      inputs=[COHORTS],
                      sources=[{"type": "file", "id": str(manifest), "sha256": sha256(manifest)},
                               {"type": "file", "id": str(FREEZE / "sessions.parquet"),
                                "sha256": sha256(FREEZE / "sessions.parquet")}])
    write_artifact(OUT, {
        "_meta": meta,
        "snapshot": "20260603",
        "freeze_manifest_sha256": cohorts["freeze_manifest_sha256"],
        "sizes": {"discovery": len(used), "confirmation": len(confirmation), "test": len(heldout),
                  "excluded": len(excluded)},
        "discovery": discovery,
        "confirmation": confirmation,
        "test": sorted(heldout),
        "excluded": sorted(excluded),
    })


if __name__ == "__main__":
    main()
