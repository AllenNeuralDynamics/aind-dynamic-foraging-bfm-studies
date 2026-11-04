#!/usr/bin/env python
"""P1 input: the group of every score session of the 149 test subjects (ADR 0004, 0007).

A session's group is ``warm-up`` (on every trial one side's reward probability is 0, or the
probabilities are only 0 and 1; P0's rule) or else ``{stage_group}|{task}``, MEvolve's label
(``final`` = STAGE_FINAL or GRADUATED, ``earlier``, ``unknown``). The warm-up flag needs the
trials' reward probabilities, which only the database has; every later P1 and P2 step reads the
groups from this file, so replay, rollouts and the animal tallies use the same labels. The
warm-up count must equal P0's (r1).

Needs /allen (the database snapshot); on a compute node:

    srun --ntasks=1 --cpus-per-task=8 --mem=32G -p aind --time=00:30:00 \\
      <dynamic-foraging-bfm-cpu env>/bin/python analysis/freeze_p1_session_groups.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
sys.path.insert(0, str(HERE))
from _meta import build_meta, write_artifact  # noqa: E402
from freeze_p0_task_checks import FREEZE, FREEZE_MANIFEST, SNAPSHOT, load_trials  # noqa: E402

P0 = HERE / "p0_task_checks.json"
OUT = HERE / "p1_session_groups.json"
FINAL_STAGES = ("STAGE_FINAL", "GRADUATED")   # mevolve.data.Session.stage_group


def stage_group(stage) -> str:
    if stage is None or pd.isna(stage):
        return "unknown"
    return "final" if stage in FINAL_STAGES else "earlier"


def main() -> None:
    manifest = FREEZE / "manifest.json"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == FREEZE_MANIFEST
    expected = json.loads(manifest.read_text())["files"]["sessions.parquet"]
    actual = hashlib.sha256((FREEZE / "sessions.parquet").read_bytes()).hexdigest()
    assert actual == expected, f"sessions.parquet {actual} differs from the manifest's {expected}"
    sessions = pd.read_parquet(FREEZE / "sessions.parquet")
    sessions = sessions[(sessions["pool"] == "heldout") & (sessions["role"] == "score")]
    assert sessions["subject_id"].nunique() == 149
    trials, db_path = load_trials(sessions)
    by_session = dict(tuple(trials.groupby("session_id", sort=False)))

    groups = {}
    for row in sessions.itertuples(index=False):
        p = by_session[row.session_id][["reward_probabilityL", "reward_probabilityR"]].to_numpy(float)
        warm_up = bool(np.isin(p, (0.0, 1.0)).all()) or bool((p.min(axis=1) == 0).all())
        groups[row.session_id] = "warm-up" if warm_up else f"{stage_group(row.stage)}|{row.task}"
    assert "all" not in groups.values()   # tally_groups reserves "all" for the total
    counts = Counter(groups.values())
    p0_warm_ups = json.loads(P0.read_text())["warm_up_score_sessions"]["total"]
    assert counts["warm-up"] == p0_warm_ups, f"{counts['warm-up']} warm-ups, P0 counted {p0_warm_ups}"

    sources = [{"type": "file", "id": str(manifest), "sha256": FREEZE_MANIFEST},
               {"type": "file", "id": db_path, "state": f"database snapshot {SNAPSHOT}"}]
    meta = build_meta("analysis/freeze_p1_session_groups.py", [], study_root=HERE.parent,
                      kind="freeze", code=[__file__, HERE / "freeze_p0_task_checks.py"],
                      inputs=[P0], sources=sources)
    write_artifact(OUT, {
        "_meta": meta,
        "freeze_manifest_sha256": FREEZE_MANIFEST,
        "n_sessions": len(groups),
        "counts": dict(counts.most_common()),
        "groups": dict(sorted(groups.items())),
    })
    print(dict(counts.most_common()), "->", OUT)


if __name__ == "__main__":
    main()
