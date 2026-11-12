"""The subjects of each cohort the selection and test scripts score (ADR 0003, ADR 0008).

- ``generation``, ``selection``: ``reference/cohorts_seed0.json``;
- ``test``: the freeze's 149 held-out subjects (those of the per-subject MLE and HB baselines);
- ``gru_test``: the GRU's own 149 test subjects, read from study 01's committed per-subject rows (v2,
  full training set): the 146 shared with ``test`` plus the three ADR 0003 kept out of every cohort.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
COHORTS = STUDY / "reference" / "cohorts_seed0.json"
GRU_ROWS = STUDY.parent / "01-gru-scaling-law" / "analysis" / "per_subject_rows.json"
NAMES = ("generation", "selection", "test", "gru_test")


def subjects(name: str, sessions: pd.DataFrame) -> list[str]:
    """Sorted subject ids of a cohort; ``sessions`` is the freeze's ``sessions.parquet``."""
    if name in ("generation", "selection"):
        return json.loads(COHORTS.read_text())["cohorts"][name]
    held = sorted(sessions.loc[sessions["pool"] == "heldout", "subject_id"].unique())
    if name == "test":
        return held
    rows = json.loads(GRU_ROWS.read_text())["rows"]
    gru = sorted({r["subject"] for r in rows if r["variant"] == "v2" and r["ratio"] == 1.0})
    extra = set(gru) - set(held)
    assert len(gru) == 149 and extra == set(json.loads(COHORTS.read_text())["excluded"]), "GRU list changed"
    return gru
