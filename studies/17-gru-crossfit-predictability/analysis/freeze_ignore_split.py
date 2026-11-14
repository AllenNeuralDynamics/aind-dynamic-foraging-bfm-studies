#!/usr/bin/env python
"""Freeze: 3-way vs 2-way GRU held-out L/R log-likelihood, split by trials since the last ignore.

For each fold's full-size 3-way and 2-way models (seed 0), both passes, scored sessions only, joins
the two models' per-trial predictions on (session_id, trial) over engaged trials (both models
predict them) and sums the L/R log-likelihood of each model per bin of k = trials since the last
ignored trial in the session (k = 1: the first trial after an ignore). Sessions are in trial order;
a trial before any ignore in its session is bin "none". Reads the `scoring` variant's outputs on
/allen scratch; each job's manifest.json sha256 is recorded as a source, as in freeze_scoring.py.

    python analysis/freeze_ignore_split.py   # writes analysis/ignore_split.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, file_digest, write_artifact  # noqa: E402

OUT_ROOT = Path("/allen/aind/scratch/han.hou/s17_scoring/out")
OUT = HERE / "ignore_split.json"
PAIRS = {"test": ("test-3way-full-s0", "test-2way-full-s0"),
         **{f"pool_{k}": (f"pool{k}-3way-full", f"pool{k}-2way-full") for k in range(1, 5)}}
BINS = [("1", 1, 1), ("2-4", 2, 4), ("5-19", 5, 19), ("20+", 20, 10**9)]


def k_since_ignore(df: pd.DataFrame) -> pd.Series:
    """Trials since the last ignored trial in the same session (NaN before the first ignore)."""
    pos = df.groupby("session_id").cumcount()
    last = pos.where(df["response"] == 2).groupby(df["session_id"]).ffill()
    return pos - last


def main() -> None:
    sources, folds = [], {}
    for fold, (m3, m2) in PAIRS.items():
        acc = {b: [0, 0.0, 0.0] for b, _, _ in BINS} | {"none": [0, 0.0, 0.0]}
        for p in "AB":
            d3 = OUT_ROOT / m3 / f"pass{p}_lam-checkpoint"
            d2 = OUT_ROOT / m2 / f"pass{p}_lam-checkpoint"
            for d in (d3, d2):
                sources.append({"type": "file", "id": str(d / "manifest.json"), "sha256": file_digest(d / "manifest.json")})
            a = pd.read_csv(d3 / "trial_predictions.csv.gz", usecols=["session_id", "trial", "response", "use", "ll_lr"])
            b = pd.read_csv(d2 / "trial_predictions.csv.gz", usecols=["session_id", "trial", "ll_lr"])
            a = a.sort_values(["session_id", "trial"], kind="stable").reset_index(drop=True)
            a["k"] = k_since_ignore(a)
            e = a[(a["response"] != 2) & (a["use"] == "scored")].merge(b, on=["session_id", "trial"], suffixes=("_3", "_2"))
            assert e["ll_lr_3"].notna().all() and e["ll_lr_2"].notna().all()
            for name, lo, hi in BINS:
                s = e[(e["k"] >= lo) & (e["k"] <= hi)]
                acc[name][0] += len(s); acc[name][1] += float(s["ll_lr_3"].sum()); acc[name][2] += float(s["ll_lr_2"].sum())
            s = e[e["k"].isna()]
            acc["none"][0] += len(s); acc["none"][1] += float(s["ll_lr_3"].sum()); acc["none"][2] += float(s["ll_lr_2"].sum())
        folds[fold] = {b: {"n_trials": v[0], "ll_lr_sum_3way": v[1], "ll_lr_sum_2way": v[2]} for b, v in acc.items()}
        print(fold, {b: v["n_trials"] for b, v in folds[fold].items()}, flush=True)
    meta = build_meta("analysis/freeze_ignore_split.py", [], study_root=STUDY, kind="freeze", code=[__file__],
                      inputs=[], sources=sources)
    write_artifact(OUT, {
        "_meta": meta,
        "estimand": "held-out L/R log-likelihood sums over scored engaged trials, per fold (both passes), "
                    "per bin of k = trials since the last ignored trial in the session",
        "bins": [b for b, _, _ in BINS] + ["none"],
        "models": PAIRS,
        "folds": folds,
    })


if __name__ == "__main__":
    main()
