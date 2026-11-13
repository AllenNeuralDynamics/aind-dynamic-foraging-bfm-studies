#!/usr/bin/env python
"""Fold assignment of study 17 (PLAN.md, "Design"; issue #148), frozen before any training.

- test fold: the freeze's 149 held-out subjects (`pool == heldout`), scored by model 0;
- pool folds 1-4: the 614 training-pool subjects, dealt into 4 parts within strata of
  tier (study 40 discovery / confirmation / excluded) x curriculum x session-count third,
  shuffled with a fixed seed, so every tier, curriculum and session-count range spreads across
  all 4 folds.

Fold model k trains on every subject outside pool fold k (the other 3 pool folds plus the test
fold). Writes ``reference/folds.json`` and the variant ``pool-folds`` sweep, whose grid point k
carries fold model k's training subjects. Reads the MEvolve data freeze on /allen (snapshot
20260603) and checks its manifest against study 40's cohort file. Needs pandas; run on a compute
node:

    make -C studies/17-gru-crossfit-predictability freeze-folds PY=<mevolve-search env>/bin/python
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

FREEZE = Path("/allen/aind/scratch/han.hou/mevolve/freeze/20260603")
COHORTS = STUDY.parent / "40-cognitive-llm-discovery" / "reference" / "cohorts_seed0_monitor.json"
OUT = STUDY / "reference" / "folds.json"
SWEEP_TEMPLATE = STUDY / "variants" / "pool-folds" / "sweep.template.yaml"
SWEEP = STUDY / "variants" / "pool-folds" / "sweep.yaml"
K = 4
SEED = 0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    cohorts = json.loads(COHORTS.read_text())
    manifest = FREEZE / "manifest.json"
    assert sha256(manifest) == cohorts["freeze_manifest_sha256"], "freeze differs from study 40's"
    sessions = pd.read_parquet(FREEZE / "sessions.parquet", columns=["subject_id", "pool", "curriculum"])
    sessions["subject_id"] = sessions["subject_id"].astype(str)
    subjects = sessions.groupby("subject_id").agg(pool=("pool", "first"), curriculum=("curriculum", "first"),
                                                  n_sessions=("pool", "size"))
    test = sorted(subjects.index[subjects["pool"] == "heldout"])
    pool = subjects[subjects["pool"] == "train"].copy()
    assert (len(pool), len(test)) == (614, 149)

    discovery = set().union(*map(set, cohorts["cohorts"].values()))
    excluded = set(cohorts["excluded"])
    pool["tier"] = ["discovery" if s in discovery else "excluded" if s in excluded else "confirmation"
                    for s in pool.index]
    assert pool["tier"].value_counts().to_dict() == {"confirmation": 315, "discovery": 296, "excluded": 3}
    pool["third"] = pd.qcut(pool["n_sessions"].rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str)

    # Deal each stratum round-robin; carry the next fold across strata so fold sizes stay within 1.
    rng = random.Random(SEED)
    fold_of: dict[str, int] = {}
    nxt = 0
    for _, group in pool.groupby(["tier", "curriculum", "third"], sort=True):
        ids = sorted(group.index)
        rng.shuffle(ids)
        for s in ids:
            fold_of[s] = nxt + 1
            nxt = (nxt + 1) % K
    folds = {str(k): sorted(s for s, f in fold_of.items() if f == k) for k in range(1, K + 1)}
    sizes = {k: len(v) for k, v in folds.items()}
    assert max(sizes.values()) - min(sizes.values()) <= 1 and sum(sizes.values()) == 614
    balance = {k: {col: pool.loc[v, col].value_counts().sort_index().to_dict()
                   for col in ("tier", "curriculum", "third")} for k, v in folds.items()}
    for k in folds:   # every tier and curriculum reaches every fold
        assert set(balance[k]["tier"]) >= {"discovery", "confirmation"}
        assert set(balance[k]["curriculum"]) == set(pool["curriculum"])
    train = {k: sorted(set(test) | (set(pool.index) - set(v))) for k, v in folds.items()}

    meta = build_meta("analysis/freeze_folds.py", [], study_root=STUDY, kind="freeze", code=[__file__],
                      inputs=[COHORTS, SWEEP_TEMPLATE],
                      sources=[{"type": "file", "id": str(manifest), "sha256": sha256(manifest)},
                               {"type": "file", "id": str(FREEZE / "sessions.parquet"),
                                "sha256": sha256(FREEZE / "sessions.parquet")}])
    write_artifact(OUT, {
        "_meta": meta,
        "snapshot": "20260603",
        "freeze_manifest_sha256": cohorts["freeze_manifest_sha256"],
        "seed": SEED,
        "strata": "tier x curriculum x session-count third, dealt round-robin into 4 pool folds",
        "sizes": {"test": len(test), **{f"pool_{k}": n for k, n in sizes.items()}},
        "balance": balance,
        "folds": {"test": test, **{f"pool_{k}": v for k, v in folds.items()}},
        "fold_of": {**{s: "test" for s in test}, **{s: f"pool_{f}" for s, f in sorted(fold_of.items())}},
    })

    # The pool-folds sweep: one grid point per fold model, carrying its training subjects.
    values = "\n".join(f"      - {json.dumps([int(s) for s in train[k]], separators=(',', ':'))}"
                       for k in sorted(train))
    SWEEP.write_text(SWEEP_TEMPLATE.read_text().replace("      - __TRAIN_SUBJECT_LISTS__", values))


if __name__ == "__main__":
    main()
