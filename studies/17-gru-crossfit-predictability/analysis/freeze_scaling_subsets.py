#!/usr/bin/env python
"""Training-mouse subsets for the cross-validated scaling curve (variant ``scaling-folds``).

For each pool fold k, fold model k's training mice (every mouse outside pool fold k:
``reference/folds.json``) are subsampled to D ~ 10, 30, 100 and 300 mice, with the subject ratios
of studies 01/02 (0.016, 0.049, 0.163, 0.489) applied per curriculum. Within a curriculum the mice are
shuffled once (seed 0) and each D takes a prefix of that order, so the subsets are nested
(D=10 within D=30 within D=100 within D=300), as the wrapper's ``subject_ratio`` sampling is. The full-size
point is the ``pool-folds`` model; the test fold's curve is study 02's ``nxd-3way`` grid.

Writes ``reference/scaling_subsets.json`` and the variant's two sweeps from their templates:
``sweep_d100.yaml`` (allocated seats), ``sweep_d10_d30.yaml`` and ``sweep_d300.yaml`` (preemptible).
The test fold's D=300 point (``sweep_test_d300.yaml``) is not written here: it uses study 02's own
``subject_ratio`` selection. Reads the
curriculum of each mouse from the MEvolve data freeze on /allen (snapshot 20260603). Needs pandas;
run on a compute node:

    make -C studies/17-gru-crossfit-predictability freeze-scaling PY=<mevolve-search env>/bin/python
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

FOLDS = STUDY / "reference" / "folds.json"
SESSIONS = Path("/allen/aind/scratch/han.hou/mevolve/freeze/20260603/sessions.parquet")
OUT = STUDY / "reference" / "scaling_subsets.json"
VARIANT = STUDY / "variants" / "scaling-folds"
SWEEPS = {"sweep_d100.yaml": ["d100"], "sweep_d10_d30.yaml": ["d10", "d30"], "sweep_d300.yaml": ["d300"]}
RATIOS = {"d10": 0.016, "d30": 0.049, "d100": 0.163, "d300": 0.489}   # studies 01/02 subject_ratio values
SEED = 0


def curricula_of(fold_of: dict[str, str]) -> dict[str, str]:
    """Curriculum per subject, from the freeze table study 17's folds were built on."""
    import pandas as pd   # only here: the freeze lives on /allen
    s = pd.read_parquet(SESSIONS, columns=["subject_id", "curriculum"])
    s["subject_id"] = s["subject_id"].astype(str)
    cur = s.drop_duplicates("subject_id").set_index("subject_id")["curriculum"].to_dict()
    assert set(fold_of) <= set(cur)
    return {k: cur[k] for k in fold_of}


def main() -> None:
    folds = json.loads(FOLDS.read_text())
    fold_of = folds["fold_of"]
    curriculum = curricula_of(fold_of)
    subsets: dict[str, dict[str, list[str]]] = {}
    for k in ("pool_1", "pool_2", "pool_3", "pool_4"):
        train = sorted(s for s, f in fold_of.items() if f != k)
        rng = random.Random(f"{SEED}-{k}")
        order: dict[str, list[str]] = {}
        for c in sorted({curriculum[s] for s in train}):
            ids = sorted(s for s in train if curriculum[s] == c)
            rng.shuffle(ids)
            order[c] = ids
        subsets[k] = {}
        for d, ratio in RATIOS.items():
            pick = [s for c, ids in order.items() for s in ids[:max(1, round(ratio * len(ids)))]]
            subsets[k][d] = sorted(pick)
        assert set(subsets[k]["d10"]) <= set(subsets[k]["d30"]) <= set(subsets[k]["d100"]) <= set(subsets[k]["d300"])
        assert not set(subsets[k]["d300"]) & set(folds["folds"][k])
    sizes = {k: {d: len(v) for d, v in ds.items()} for k, ds in subsets.items()}

    templates = [VARIANT / f"{name.removesuffix('.yaml')}.template.yaml" for name in SWEEPS]
    meta = build_meta("analysis/freeze_scaling_subsets.py", [], study_root=STUDY, kind="freeze",
                      code=[__file__], inputs=[FOLDS, *templates],
                      sources=[{"type": "file", "id": str(SESSIONS),
                                "sha256": hashlib.sha256(SESSIONS.read_bytes()).hexdigest()}])
    write_artifact(OUT, {
        "_meta": meta,
        "seed": SEED,
        "ratios": RATIOS,
        "rule": "per curriculum, shuffle fold model k's training mice once (seed '0-pool_k') and take "
                "a prefix of round(ratio x n_curriculum); nested across D",
        "sizes": sizes,
        "subsets": subsets,
    })

    # Sweeps: one grid point per (fold, D), carrying that model's training subjects.
    for name, ds in SWEEPS.items():
        values = "\n".join(f"      - {json.dumps([int(s) for s in subsets[k][d]], separators=(',', ':'))}"
                           for d in ds for k in sorted(subsets))
        template = VARIANT / f"{name.removesuffix('.yaml')}.template.yaml"
        (VARIANT / name).write_text(template.read_text().replace("      - __TRAIN_SUBJECT_LISTS__", values))


if __name__ == "__main__":
    main()
