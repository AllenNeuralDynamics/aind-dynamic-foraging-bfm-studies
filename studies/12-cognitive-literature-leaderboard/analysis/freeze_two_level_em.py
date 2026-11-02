#!/usr/bin/env python
"""Freeze the 2-level EM scores of the literature models (``make freeze-two-level-em``).

Reads the per-model output files that MEvolve's ``mevolve.score`` CLI wrote for the two-level-em
launches (one JSON per model, under /allen) and writes one row per (literature model, subject) to
``analysis/two_level_em_fits.csv``. ``analysis/two_level_em_fits.json`` records, per model, the
source file's sha256 (the immutable key for its numbers), the MEvolve SHA, the program and
data-freeze digests and the settings. Per-session scores stay in the source files. Needs /allen,
so it runs on a compute node; r3_2level_em.py reads only the committed files.
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

# The two-level-em launches, pinned: the Hattori2019 cost run and the other 18 (variant notes).
OUTPUT_ROOT = Path("/allen/aind/scratch/han.hou/outputs/study12/two-level-em")
LAUNCHES = {"two-level-em@20261102-003154": OUTPUT_ROOT / "20261102-003154",
            "two-level-em@20261102-004644": OUTPUT_ROOT / "20261102-004644"}
MEVOLVE_SHA = "6af7d58a0a2dadb4d4eec7ab72108c6350e8dfbe"
FREEZE_MANIFEST = "cfea3d1e21f49ed7baa42f11e1fdda66fd5901c519a47773c08716e062e7eef0"
SETTINGS = {"regime": "2level", "max_em": 100, "n_draws": 2048, "seed": 0}
N_MODELS, N_SUBJECTS = 19, 763
COLUMNS = ["literature_model", "subject_id", "pool", "log_lik", "n_trials", "fit_log_lik",
           "n_fit_trials", "em_iterations", "converged", "ess_median", "ess_min",
           "typical_params", "log_sigma"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows, runs, sources = [], [], []
    for group, directory in LAUNCHES.items():
        for path in sorted(directory.glob("2level_*.json")):
            d = json.loads(path.read_text())
            assert d["mevolve_git_sha"] == MEVOLVE_SHA, f"{path}: MEvolve {d['mevolve_git_sha']}"
            assert d["freeze_manifest_sha256"] == FREEZE_MANIFEST, f"{path}: other data freeze"
            assert all(d[k] == v for k, v in SETTINGS.items()), f"{path}: not the reported settings"
            assert sorted(d["pools"]) == ["heldout", "train"], f"{path}: pools {d['pools']}"
            digest = _sha256(path)
            runs.append({"literature_model": d["model"], "wandb_group": group, "path": str(path),
                         "sha256": digest, "program_sha256": d["program_sha256"],
                         "wall_seconds": d["wall_seconds"], "n_not_converged": d["n_not_converged"]})
            sources.append({"type": "file", "id": str(path), "sha256": digest})
            for r in d["subjects"]:
                ess = [e for e in r["ess"] if e is not None]
                rows.append({"literature_model": d["model"], "subject_id": str(r["subject_id"]),
                             "pool": r["pool"], "log_lik": r["log_lik"], "n_trials": r["n_trials"],
                             "fit_log_lik": r["fit_log_lik"], "n_fit_trials": r["n_fit_trials"],
                             "em_iterations": r["em_iterations"], "converged": r["converged"],
                             "ess_median": statistics.median(ess) if ess else "",
                             "ess_min": min(ess) if ess else "",
                             "typical_params": json.dumps(r["typical_params"], sort_keys=True),
                             "log_sigma": json.dumps(r["log_sigma"])})

    # Every check runs before any write.
    models = [r["literature_model"] for r in runs]
    assert len(models) == len(set(models)) == N_MODELS, f"{len(models)} model files: {models}"
    per_model = {}
    for row in rows:
        per_model.setdefault(row["literature_model"], set()).add(row["subject_id"])
    assert all(len(s) == N_SUBJECTS for s in per_model.values()), "a model lacks subjects"
    assert len({frozenset(s) for s in per_model.values()}) == 1, "models scored different subjects"

    rows.sort(key=lambda r: (r["literature_model"], r["subject_id"]))
    out_csv = HERE / "two_level_em_fits.csv"
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    # No W&B runs exist: the launch labels identify the launches (runs[].wandb_group), so the
    # artifact names no W&B group.
    meta = build_meta("analysis/freeze_two_level_em.py", [], study_root=HERE.parent,
                      kind="freeze", code=[__file__], sources=sources)
    write_artifact(HERE / "two_level_em_fits.json",
                   {"_meta": meta, "columns": COLUMNS, "mevolve_git_sha": MEVOLVE_SHA,
                    "freeze_manifest_sha256": FREEZE_MANIFEST, "settings": SETTINGS,
                    "runs": sorted(runs, key=lambda r: r["literature_model"])},
                   outputs=[out_csv])
    print(f"froze {len(rows)} rows from {len(runs)} models to {out_csv}")


if __name__ == "__main__":
    main()
