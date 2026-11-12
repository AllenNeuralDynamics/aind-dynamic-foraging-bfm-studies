#!/usr/bin/env python
"""The simplification floors' anchors (ADR 0005, ADR 0008), rescored on the validation subjects.

A floor keeps a share of the gain from the best literature model (Miller-RHG) to the search's best, in
validation 2-level EM scores. The simplification run checks each proposal against its floor with the
search's evaluator settings (``max_em`` 20), so both anchors are scored with those settings and the
per-subject fallback (MEvolve 146153c), not taken from the search, whose evaluator had no per-subject
fallback. The five candidates are all rescored, before the selection picks one, so the anchor of
whichever is chosen is fixed without looking at a selection or test score.

Needs /allen and MEvolve 146153c on PYTHONPATH; on a compute node:

    sbatch -p aind --ntasks=1 --cpus-per-task=24 --mem-per-cpu=3000M --time=04:00:00 --wrap \\
      "env PYTHONPATH=<mevolve>/src <mevolve-search env>/bin/python analysis/freeze_floor_anchors.py"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

FREEZE = Path("/allen/aind/scratch/han.hou/mevolve/freeze/20260603")
COHORTS = STUDY / "reference" / "cohorts_seed0.json"
PROGRAMS = HERE / "selection" / "programs"
REFERENCE = Path("/allen/aind/scratch/han.hou/mevolve/selection/mevolve-146153c/src/mevolve/programs/literature/miller_rhg.py")
OUT = HERE / "simplify" / "floor_anchors.json"
MEVOLVE_SHA = "146153c01f6391122414525022e9126f200a1586"
MAX_EM, N_DRAWS, SEED = 20, 2048, 0          # the search's evaluator settings (seeded-islands-2)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    from mevolve import evaluate
    from mevolve.data import Freeze

    subjects = json.loads(COHORTS.read_text())["cohorts"]["validation"]
    freeze = Freeze(FREEZE)
    data = [evaluate.subject_data(freeze, s) for s in subjects]
    programs = {"reference_miller_rhg": REFERENCE} | {p.stem: p for p in sorted(PROGRAMS.glob("rank*.py"))}
    out = {}
    for name, path in programs.items():
        records = evaluate.evaluate(str(path), data, regime="2level", workers=a.workers, max_em=MAX_EM,
                                    n_draws=N_DRAWS, isolate=True)
        s = evaluate.program_score(records)
        out[name] = {"program_sha256": _sha(path), "score_x100": 100 * s["score"],
                     "two_level_x100": 100 * s["two_level"], "per_subject_x100": 100 * s["per_subject"],
                     "two_level_failed": s["two_level_failed"], "n_fallback": s["n_fallback"],
                     "fallback_subjects": s["fallback_subjects"]}
        print(name, round(out[name]["score_x100"], 4), s["n_fallback"], flush=True)
    meta = build_meta(f"analysis/{Path(__file__).name}", [], study_root=STUDY, kind="freeze", code=[__file__],
                      inputs=[COHORTS, *(p for n, p in programs.items() if n != "reference_miller_rhg")],
                      sources=[{"type": "file", "id": str(REFERENCE), "sha256": _sha(REFERENCE)},
                               {"type": "file", "id": str(FREEZE / "manifest.json"), "sha256": _sha(FREEZE / "manifest.json")},
                               {"type": "url", "id": "https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-mevolve/commit/"
                                + MEVOLVE_SHA}])
    OUT.parent.mkdir(exist_ok=True)
    write_artifact(OUT, {"_meta": meta, "cohort": "validation", "n_subjects": len(subjects),
                         "settings": {"regime": "2level", "max_em": MAX_EM, "n_draws": N_DRAWS, "seed": SEED,
                                      "fallback_tol": evaluate.FALLBACK_TOL, "mevolve": MEVOLVE_SHA},
                         "programs": out})


if __name__ == "__main__":
    main()
