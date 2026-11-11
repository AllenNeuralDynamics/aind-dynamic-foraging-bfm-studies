#!/usr/bin/env python
"""The three simplification floors and each run's seed programs (ADR 0005, ADR 0008). Offline.

Validation floors (what each simplification run checks): Miller-RHG's and the chosen search-best's
validation scores from ``simplify/floor_anchors.json`` (the search's evaluator settings, per-subject
fallback), and floor = low + share x (high - low) for share 0.9, 0.75 and 0.5. Seeds: the five
selection candidates, the chosen one first (island 0) and the rest by search rank, keeping those whose
rescored validation score reaches the floor (a seed below it would stop the run).

Selection floors (how a floor's program is chosen afterwards): the same shares of the gain between
Miller-RHG's and the chosen program's selection scores (``selection/selection/*.json``, ``max_em`` 100).

Checks that each ``variants/simplify-NN/run.json`` carries exactly these floors and seeds. Writes
``simplify/floors.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

ANCHORS = HERE / "simplify" / "floor_anchors.json"
GATE = HERE / "selection_selection.json"
CANDIDATES = STUDY / "reference" / "selection_candidates_seeded-islands-2.json"
SELECTION = HERE / "selection" / "selection"
REFERENCE = "reference_miller_rhg"
SHARES = {"simplify-90": 0.9, "simplify-75": 0.75, "simplify-50": 0.5}
OUT = HERE / "simplify" / "floors.json"


def main() -> None:
    anchors = json.loads(ANCHORS.read_text())["programs"]
    gate = json.loads(GATE.read_text())
    chosen = gate["chosen"]
    assert chosen, "no candidate passed the gate: run the next ranks first (ADR 0008)"
    ranked = [f"rank{x['rank']}_{x['id'][:8]}" for x in json.loads(CANDIDATES.read_text())["ranked"]]
    order = [chosen] + [n for n in ranked if n in anchors and n != chosen]
    lo, hi = anchors[REFERENCE]["score_x100"], anchors[chosen]["score_x100"]
    sel = {n: 100 * json.loads((SELECTION / f"{n}.json").read_text())["score"]["score"] for n in (REFERENCE, chosen)}
    runs = {}
    for variant, share in SHARES.items():
        floor = lo + share * (hi - lo)
        seeds = [n for n in order if anchors[n]["score_x100"] >= floor]
        runs[variant] = {"share": share, "floor": floor,
                         "seed_programs": [f"$STUDY_DIR/analysis/selection/programs/{n}.py" for n in seeds],
                         "selection_floor": sel[REFERENCE] + share * (sel[chosen] - sel[REFERENCE])}
        cfg = json.loads((STUDY / "variants" / variant / "run.json").read_text())
        assert cfg["floor"] == round(floor, 6), f"{variant}: run.json floor {cfg['floor']} != {round(floor, 6)}"
        assert cfg["seed_programs"] == runs[variant]["seed_programs"], f"{variant}: seed programs differ"
    meta = build_meta(f"analysis/{Path(__file__).name}", [], study_root=STUDY, code=[__file__],
                      inputs=[ANCHORS, GATE, CANDIDATES, SELECTION / f"{REFERENCE}.json", SELECTION / f"{chosen}.json",
                              *(STUDY / "variants" / v / "run.json" for v in SHARES)])
    write_artifact(OUT, {"_meta": meta, "chosen": chosen,
                         "validation_anchors_x100": {"low": lo, "high": hi, "gain": hi - lo},
                         "selection_anchors_x100": {"low": sel[REFERENCE], "high": sel[chosen],
                                                    "gain": sel[chosen] - sel[REFERENCE]},
                         "runs": runs})
    print(json.dumps(runs, indent=1))


if __name__ == "__main__":
    main()
