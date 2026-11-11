#!/usr/bin/env python
"""Each floor's program (ADR 0008, floors amendment). Offline.

For each simplification run, its candidates (``simplify/<variant>/run.json``) scored on the selection
subjects (``selection/selection/<variant>_c<k>_<id8>.json``, ``freeze_selection.py``) are checked with
the gate as amended (replay and rollout fingerprint scores and curve RMSEs all no larger than
Miller-RHG's) and against the selection floor (``simplify/floors.json``). The floor program is the
passing candidate of lowest Halstead effort; the replacements (ranks 6 to 10) are checked only if no
candidate qualifies, and no choice is published while a candidate of the tier in play is unscored. A candidate whose code equals an already scored one's once
docstrings are dropped (the same syntax tree) is the same program: it takes that score instead of being
fitted again, and is marked ``same_program_as``. Writes ``simplify_choice.json``.

    python analysis/simplify_choice.py
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, write_artifact  # noqa: E402
from selection_gate import GATE, REFERENCE, get  # noqa: E402

FLOORS = HERE / "simplify" / "floors.json"
SEL = HERE / "selection" / "selection"
VARIANTS = ("simplify-90", "simplify-75", "simplify-50")


def program_key(source: str) -> str:
    """The program without its docstrings and comments: its syntax tree, hashed."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            body = getattr(node, field, None)
            if isinstance(body, list):
                kept = [x for x in body if not (isinstance(x, ast.Expr) and isinstance(x.value, ast.Constant)
                                                and isinstance(x.value.value, str))]
                setattr(node, field, kept or body)
    return hashlib.sha256(ast.dump(tree, annotate_fields=False).encode()).hexdigest()


def main() -> None:
    floors = json.loads(FLOORS.read_text())["runs"]
    ref_path = SEL / f"{REFERENCE}.json"
    ref = json.loads(ref_path.read_text())
    inputs, out = [FLOORS, ref_path], {}
    for v in VARIANTS:
        run_path = HERE / "simplify" / v / "run.json"
        run = json.loads(run_path.read_text())
        inputs.append(run_path)
        rows, scored, chosen = [], {}, None
        for role in ("candidate", "replacement"):   # ADR 0008: the replacements only if no candidate qualifies
            tier = [c for c in run["candidates"] if c["role"] == role]
            for c in tier:
                inputs.append(STUDY / c["file"])         # the program decides any reuse of another's score
                name = f"{v}_c{c['rank']}_{c['id'][:8]}"
                path = SEL / f"{name}.json"
                key = program_key((STUDY / c["file"]).read_text())
                same = None
                if not path.exists():
                    # no choice while a candidate that could outrank the winner is unscored (PR #138 review)
                    assert key in scored, f"{name}: not scored on the selection subjects; no choice is published"
                    same, path = scored[key]
                else:
                    scored.setdefault(key, (name, path))
                if path not in inputs:
                    inputs.append(path)
                r = json.loads(path.read_text())
                # a full cohort, and likelihood and fingerprints from the same fits (PR #138 review)
                assert r["n_subjects"] == len(r["subjects"]) == 100, f"{path.name}: not the 100 selection subjects"
                assert not r["score"]["two_level_failed"], f"{path.name}: 2-level fit failed; fingerprints would differ"
                vals = {k: get(r, k) for k in GATE}
                gate = all(vals[k] is not None and vals[k] <= get(ref, k) for k in GATE)
                score = 100 * r["score"]["score"]
                rows.append({"name": name, "rank": c["rank"], "role": c["role"], "effort": c["effort"],
                             "validation_x100": c["validation_x100"], "n_params": c["n_params"],
                             "n_latent": c["n_latent"], "score_x100": score, "n_fallback": r["score"]["n_fallback"],
                             **vals, "passes_gate": gate, "reaches_floor": score >= floors[v]["selection_floor"],
                             "file": c["file"], "same_program_as": same})
            ok = [x for x in rows if x["passes_gate"] and x["reaches_floor"]]
            chosen = min(ok, key=lambda x: (x["effort"], x["rank"])) if ok else None
            if chosen:
                break
        out[v] = {"selection_floor_x100": floors[v]["selection_floor"], "validation_floor_x100": floors[v]["floor"],
                  "candidates": rows, "chosen": chosen and chosen["name"], "chosen_file": chosen and chosen["file"]}
        print(v, "chosen", out[v]["chosen"], [(x["rank"], round(x["effort"]), round(x["score_x100"], 3),
                                               x["passes_gate"], x["reaches_floor"]) for x in rows])
    write_artifact(HERE / "simplify_choice.json", {
        "_meta": build_meta(f"analysis/{Path(__file__).name}", [], study_root=STUDY,
                            code=[__file__, HERE / "selection_gate.py"],   # the gate rule it imports
                            inputs=inputs),
        "reference": {k: get(ref, k) for k in GATE} | {"score_x100": 100 * ref["score"]["score"]},
        "rule": "the passing candidate of lowest Halstead effort: replay and rollout fingerprint scores and curve "
                "RMSEs all <= Miller-RHG's, and selection score >= the selection floor (study 40 ADR 0008)",
        "floors": out})


if __name__ == "__main__":
    main()
