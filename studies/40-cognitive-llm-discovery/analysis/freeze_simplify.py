#!/usr/bin/env python
"""Freeze one simplification run (``variants/simplify-NN``) and its floor-program candidates (ADR 0008).

Reads the run directory MEvolve copied out at the end and checks every file the W&B archive recorded
(``archive.json``) against it. Writes ``simplify/<variant>/run.json``: every program (generation,
island, parent, correct, Halstead effort, validation score, parameter and latent counts, name, code
sha256), the Claude cost, the stop and the candidates: the five correct programs of lowest Halstead
effort, distinct by code (ties broken by proposal number), then ranks 6 to 10 as replacements. Each
candidate's code is written to ``simplify/<variant>/programs/c<k>_<id8>.py`` and its effort is
recomputed from the code with MEvolve's rule, which must equal the run's.

Needs /allen and MEvolve 4465615 on PYTHONPATH (``mevolve.search.simplify``); reads files only:

    python analysis/freeze_simplify.py --variant simplify-90
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

LAUNCH = "20261108-230437"
RUNS = Path("/allen/aind/scratch/han.hou/mevolve/runs")
N_CANDIDATES, N_REPLACEMENTS = 5, 5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["simplify-90", "simplify-75", "simplify-50"])
    a = ap.parse_args()
    from mevolve.search.simplify import halstead_effort

    run = RUNS / a.variant / LAUNCH / "search"
    archive = json.loads((run / "archive.json").read_text())
    sources = [{"type": "url", "id": f"wandb-artifact://{archive['artifact']}"}]
    for name, sha in archive["sha256"].items():
        assert _sha256(run / name) == sha, f"{run / name} differs from the archived file"
        sources.append({"type": "file", "id": f"{archive['artifact']}/{name}", "sha256": sha})
    floor = json.loads((STUDY / "variants" / a.variant / "run.json").read_text())["floor"]
    with tempfile.TemporaryDirectory() as tmp:
        for suffix in ("", "-wal"):
            if (run / f"programs.sqlite{suffix}").exists():
                shutil.copyfile(run / f"programs.sqlite{suffix}", Path(tmp) / f"programs.sqlite{suffix}")
        con = sqlite3.connect(Path(tmp) / "programs.sqlite")
        con.row_factory = sqlite3.Row
        db = {r["id"]: r for r in con.execute("SELECT id, code, combined_score, correct, public_metrics FROM programs")}
        rows = list(csv.DictReader((run / "proposals.csv").open()))
        programs = []
        for r in rows:
            row = db[r["id"]]
            pub = json.loads(row["public_metrics"] or "{}")
            score = row["combined_score"]
            programs.append({
                "generation": int(r["generation"]), "id": r["id"], "parent_id": r["parent_id"] or None,
                "island": int(r["island"]), "seed": r["seed"] or None, "correct": r["correct"] == "True",
                "effort": -float(score) if score is not None and math.isfinite(float(score)) and float(score) < 0 else None,
                "validation_x100": pub.get("validation_score"), "n_params": pub.get("n_params"),
                "n_latent": pub.get("n_latent"), "name": r["name"] or None, "cost_usd": float(r["cost_usd"] or 0),
                "code_sha256": hashlib.sha256(row["code"].encode()).hexdigest()})
        correct = sorted((p for p in programs if p["generation"] > 0 and p["correct"] and p["effort"] is not None),
                         key=lambda p: (p["effort"], p["generation"]))
        seen, ranked = set(), []
        for p in correct:
            if p["code_sha256"] not in seen:
                seen.add(p["code_sha256"])
                ranked.append(p)
        out_dir = HERE / "simplify" / a.variant
        prog_dir = out_dir / "programs"
        prog_dir.mkdir(parents=True, exist_ok=True)
        candidates = []
        for k, p in enumerate(ranked[: N_CANDIDATES + N_REPLACEMENTS], start=1):
            code = db[p["id"]]["code"]
            effort = halstead_effort(code)
            assert abs(effort - p["effort"]) < 1e-6, f"{p['id']}: effort {effort} != run's {p['effort']}"
            assert p["validation_x100"] >= round(floor, 3) - 1e-9, f"{p['id']} below the floor"
            path = prog_dir / f"c{k}_{p['id'][:8]}.py"
            path.write_text(code)
            candidates.append({"rank": k, "file": str(path.relative_to(STUDY)), **p,
                               "role": "candidate" if k <= N_CANDIDATES else "replacement"})
        con.close()
    summary = json.loads((run / "run_summary.json").read_text())
    stop = json.loads((run / "stop_rule.json").read_text()) if (run / "stop_rule.json").exists() else None
    seeds = [p for p in programs if p["generation"] == 0]
    write_artifact(out_dir / "run.json", {
        "_meta": build_meta(f"analysis/{Path(__file__).name}", [f"{a.variant}@{LAUNCH}"], study_root=STUDY,
                            kind="freeze", code=[__file__], inputs=[STUDY / "variants" / a.variant / "run.json"],
                            sources=sources),
        "variant": a.variant, "floor_x100": floor,
        "archive": {k: archive[k] for k in ("artifact", "digest", "n_files", "bytes", "archive_run")},
        "summary": {k: summary.get(k) for k in ("seconds", "calls", "cost_usd", "stop_reasons", "errors")},
        "stop_rule": stop, "seed_effort_min": min(p["effort"] for p in seeds if p["effort"] is not None),
        "candidates": candidates, "programs": programs})
    print(a.variant, {"programs": len(programs), "correct": len(correct), "distinct": len(ranked),
                      "candidates": [(c["rank"], round(c["effort"]), c["validation_x100"]) for c in candidates]})


if __name__ == "__main__":
    main()
