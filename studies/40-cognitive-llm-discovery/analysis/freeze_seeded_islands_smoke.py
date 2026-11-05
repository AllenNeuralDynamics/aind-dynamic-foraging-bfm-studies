#!/usr/bin/env python
"""Freeze the seeded-islands-smoke run into one committed file (``make freeze-seeded-islands-smoke``).

Reads the run directory of ``seeded-islands-smoke@20261105-011815`` from its W&B archive (the
artifact ``archive.json`` names, downloaded at its digest), not from /allen: the job's exit trap copied
the node-local database over the archived one after the archive, so the bytes on /allen differ
(same rows; SQLite had folded its log into the main file). Writes
``analysis/seeded_islands_smoke/run.json``: the run's summary, every Claude call's duration and cost
from the ledger, one row per program (generation, parent, island, seed, correct, score, per-subject
and 2-level validation scores, whether its 2-level fit failed, whether it was a repeat, parameter and
latent counts, evaluation seconds, cost, failure reason) and the proposals that left no program. It
checks the downloaded files against the sha256 values ``archive.json`` recorded at upload.

Shinka's database is in WAL mode with a non-empty ``-wal``; both are read together from the download.
Needs the W&B SDK and the key in ``~/.config/mevolve/`` (``WANDB_API_KEY``), and /allen for
``archive.json``, so it runs on a compute node; ``seeded_islands_smoke.py`` reads only the committed file.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

GROUP = "seeded-islands-smoke@20261105-011815"
RECORD = Path("/allen/aind/scratch/han.hou/mevolve/runs/seeded-islands-smoke/20261105-011815/search/archive.json")
OUT = HERE / "seeded_islands_smoke" / "run.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _num(x):
    return float(x) if isinstance(x, (int, float)) else None


def main() -> None:
    import wandb

    archive = json.loads(RECORD.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        art = wandb.Api().artifact(archive["artifact"])
        assert art.digest == archive["digest"], f"{archive['artifact']} has digest {art.digest}"
        RUN = Path(art.download(root=str(Path(tmp) / "run")))
        main_with(RUN, archive)


def main_with(RUN: Path, archive: dict) -> None:
    sources = [{"type": "url", "id": f"wandb-artifact://{archive['artifact']}"}]   # its digest is in "archive"
    for name, sha in archive["sha256"].items():
        assert _sha256(RUN / name) == sha, f"{RUN / name} differs from the archived file"
        sources.append({"type": "file", "id": f"{archive['artifact']}/{name}", "sha256": sha})
    sources.append({"type": "file", "id": f"{archive['artifact']}/proposals.csv", "sha256": _sha256(RUN / "proposals.csv")})

    with tempfile.TemporaryDirectory() as tmp:
        for suffix in ("", "-wal"):
            shutil.copy2(RUN / f"programs.sqlite{suffix}", Path(tmp) / f"programs.sqlite{suffix}")
        con = sqlite3.connect(Path(tmp) / "programs.sqlite")
        con.row_factory = sqlite3.Row
        db = {r["id"]: r for r in con.execute("SELECT id, public_metrics, private_metrics, text_feedback, metadata FROM programs")}
        programs = []
        for r in csv.DictReader((RUN / "proposals.csv").open()):
            row = db[r["id"]]
            pub, priv = json.loads(row["public_metrics"] or "{}"), json.loads(row["private_metrics"] or "{}")
            val = priv.get("validation") or {}
            correct = r["correct"] == "True"
            programs.append({
                "generation": int(r["generation"]), "id": r["id"], "parent_id": r["parent_id"] or None,
                "island": int(r["island"]), "seed": r["seed"] or None, "correct": correct,
                "score": float(r["score"]) if correct else None,
                "validation_per_subject": _num(val.get("per_subject")), "validation_two_level": _num(val.get("two_level")),
                "two_level_failed": val.get("two_level_failed"), "repeat_of": priv.get("repeat_of"),
                "n_params": pub.get("n_params"), "n_latent": pub.get("n_latent"),
                "eval_seconds": float(r["eval_seconds"] or 0), "cost_usd": float(r["cost_usd"] or 0),
                "name": r["name"] or None,
                "failure": None if correct else (row["text_feedback"] or "").removeprefix("The program failed and was not scored: ")})
        con.close()
    seen = {p["generation"] for p in programs}
    no_program = []
    for g in sorted(set(range(1, max(seen) + 1)) - seen):
        f = json.loads((RUN / f"gen_{g}" / "failure.json").read_text())
        sources.append({"type": "file", "id": f"{archive['artifact']}/gen_{g}/failure.json",
                        "sha256": _sha256(RUN / f"gen_{g}" / "failure.json")})
        no_program.append({"generation": g, "failure_class": f.get("failure_class"), "failure_stage": f.get("failure_stage")})
    calls = [json.loads(line) for line in (RUN / "ledger.jsonl").read_text().splitlines()]
    summary = json.loads((RUN / "run_summary.json").read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_artifact(OUT, {
        "_meta": build_meta("analysis/freeze_seeded_islands_smoke.py", [GROUP], study_root=STUDY, kind="freeze",
                            code=[__file__], sources=sources),
        "about": "The seeded-islands-smoke run, read from its run directory and checked against its W&B archive.",
        "archive": {"artifact": archive["artifact"], "digest": archive["digest"], "n_files": archive["n_files"],
                    "bytes": archive["bytes"]},
        "summary": {k: summary[k] for k in ("seconds", "calls", "cost_usd", "stop_reasons", "errors")},
        "calls": [{"seconds": c["seconds"], "cost": c["cost"], "stop_reason": c["stop_reason"],
                   "output_tokens": c["output_tokens"]} for c in calls],
        "programs": programs, "no_program": no_program})
    print(f"wrote {len(programs)} programs, {len(no_program)} proposals without one, {len(calls)} calls to {OUT}")


if __name__ == "__main__":
    main()
