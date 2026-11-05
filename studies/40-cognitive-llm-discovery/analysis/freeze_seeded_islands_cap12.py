#!/usr/bin/env python
"""Freeze the seeded-islands-cap12 run into one committed file (``make freeze-seeded-islands-cap12``).

The same read as ``freeze_seeded_islands_smoke.py``: the run directory of
``seeded-islands-cap12@20261105-100249`` from its W&B archive (the artifact ``archive.json`` names,
downloaded at its digest and checked against the sha256 values recorded at upload). Writes
``analysis/seeded_islands_cap12/run.json`` with the smoke freeze's fields, plus the best program's
code (its constants are a pre-registered number) and the Slurm job's allocation and evaluation steps
from ``sacct`` (the pass rule needs every step to have run).

Needs the W&B SDK and the key in ``~/.config/mevolve/`` (``WANDB_API_KEY``), /allen and ``sacct``, so
it runs on a compute node; ``seeded_islands_cap12.py`` reads only the committed file.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

GROUP = "seeded-islands-cap12@20261105-100249"
JOB = "27634299"                                    # variants/seeded-islands-cap12/notes.md
RECORD = Path("/allen/aind/scratch/han.hou/mevolve/runs/seeded-islands-cap12/20261105-100249/search/archive.json")
OUT = HERE / "seeded_islands_cap12" / "run.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _num(x):
    """A finite number, else None (JSON null): a failed 2-level fit can leave NaN."""
    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else None


def _seconds(elapsed: str) -> int:
    """sacct Elapsed ([D-]HH:MM:SS) in seconds."""
    days, _, hms = elapsed.rpartition("-")
    h, m, s = (int(v) for v in hms.split(":"))
    return int(days or 0) * 86400 + h * 3600 + m * 60 + s


def slurm() -> dict:
    rows = subprocess.run(["sacct", "-j", JOB, "-n", "-P", "-o", "JobID,JobName,NodeList,AllocCPUS,Elapsed,State"],
                          capture_output=True, text=True, check=True).stdout.splitlines()
    rows = [dict(zip(("id", "name", "nodes", "cpus", "elapsed", "state"), r.split("|"))) for r in rows]
    job = next(r for r in rows if r["id"] == JOB)
    return {"job": JOB, "cpus": int(job["cpus"]), "seconds": _seconds(job["elapsed"]), "state": job["state"],
            "steps": [{"step": r["id"], "node": r["nodes"], "seconds": _seconds(r["elapsed"]), "state": r["state"]}
                      for r in rows if r["name"] == "mevolve-eval"]}


def main() -> None:
    import wandb

    archive = json.loads(RECORD.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        art = wandb.Api().artifact(archive["artifact"])
        assert art.digest == archive["digest"], f"{archive['artifact']} has digest {art.digest}"
        RUN = Path(art.download(root=str(Path(tmp) / "run")))
        main_with(RUN, archive, slurm())


def main_with(RUN: Path, archive: dict, job: dict) -> None:
    sources = [{"type": "url", "id": f"wandb-artifact://{archive['artifact']}"},   # its digest is in "archive"
               {"type": "url", "id": f"slurm-sacct://{JOB}", "state": job["state"]}]
    for name, sha in archive["sha256"].items():
        assert _sha256(RUN / name) == sha, f"{RUN / name} differs from the archived file"
        sources.append({"type": "file", "id": f"{archive['artifact']}/{name}", "sha256": sha})
    sources.append({"type": "file", "id": f"{archive['artifact']}/proposals.csv", "sha256": _sha256(RUN / "proposals.csv")})

    with tempfile.TemporaryDirectory() as tmp:
        for suffix in ("", "-wal"):
            if (RUN / f"programs.sqlite{suffix}").exists():
                shutil.copyfile(RUN / f"programs.sqlite{suffix}", Path(tmp) / f"programs.sqlite{suffix}")
        con = sqlite3.connect(Path(tmp) / "programs.sqlite")
        con.row_factory = sqlite3.Row
        db = {r["id"]: r for r in con.execute("SELECT id, code, combined_score, public_metrics, private_metrics, "
                                              "text_feedback FROM programs")}
        programs = []
        for r in csv.DictReader((RUN / "proposals.csv").open()):
            row = db[r["id"]]
            pub, priv = json.loads(row["public_metrics"] or "{}"), json.loads(row["private_metrics"] or "{}")
            val = priv.get("validation") or {}
            correct = r["correct"] == "True"
            programs.append({
                "generation": int(r["generation"]), "id": r["id"], "parent_id": r["parent_id"] or None,
                "island": int(r["island"]), "seed": r["seed"] or None, "correct": correct,
                "score": float(row["combined_score"]) if correct else None,     # full precision; the csv rounds
                "validation_per_subject": _num(val.get("per_subject")), "validation_two_level": _num(val.get("two_level")),
                "two_level_failed": val.get("two_level_failed"), "repeat_of": priv.get("repeat_of"),
                "n_params": pub.get("n_params"), "n_latent": pub.get("n_latent"),
                "eval_seconds": float(r["eval_seconds"] or 0), "cost_usd": float(r["cost_usd"] or 0),
                "name": r["name"] or None,
                "failure": None if correct else (row["text_feedback"] or "").removeprefix("The program failed and was not scored: ")})
        best = max((p for p in programs if p["generation"] > 0 and p["correct"]), key=lambda p: p["score"])
        best_code = db[best["id"]]["code"]
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
        "_meta": build_meta("analysis/freeze_seeded_islands_cap12.py", [GROUP], study_root=STUDY, kind="freeze",
                            code=[__file__], sources=sources),
        "about": "The seeded-islands-cap12 run, read from its run directory and checked against its W&B archive, "
                 "with its Slurm job from sacct.",
        "archive": {"artifact": archive["artifact"], "digest": archive["digest"], "n_files": archive["n_files"],
                    "bytes": archive["bytes"]},
        "summary": {k: summary[k] for k in ("seconds", "calls", "cost_usd", "stop_reasons", "errors")},
        "slurm": job,
        "calls": [{"seconds": c["seconds"], "cost": c["cost"], "stop_reason": c["stop_reason"],
                   "output_tokens": c["output_tokens"]} for c in calls],
        "programs": programs, "no_program": no_program,
        "best_code": {"id": best["id"], "code": best_code}})
    print(f"wrote {len(programs)} programs, {len(no_program)} proposals without one, {len(calls)} calls, "
          f"{len(job['steps'])} evaluation steps to {OUT}")


if __name__ == "__main__":
    main()
