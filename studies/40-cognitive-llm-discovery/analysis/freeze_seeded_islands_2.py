#!/usr/bin/env python
"""Freeze the main search, seeded-islands-2 (``seeded-islands@20261106-114959``), into one committed file.

Reads the run directory MEvolve copied out at the end of the search and checks every file the W&B archive
recorded (``archive.json``: sha256 per file at upload) against it, so the freeze is the archived run.
Writes ``analysis/seeded_islands_2/run.json``: every program (generation, island, parent, seed, score at
full precision, validation per-subject and 2-level, parameter and latent counts and names, timestamp,
name, cost),
every Claude call (time, cost, stop reason, output tokens), the stopping rule's record, the final
archive's members, the best program's code, and the Slurm driver job and evaluation jobs from ``sacct``.

Needs /allen and ``sacct``, so it runs on a compute node (it reads files only; no W&B call):

    python analysis/freeze_seeded_islands_2.py
"""
from __future__ import annotations

import ast
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

GROUP = "seeded-islands@20261106-114959"
JOB = "27675923"                                    # launch_record/seeded-islands-2/hpc_resubmit1.json
RUN = Path("/allen/aind/scratch/han.hou/mevolve/runs/seeded-islands/20261106-114959/search")
OUT = HERE / "seeded_islands_2" / "run.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _num(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x


def _seconds(elapsed: str) -> int:
    days, _, hms = elapsed.rpartition("-")
    h, m, s = (int(v) for v in hms.split(":"))
    return int(days or 0) * 86400 + h * 3600 + m * 60 + s


def slurm() -> dict:
    def rows(args):
        out = subprocess.run(["sacct", "-n", "-P", "-X", *args, "-o", "JobID,JobName,AllocCPUS,Elapsed,State,Start,End"],
                             capture_output=True, text=True, check=True).stdout.splitlines()
        return [dict(zip(("id", "name", "cpus", "elapsed", "state", "start", "end"), r.split("|"))) for r in out]
    job = rows(["-j", JOB])[0]
    evals = [r for r in rows(["-S", job["start"], "-E", job["end"], "--name", "mevolve-eval"])]
    states: dict = {}
    for r in evals:
        states[r["state"]] = states.get(r["state"], 0) + 1
    return {"job": JOB, "cpus": int(job["cpus"]), "seconds": _seconds(job["elapsed"]), "state": job["state"],
            "start": job["start"], "end": job["end"],
            "evaluation_jobs": {"n": len(evals), "states": states,
                                "cpu_hours": round(sum(int(r["cpus"]) * _seconds(r["elapsed"]) for r in evals) / 3600, 1)}}


def names(code: str) -> tuple[list, list]:
    """The program's parameter names (``PARAMS`` keys) and latent names (``LATENT_NAMES``), read from its
    syntax tree without running it; they carry the mechanisms (r2 tags programs by them)."""
    params, latents = [], []
    try:
        body = ast.parse(code).body
    except SyntaxError:      # a failed edit's text (merge markers): never scored
        return params, latents
    for node in body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "PARAMS" and isinstance(node.value, ast.Dict):
                params = [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
            if node.targets[0].id == "LATENT_NAMES" and isinstance(node.value, (ast.Tuple, ast.List)):
                latents = [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
    return params, latents


def main() -> None:
    archive = json.loads((RUN / "archive.json").read_text())
    sources = [{"type": "url", "id": f"wandb-artifact://{archive['artifact']}"},   # its digest is in "archive"
               {"type": "url", "id": f"slurm-sacct://{JOB}"}]
    for name, sha in archive["sha256"].items():
        assert _sha256(RUN / name) == sha, f"{RUN / name} differs from the archived file"
        sources.append({"type": "file", "id": f"{archive['artifact']}/{name}", "sha256": sha})
    for name in ("proposals.csv", "stop_rule.json", "run_summary.json"):
        sources.append({"type": "file", "id": f"{archive['artifact']}/{name}", "sha256": _sha256(RUN / name)})

    with tempfile.TemporaryDirectory() as tmp:
        for suffix in ("", "-wal"):
            if (RUN / f"programs.sqlite{suffix}").exists():
                shutil.copyfile(RUN / f"programs.sqlite{suffix}", Path(tmp) / f"programs.sqlite{suffix}")
        con = sqlite3.connect(Path(tmp) / "programs.sqlite")
        con.row_factory = sqlite3.Row
        db = {r["id"]: r for r in con.execute("SELECT id, code, combined_score, public_metrics, private_metrics, "
                                              "text_feedback, timestamp, metadata FROM programs")}
        archived = [r[0] for r in con.execute("SELECT program_id FROM archive")]
        programs = []
        for r in csv.DictReader((RUN / "proposals.csv").open()):
            row = db[r["id"]]
            pub, priv = json.loads(row["public_metrics"] or "{}"), json.loads(row["private_metrics"] or "{}")
            val = priv.get("validation") or {}
            correct = r["correct"] == "True"
            programs.append({
                "generation": int(r["generation"]), "id": r["id"], "parent_id": r["parent_id"] or None,
                "island": int(r["island"]), "seed": r["seed"] or None, "correct": correct,
                "score": float(row["combined_score"]) if correct else None,
                "validation_per_subject": _num(val.get("per_subject")), "validation_two_level": _num(val.get("two_level")),
                "two_level_failed": val.get("two_level_failed"), "repeat_of": priv.get("repeat_of"),
                "n_params": pub.get("n_params"), "n_latent": pub.get("n_latent"), "timestamp": float(row["timestamp"]),
                "eval_seconds": float(r["eval_seconds"] or 0), "cost_usd": float(r["cost_usd"] or 0),
                "name": r["name"] or None,
                "code_sha256": hashlib.sha256(row["code"].encode()).hexdigest(),
                "param_names": names(row["code"])[0], "latent_names": names(row["code"])[1],
                "failure": None if correct else (row["text_feedback"] or "").removeprefix("The program failed and was not scored: ")})
        best = max((p for p in programs if p["generation"] > 0 and p["correct"]), key=lambda p: p["score"])
        best_code = db[best["id"]]["code"]
        con.close()
    calls = [json.loads(line) for line in (RUN / "ledger.jsonl").read_text().splitlines()]
    summary = json.loads((RUN / "run_summary.json").read_text())
    stop = json.loads((RUN / "stop_rule.json").read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_artifact(OUT, {
        "_meta": build_meta(f"analysis/{Path(__file__).name}", [GROUP], study_root=STUDY, kind="freeze",
                            code=[__file__], sources=sources),
        "about": "The seeded-islands-2 search, read from its run directory and checked against its W&B archive, "
                 "with its Slurm jobs from sacct.",
        "archive": {k: archive[k] for k in ("artifact", "digest", "n_files", "bytes", "archive_run")},
        "summary": {k: summary.get(k) for k in ("seconds", "calls", "cost_usd", "stop_reasons", "errors")},
        "stop_rule": stop,
        "slurm": slurm(),
        "calls": [{"time": c["time"], "seconds": c["seconds"], "cost": c["cost"], "stop_reason": c["stop_reason"],
                   "output_tokens": c["output_tokens"]} for c in calls],
        "programs": programs, "final_archive": archived,
        "best_code": {"id": best["id"], "code": best_code}})
    print(f"wrote {len(programs)} programs, {len(calls)} calls, archive of {len(archived)} to {OUT}")


if __name__ == "__main__":
    main()
