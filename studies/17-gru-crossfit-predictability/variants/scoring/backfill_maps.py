#!/usr/bin/env python
"""Backfill (wrapper #129): log `heldout-rescore-<run>:v1` on each existing scoring run.

The scoring runs were made before wrapper PR #130, so their `:v0` artifact lacks what rebuilds a
fine-tuned held-out model: the fine-tune's `subject_index_map.json`, `session_context_map.json`
(session-conditioned models) and the roles input. For each job of the given launch records this:
  1. finds the job's W&B run (`<out>/wandb/run-*-<id>`) and its single `heldout-rescore` `:v0`;
  2. checks every `:v0` file against the job's scratch output (md5, as W&B stores it);
  3. resumes the run and logs `:v1` = the `:v0` files (same bytes) + the two maps from the
     fine-tune folder + `roles.csv` (this study's committed reference/roles.csv, whose sha256 must
     equal the manifest's), then reads `:v1` back and checks its file list and md5s.
A run that already has a `:v1` is skipped. Results (run id, v0/v1 digests) go to --results.

Run on a compute node, mevolve-search env (wandb 0.31; the shared cpu env cannot read artifacts),
NeuralDynamics key in WANDB_API_KEY:
    python variants/scoring/backfill_maps.py --label score-28 [--only <out>] --results <json>
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
ROLES = STUDY / "reference" / "roles.csv"
SCRATCH = Path("/allen/aind/scratch/han.hou/s17_scoring")
ENTITY, PROJECT = "NeuralDynamics", "dfbfm_17-gru-crossfit-predictability"
MAPS = ("subject_index_map.json", "session_context_map.json")


def md5_b64(path: Path) -> str:
    return base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode()


def run_id_of(out: Path) -> str:
    runs = sorted((out / "wandb").glob("run-*-*"))
    assert len(runs) == 1, f"{out}: expected one W&B run dir, found {runs}"
    return runs[0].name.rsplit("-", 1)[1]


def backfill(api: wandb.Api, out: Path) -> dict:
    rid = run_id_of(out)
    name = f"{ENTITY}/{PROJECT}/heldout-rescore-{rid}"
    versions = sorted(a.version for a in api.artifacts("heldout-rescore", f"{ENTITY}/{PROJECT}/heldout-rescore-{rid}"))
    if "v1" in versions:
        v1 = api.artifact(f"{name}:v1")
        return {"out": str(out), "run_id": rid, "status": "skipped (v1 exists)", "v1_digest": v1.digest}
    assert versions == ["v0"], f"{rid}: unexpected versions {versions}"
    v0 = api.artifact(f"{name}:v0")
    entries = {n: e.digest for n, e in v0.manifest.entries.items()}
    for n, d in entries.items():
        assert md5_b64(out / n) == d, f"{rid}: scratch {n} differs from :v0"
    manifest = json.loads((out / "manifest.json").read_text())
    assert hashlib.sha256(ROLES.read_bytes()).hexdigest() == manifest["roles"]["sha256"], f"{rid}: roles sha256"
    ft = sorted(out.glob("finetune/run/*/target_rescore_pass_*/outputs"))
    assert len(ft) == 1, f"{rid}: fine-tune outputs {ft}"
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for n in entries:
            shutil.copyfile(out / n, stage / n)
        for n in MAPS:
            if (ft[0] / n).exists():
                shutil.copyfile(ft[0] / n, stage / n)
        assert (stage / MAPS[0]).exists(), f"{rid}: no subject_index_map.json"
        shutil.copyfile(ROLES, stage / "roles.csv")
        run = wandb.init(entity=ENTITY, project=PROJECT, id=rid, resume="must",
                         settings=wandb.Settings(silent=True, x_disable_stats=True, console="off"))
        art = wandb.Artifact(f"heldout-rescore-{rid}", type="heldout-rescore",
                             metadata={"backfill": "wrapper#129: adds the fine-tune maps and the roles input",
                                       "supersedes": f"heldout-rescore-{rid}:v0"})
        for f in sorted(stage.iterdir()):
            art.add_file(str(f))
        run.log_artifact(art)
        art.wait()
        run.finish()
        staged = {f.name: md5_b64(f) for f in stage.iterdir()}
    v1 = api.artifact(f"{name}:v1")
    got = {n: e.digest for n, e in v1.manifest.entries.items()}
    assert got == staged, f"{rid}: :v1 files differ from the staged files"
    return {"out": str(out), "run_id": rid, "status": "backfilled", "v0_digest": v0.digest,
            "v1_digest": v1.digest, "v1_files": sorted(got)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="launch_record/<label>/jobs.csv")
    ap.add_argument("--only", help="one job's `out` column (e.g. pool4-3way-full/passA_lam-checkpoint)")
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    assert os.environ.get("WANDB_API_KEY"), "WANDB_API_KEY (NeuralDynamics) not set"
    rows = list(csv.DictReader(open(HERE / "launch_record" / a.label / "jobs.csv")))
    outs = [r["out"] for r in rows if a.only in (None, r["out"])]
    assert outs, f"no job matches --only {a.only}"
    api = wandb.Api(timeout=120)
    results = []
    for o in outs:
        out = SCRATCH / "out" / o
        if not (out / "manifest.json").exists():
            results.append({"out": str(out), "status": "no output (job not run)"})
            continue
        results.append(backfill(api, out))
        print(results[-1]["run_id"], results[-1]["status"], flush=True)
    Path(a.results).write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
