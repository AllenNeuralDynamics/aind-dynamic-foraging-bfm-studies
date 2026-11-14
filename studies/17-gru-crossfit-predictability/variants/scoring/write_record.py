#!/usr/bin/env python
"""Write the launch-record v2 (hpc_launch.json) of one hand-run scoring array (score.slurm).

Run on a compute node from the studies root, after sbatch, with the dispatcher's launch tools
(``launch_inputs.image_record_hpc``) importable:
    python variants/scoring/write_record.py --label <label> --launch-id <id> --job <slurm id> \\
        --jobs <launch_record/<label>/jobs.csv> --note "..."
The record names the studies/wrapper/foraging-models SHAs, the conda env digest and the data
snapshot; the scoring runtime imports no dispatcher code (see `deviations`).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta  # noqa: E402

DISPATCHER = Path("/tmp/s17/disp-main")
WRAPPER = Path.home() / "code/wt/17-score/aind-dynamic-foraging-bfm-wrapper"
ENV = Path("/allen/aind/scratch/han.hou/miniforge3/envs/dynamic-foraging-bfm-cpu")
SCRATCH = "/allen/aind/scratch/han.hou/s17_scoring"


def sha(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("--label", "--launch-id", "--job", "--jobs", "--note"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(DISPATCHER / "code"))
    from launch_inputs import image_record_hpc
    img = image_record_hpc([Path(p) for p in glob.glob(str(ENV / "lib/python3*/site-packages"))])
    du = json.loads(Path(glob.glob(str(ENV / "lib/python3*/site-packages/aind_dynamic_foraging_models*.dist-info/direct_url.json"))[0]).read_text())
    n = sum(1 for _ in csv.DictReader(open(a.jobs)))
    group = f"scoring@{a.launch_id}"
    meta = build_meta("variants/scoring/write_record.py", [group], study_root=STUDY)
    meta["schema_version"] = 2
    jobs_rel = str(Path(a.jobs).resolve().relative_to(HERE))
    rec = {
        "_meta": meta, "kind": "initial", "platform": "hpc", "wandb_group": group, "label": a.label, "note": a.note,
        "job_refs": [{"type": "slurm_array_job", "id": a.job}],
        "revisions": {"studies": sha(STUDY), "dispatcher": sha(DISPATCHER), "wrapper": sha(WRAPPER),
                      "foraging_models": du["vcs_info"]["commit_id"]},
        "manifests": ["models.yaml", jobs_rel, "score.slurm"],
        "command": (f"sbatch --array=0-{n - 1}%39 --export=ALL,WRAPPER_ROOT={WRAPPER},S17_GROUP={group},"
                    f"S17_SCORING={SCRATCH},S17_JOBS={SCRATCH}/launch_{a.launch_id}/jobs.csv,"
                    f"S17_ROLES={SCRATCH}/launch_{a.launch_id}/roles.csv variants/scoring/score.slurm"),
        "n_tasks": n,
        "data": {"source": "snapshot", "config": "mice_snapshot_scaling", "type": "mice_snapshot", "snapshot": "20260603",
                 "uri": "s3://aind-scratch-data/aind-dynamic-foraging-cache/snapshots/20260603/",
                 "manifest_sha256": "e5fd51da432b5f3e1dda969a5ceb415766a9f96d0c90131157be3b809cdac0d3"},
        "image": img,
        "deviations": {
            "dispatcher_not_used": "The scoring runtime imports no dispatcher code (the wrapper rebuilds each model's config from its own inputs.yaml); the dispatcher SHA is the checkout whose launch_inputs.image_record_hpc wrote this record.",
            "inputs_copied": "jobs.csv and reference/roles.csv were copied to the launch_<id> scratch folder before submit, so the running jobs never read the studies checkout."},
    }
    out = Path(a.jobs).resolve().parent / "hpc_launch.json"
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print(out, rec["revisions"])


if __name__ == "__main__":
    main()
