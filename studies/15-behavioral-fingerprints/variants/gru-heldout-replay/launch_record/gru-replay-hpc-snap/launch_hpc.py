#!/usr/bin/env python
"""Launch the GRU held-out replay on the Allen HPC as well (the Beaker hubs were full).

Same job as ``launch.py``: study 01's offline held-out fine-tune of the v1 D=614 GRU, one array task
per seed (``run_hpc.sbatch``), from the three source result datasets fetched to ``--prior-root``
(``beaker dataset fetch <id> --prefix run/``), with the wrapper and foraging-models checked out at
the same SHAs as the Beaker launch. Submit-only (allowed on the login node):

    python launch_hpc.py --label gru-replay-hpc --note "..." --wrapper <checkout at the SHA> \\
        --models <foraging-models checkout> --prior-root <dir with s0, s1, s2>
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from launch import DATA, REPOS, STUDY, VARIANT, _main_sha, _run, _studies_sha  # noqa: E402
from launch_record import write_initial  # noqa: E402

CONDA_ENV = "/allen/aind/scratch/han.hou/miniforge3/envs/dynamic-foraging-bfm-cpu"
OUT_BASE = Path("/allen/aind/scratch/han.hou/outputs/study15/gru-heldout-replay/hpc")
LOGS = Path("/allen/aind/scratch/han.hou/tmp/s15/logs")


def _clean_sha(repo: Path, src: str) -> str:
    if _run("git", "-C", str(repo), "status", "--porcelain", "--", src).strip():
        sys.exit(f"{repo} has uncommitted changes under {src}")
    return _run("git", "-C", str(repo), "rev-parse", "HEAD").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--note", required=True)
    ap.add_argument("--wrapper", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--prior-root", type=Path, required=True)
    ap.add_argument("--job", help="record this already-submitted array instead of submitting")
    ap.add_argument("--launch-id", help="with --job: the launch id its outputs were written under")
    ap.add_argument("--studies-sha", help="with --job: the studies commit the job was submitted from")
    a = ap.parse_args()
    studies = _studies_sha()
    if a.job:   # recorded after submission: name the commit it ran from; manifests come from git
        assert a.launch_id and a.studies_sha, "--job needs --launch-id and --studies-sha"
        studies = a.studies_sha
    # The job does not read the dispatcher; the schema asks for one, so this is the Beaker twin's pin.
    revisions = {"studies": studies, "dispatcher": _main_sha(REPOS["dispatcher"]),
                 "wrapper": _clean_sha(a.wrapper, "code"), "foraging_models": _clean_sha(a.models, "src")}
    for seed in range(3):
        assert (a.prior_root / f"s{seed}" / "run" / "outputs" / "checkpoints" / "index.json").exists(), seed
    launch_id = a.launch_id or datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    out_root = OUT_BASE / launch_id
    env = (f"ALL,WRAPPER={a.wrapper},MODELS={a.models},CONFIG={HERE / 'heldout_offline.yaml'},"
           f"PRIOR_ROOT={a.prior_root},OUT_ROOT={out_root},PATH={CONDA_ENV}/bin:/usr/bin:/bin")
    command = ["sbatch", "--parsable", f"--export={env}", "-J", "s15-gru-replay",
               "-o", str(LOGS / "gru_replay_%A_%a.log"), str(HERE / "run_hpc.sbatch")]
    job = a.job or _run(*command).strip().split(";")[0]
    assert re.fullmatch(r"\d+", job), job
    packages = _run(f"{CONDA_ENV}/bin/python", "-m", "pip", "list", "--format=freeze")
    write_initial(HERE / "launch_record" / a.label / f"hpc_{launch_id}.json", platform="hpc",
                  wandb_group=f"{VARIANT}@{launch_id}", study_root=STUDY, label=a.label, note=a.note,
                  job_refs=[{"type": "slurm_array_job", "id": job}], revisions=revisions,
                  manifests=["run_hpc.sbatch", "heldout_offline.yaml", "launch_hpc.py"], data=DATA,
                  image={"platform": "hpc", "conda_env": CONDA_ENV,
                         "lock_sha256": hashlib.sha256(packages.encode()).hexdigest()},
                  command=" ".join(command), n_tasks=3, produced_by=f"variants/{VARIANT}/launch_hpc.py",
                  recorded_after_launch=bool(a.job))
    print(f"slurm array {job}; outputs {out_root}; record in launch_record/{a.label}/")


if __name__ == "__main__":
    main()
