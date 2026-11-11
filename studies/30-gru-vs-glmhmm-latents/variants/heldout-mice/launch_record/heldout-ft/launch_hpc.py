#!/usr/bin/env python
"""Launch the held-out fine-tune of study 02's H256 / D614 / seed 0 3-way GRU on the Allen HPC.

One slurm job (``run_hpc.sbatch``): assemble the source run's model_dir from its W&B artifact, then
run the wrapper's offline fine-tune with ``heldout_offline.yaml``. The wrapper and foraging-models
checkouts must be clean; their HEADs are recorded. Submit-only:

    python launch_hpc.py --label heldout-ft --note "..." --wrapper <checkout> --models <checkout>
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
REPO = STUDY.parents[1]
sys.path.insert(0, str(REPO / "studies" / "util"))
from launch_record import conda_image, write_initial  # noqa: E402

VARIANT = HERE.name
CPU_ENV = "/allen/aind/scratch/han.hou/miniforge3/envs/dynamic-foraging-bfm-cpu"
WANDB_ENV = "/allen/aind/scratch/han.hou/miniforge3/envs/mevolve-search"   # artifact reads only
OUT_BASE = Path("/allen/aind/scratch/han.hou/outputs/study30/heldout-mice")
LOGS = Path("/allen/aind/scratch/han.hou/tmp/s30/logs")
# The data the source run trained and held out on (its inputs.yaml data config; study 02's sweep
# uses data=mice_snapshot_scaling). Manifest digest as recorded by study 15 for the same snapshot.
DATA = {"source": "snapshot", "config": "mice_snapshot_scaling", "type": "mice_snapshot",
        "snapshot": "20260603", "uri": "s3://aind-scratch-data/aind-dynamic-foraging-cache/snapshots/20260603/",
        "manifest_sha256": "e5fd51da432b5f3e1dda969a5ceb415766a9f96d0c90131157be3b809cdac0d3"}


def _run(*cmd: str) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"{' '.join(cmd)} failed:\n{out.stderr}")
    return out.stdout


def _main_sha(repo: str) -> str:
    sha = _run("git", "ls-remote", f"https://github.com/AllenNeuralDynamics/{repo}.git", "refs/heads/main").split()[0]
    assert re.fullmatch(r"[0-9a-f]{40}", sha), sha
    return sha


def _studies_sha() -> str:
    """HEAD of this checkout, refusing uncommitted study files or a commit not on origin."""
    dirty = _run("git", "-C", str(REPO), "status", "--porcelain", "--", str(STUDY)).strip()
    if dirty:
        sys.exit(f"uncommitted changes under {STUDY}:\n{dirty}")
    sha = _run("git", "-C", str(REPO), "rev-parse", "HEAD").strip()
    if not _run("git", "-C", str(REPO), "branch", "-r", "--contains", sha).strip():
        sys.exit(f"studies commit {sha} is not pushed")
    return sha


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
    a = ap.parse_args()
    if len(a.note.strip()) < 10:   # write_initial's rule, checked before anything is submitted
        raise SystemExit("--note must say why this launch exists (at least 10 characters)")
    a.wrapper, a.models = a.wrapper.resolve(), a.models.resolve()
    record_dir = HERE / "launch_record" / a.label
    if record_dir.exists():   # write_initial refuses a used label, after submission
        raise SystemExit(f"launch_record/{a.label}/ exists: give this launch a new --label")
    # The job does not read the dispatcher; the schema asks for one, so this is its main.
    revisions = {"studies": _studies_sha(), "dispatcher": _main_sha("aind-dynamic-foraging-bfm-dispatcher"),
                 "wrapper": _clean_sha(a.wrapper, "code"), "foraging_models": _clean_sha(a.models, "src")}
    launch_id = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    out_root = OUT_BASE / launch_id
    out_root.mkdir(parents=True)
    config = out_root / "heldout_offline.yaml"   # the job reads a copy fixed at submission
    shutil.copyfile(HERE / "heldout_offline.yaml", config)
    LOGS.mkdir(parents=True, exist_ok=True)
    env = (f"ALL,WRAPPER={a.wrapper},MODELS={a.models},CONFIG={config},OUT_ROOT={out_root},"
           f"CPU_ENV={CPU_ENV},WANDB_ENV={WANDB_ENV}")
    command = ["sbatch", "--parsable", f"--export={env}", "-J", "s30-heldout-ft",
               "-o", str(LOGS / "heldout_ft_%j.log"), str(HERE / "run_hpc.sbatch")]
    job = _run(*command).strip().split(";")[0]
    assert re.fullmatch(r"\d+", job), job
    write_initial(record_dir / f"hpc_{launch_id}.json", platform="hpc",
                  wandb_group=f"{VARIANT}@{launch_id}", study_root=STUDY, label=a.label, note=a.note,
                  job_refs=[{"type": "slurm_job", "id": job}], revisions=revisions,
                  manifests=["run_hpc.sbatch", "heldout_offline.yaml", "launch_hpc.py"], data=DATA,
                  image=conda_image(CPU_ENV), command=" ".join(command), n_tasks=1,
                  produced_by=f"variants/{VARIANT}/launch_hpc.py")
    print(f"slurm job {job}; outputs {out_root}; record in launch_record/{a.label}/")


if __name__ == "__main__":
    main()
