#!/usr/bin/env python
"""Launch the GRU held-out replay: study 01's v1 D = 614 GRU, re-fine-tuned offline, per seed.

For each of the three seeds of study 01's v1 training run at D = 614 (subject_ratio 1.0, Beaker
experiment ``SOURCE_EXP``), one Beaker task mounts that seed's result dataset at ``/prior`` and runs
the wrapper's ``run_analysis finetune`` with ``heldout_offline.yaml`` (study 01's held-out config,
unchanged): 500 steps that fit only the held-out subjects' embeddings on their even sessions, then
scoring of their odd sessions. Current wrapper main also writes ``test_trial_predictions.csv``,
each scored trial's p(right), after asserting that it reproduces the final checkpoint's likelihood.
Study 15 reads those files (P1) and gates them on study 01 r2's per-subject values.

No W&B runs: the config keeps ``wandb: {}``; the outputs are the tasks' Beaker result datasets.
The W&B group ``gru-heldout-replay@<launch_id>`` is only the launch's provenance key.

Submit-only (allowed on the login node), from a clean, pushed studies checkout:

    python launch.py --label gru-replay --note "why this run exists" [--no-submit]
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
REPO = STUDY.parents[1]
sys.path.insert(0, str(REPO / "studies" / "util"))
from launch_record import write_initial  # noqa: E402

VARIANT = HERE.name
SOURCE_EXP = "01KVQ7EJ3C5YJ8FJVNJB8C8N36"   # study 01 v1 training, group mice-data-scaling-gru
SOURCE_RATIO = 1.0                          # D = 614
CONFIG = HERE / "heldout_offline.yaml"
IMAGE = "han-hou/dynamic-foraging-bfm-wrapper-main-20260902"
ENTRYPOINT = "/workspace/aind-dynamic-foraging-bfm-wrapper/beaker/entrypoint.sh"
CLUSTER = "ai1/octo-hub-onprem-h200"         # reaches the S3 snapshot cache
WORKSPACE = "ai1/aind-dynamic-foraging-foundation-model"
REPOS = {"wrapper": "aind-dynamic-foraging-bfm-wrapper", "dispatcher": "aind-dynamic-foraging-bfm-dispatcher",
         "foraging_models": "aind-dynamic-foraging-models"}
# The data the source run trained and held out on (its inputs.yaml data config).
DATA = {"source": "snapshot", "config": "mice_snapshot_scaling", "type": "mice_snapshot",
        "snapshot": "20260603", "uri": "s3://aind-scratch-data/aind-dynamic-foraging-cache/snapshots/20260603/",
        "manifest_sha256": "e5fd51da432b5f3e1dda969a5ceb415766a9f96d0c90131157be3b809cdac0d3"}
BEAKER = shutil.which("beaker") or str(Path.home() / ".local/bin/beaker")


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


def source_tasks() -> list[dict]:
    """The source experiment's finished D = 614 tasks: seed and result dataset."""
    exp = json.loads(_run(BEAKER, "experiment", "get", SOURCE_EXP, "--format", "json"))
    exp = exp[0] if isinstance(exp, list) else exp
    rows = []
    for job in exp["jobs"]:
        cmd = " ".join(str(c) for c in job.get("execution", {}).get("spec", {}).get("command") or [])
        ratio = re.search(r"subject_ratio=([0-9.eE+-]+)", cmd)
        seed = re.search(r"(?:^| )seed=([0-9]+)", cmd)
        result = (job.get("result") or {}).get("beaker") or (job.get("execution", {}).get("result") or {}).get("beaker")
        if ratio and seed and result and float(ratio.group(1)) == SOURCE_RATIO:
            assert (job.get("status") or {}).get("exitCode") == 0, f"seed {seed.group(1)} did not finish"
            rows.append({"seed": seed.group(1), "result_dataset": result})
    assert sorted(r["seed"] for r in rows) == ["0", "1", "2"], rows
    return sorted(rows, key=lambda r: r["seed"])


def build_spec(tasks: list[dict], refs: dict[str, str]) -> dict:
    config = base64.b64encode(CONFIG.read_bytes()).decode()
    inner = ("python -c \"import base64;open('/tmp/heldout.yaml','wb').write(base64.b64decode('"
             f"{config}'))\" && python -m run_analysis finetune --config /tmp/heldout.yaml")
    env = [{"name": f"{k.upper()}_REF", "value": v} for k, v in refs.items()]
    return {"version": "v2", "description": f"{VARIANT}: study 01 v1 D=614 GRU, offline held-out fine-tune per seed",
            "tasks": [{
                "name": f"{VARIANT}-s{t['seed']}",
                "image": {"beaker": IMAGE},
                "command": ["bash", ENTRYPOINT, "bash", "-lc", inner],
                "context": {"priority": "low", "preemptible": True},
                "constraints": {"cluster": [CLUSTER]},
                "resources": {"gpuCount": 1, "cpuCount": 12, "memory": "90GiB"},   # one GPU bundle
                "datasets": [{"mountPath": "/prior", "source": {"beaker": t["result_dataset"]}}],
                "envVars": [dict(e) for e in env],   # one copy per task: no YAML anchors
                "result": {"path": "/results"},
            } for t in tasks]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--note", required=True)
    ap.add_argument("--no-submit", action="store_true")
    a = ap.parse_args()
    if not a.no_submit and (HERE / "launch_record" / a.label).exists():   # write_initial refuses it, too late
        raise SystemExit(f"launch_record/{a.label}/ exists: give this launch a new --label")
    studies = _studies_sha()
    refs = {k: _main_sha(r) for k, r in REPOS.items()}
    tasks = source_tasks()
    image = json.loads(_run(BEAKER, "image", "get", IMAGE, "--format", "json"))
    image_id = (image[0] if isinstance(image, list) else image)["id"]
    launch_id = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    spec = build_spec(tasks, refs)
    record_dir = Path("/tmp") if a.no_submit else HERE / "launch_record" / a.label   # a dry run leaves the study clean
    record_dir.mkdir(parents=True, exist_ok=True)
    spec_path = record_dir / f"spec_{launch_id}.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    print(f"rendered {spec_path} ({len(tasks)} tasks); refs {refs}")
    if a.no_submit:
        return
    out = _run(BEAKER, "experiment", "create", "-w", WORKSPACE, str(spec_path))
    print(out)
    m = re.search(r"beaker\.org/ex/(\w+)|Experiment\s+(\w+)", out)
    exp_id = next(g for g in m.groups() if g)
    write_initial(record_dir / f"beaker_{launch_id}.json", platform="beaker",
                  wandb_group=f"{VARIANT}@{launch_id}", study_root=STUDY, label=a.label, note=a.note,
                  job_refs=[{"type": "beaker_experiment", "id": exp_id}],
                  revisions={"studies": studies, **refs},
                  manifests=["launch.py", "heldout_offline.yaml"], data=DATA,
                  image={"platform": "beaker", "name": IMAGE, "id": image_id},
                  command=" ".join(sys.argv), n_tasks=len(tasks), produced_by=f"variants/{VARIANT}/launch.py")
    print(f"experiment https://beaker.org/ex/{exp_id}; record in {record_dir}")


if __name__ == "__main__":
    main()
