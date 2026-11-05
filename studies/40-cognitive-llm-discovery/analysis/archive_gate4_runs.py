#!/usr/bin/env python
"""Archive gate 4's two search runs as W&B artifacts (``make archive-gate4``; once per run).

Scratch is not durable, and a search's database is too big to commit once runs grow (Han,
2026-11-04: archives go to W&B artifacts). For each run this logs one artifact of type
``mevolve-run`` to ``NeuralDynamics/dfbfm-mevolve``, from an ``archive`` run in the launch's own
W&B group: the whole ``search/`` directory (Shinka's database with its ``-wal``, ledger, config,
summary, logs and one folder per proposal) and ``heldout/``, without ``search/wandb/`` and the
database's ``-shm``. It writes ``analysis/gate4/archives.json``: each artifact's name, version and
digest (the immutable key) and the sha256 of the files the freezes read, which they check.

To read a run after scratch is purged: ``wandb artifact get <artifact> --root <dir>``, then point
the freezes' ``RUNS`` at that directory. Needs the W&B SDK and the key in ``~/.config/mevolve/``.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

ENTITY, PROJECT = "NeuralDynamics", "dfbfm-mevolve"
RUNS = Path("/allen/aind/scratch/han.hou/mevolve/runs")
ARMS = {"seeded": ("gate4-recovery", "20261104-132600"), "noseed": ("gate4-noseed", "20261104-141503")}
KEY_FILES = ("search/programs.sqlite", "search/programs.sqlite-wal", "search/ledger.jsonl", "search/run_config.json",
             "search/run_summary.json", "heldout/heldout.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import wandb

    out, sources = {}, []
    for arm, (variant, launch) in ARMS.items():
        d = RUNS / variant / launch
        files = sorted(p for p in d.rglob("*") if p.is_file() and p.relative_to(d).parts[0] in ("search", "heldout")
                       and "wandb" not in p.relative_to(d).parts and not p.name.endswith("-shm"))
        shas = {k: _sha256(d / k) for k in KEY_FILES}
        group = f"{variant}@{launch}"
        run = wandb.init(entity=ENTITY, project=PROJECT, group=group, job_type="archive", name=f"archive-{variant}",
                         notes=f"Archive of {group}'s run directory (scratch is not durable).")
        art = wandb.Artifact(f"run-{variant}-{launch}", type="mevolve-run",
                             description=f"{group}: search/ and heldout/ of the run directory",
                             metadata={"wandb_group": group, "sha256": shas})
        for p in files:
            art.add_file(str(p), name=str(p.relative_to(d)))
        run.log_artifact(art)
        art.wait()
        assert {k: _sha256(d / k) for k in KEY_FILES} == shas, f"{d} changed while it was archived"
        out[arm] = {"wandb_group": group, "artifact": f"{ENTITY}/{PROJECT}/{art.name}", "digest": art.digest,
                    "n_files": len(files), "bytes": sum(p.stat().st_size for p in files), "run_dir": str(d),
                    "archive_run": run.url, "sha256": shas}
        run.finish()
        sources += [{"type": "url", "id": f"wandb-artifact://{ENTITY}/{PROJECT}/{art.name}"},
                    *({"type": "file", "id": str(d / k), "sha256": s} for k, s in shas.items())]
        print(f"{arm}: {out[arm]['artifact']} ({len(files)} files) digest {art.digest}", flush=True)
    write_artifact(HERE / "gate4" / "archives.json", {
        "_meta": build_meta("analysis/archive_gate4_runs.py", [o["wandb_group"] for o in out.values()],
                            study_root=STUDY, kind="freeze", code=[__file__], sources=sources),
        "about": "W&B artifacts holding gate 4's two run directories; the digest is the immutable key and "
                 "sha256 covers the files the freezes read.", "archives": out})


if __name__ == "__main__":
    main()
