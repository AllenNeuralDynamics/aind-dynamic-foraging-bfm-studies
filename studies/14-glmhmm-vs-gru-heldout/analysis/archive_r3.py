#!/usr/bin/env python
"""Archive r3's freeze as a W&B artifact (``make archive-r3``; once per freeze).

The r3 freeze is too big to commit (about 520 MB: 459 MB of float32 parts and 61 MB of merged
npz), and scratch is not durable (Han, 2026-11-11: archives go to W&B artifacts, as in study
40). This logs one artifact of type ``freeze`` to ``AIND-disRNN/mice_glmhmm``, the project of the
runs r3 reads, from an ``archive`` run: ``parts/``, every file of the SLURM array's parts
directory (``R3_PARTS``), and ``merged/``, the npz files ``freeze_r3.py``'s merge wrote to
``_cache_r3/``. It checks each merged file against the sha256 that ``frozen_r3.json`` records,
and writes ``archives_r3.json``: the artifact's name, version and digest (the immutable key)
and those sha256s, which ``explorer_r3.py`` checks after download.

To read it: ``wandb artifact get <artifact> --root <dir>``. Needs the W&B SDK and a key for
``AIND-disRNN``.

    R3_PARTS=<parts dir> make -C studies/14-glmhmm-vs-gru-heldout archive-r3
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

ENTITY, PROJECT = "AIND-disRNN", "mice_glmhmm"
GROUP = "r3-state-spread-archive"        # the archive run's own W&B group
DATA_GROUPS = ["hurdle@20261027-212905", "reference@20261027-213006", "refit-prior@20261101-130226"]
FROZEN = HERE / "frozen_r3.json"
CACHE = HERE / "_cache_r3"
OUT = HERE / "archives_r3.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import wandb

    parts = Path(os.environ["R3_PARTS"])
    expected = json.loads(FROZEN.read_text())["data"]["archive_files"]
    shas = {name: _sha256(CACHE / name) for name in expected}
    if shas != expected:
        raise RuntimeError(f"{CACHE} does not hold the files frozen_r3.json records; rerun the merge")
    files = sorted(p for p in parts.iterdir() if p.is_file())
    run = wandb.init(entity=ENTITY, project=PROJECT, group=GROUP, job_type="archive", name="archive-r3",
                     notes="Archive of study 14 r3's freeze: float32 parts and merged npz (scratch is not durable).")
    art = wandb.Artifact("study14-r3-freeze", type="freeze",
                         description="Study 14 r3 (state spread): the SLURM array's parts and the merge's npz files",
                         metadata={"parts_dir": str(parts), "sha256": shas})
    for p in files:
        art.add_file(str(p), name=f"parts/{p.name}")
    for name in sorted(shas):
        art.add_file(str(CACHE / name), name=f"merged/{name}")
    run.log_artifact(art)
    art.wait()
    if {name: _sha256(CACHE / name) for name in shas} != shas:
        raise RuntimeError(f"{CACHE} changed while it was archived")
    name = f"{ENTITY}/{PROJECT}/{art.name}"
    out = {"wandb_group": GROUP, "artifact": name, "digest": art.digest, "archive_run": run.url,
           "n_part_files": len(files), "part_bytes": sum(p.stat().st_size for p in files), "parts_dir": str(parts),
           "merged": {"path": "merged/", "sha256": shas}}
    run.finish()
    write_artifact(OUT, {
        "_meta": build_meta("analysis/archive_r3.py", DATA_GROUPS, study_root=STUDY, kind="freeze",
                            code=[__file__], inputs=[FROZEN],
                            sources=[{"type": "url", "id": f"wandb-artifact://{name}"},
                                     *({"type": "file", "id": str(CACHE / n), "sha256": s} for n, s in shas.items())]),
        "about": "W&B artifact holding r3's freeze (parts/ and merged/); the digest is the immutable key and "
                 "sha256 covers the merged files explorer_r3.py reads.",
        "archive": out})
    print(f"{name} ({len(files)} part files, {len(shas)} merged) digest {art.digest}", flush=True)


if __name__ == "__main__":
    main()
