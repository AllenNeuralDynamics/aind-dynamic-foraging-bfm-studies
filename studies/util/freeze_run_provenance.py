"""Freeze what past launches trained on and ran in, recovered from W&B and Beaker (#28).

Launches before #25 recorded neither their data snapshot nor their container image. Each W&B
run's config still holds ``data.snapshot`` and the code commits (``wrapper_commit``,
``dispatcher_commit``, ``foraging_models_commit``), and the Beaker job it ran in
(``BEAKER_JOB_ID``) names the immutable image id. This freeze reads them for the given W&B
groups and writes ``<study>/analysis/provenance/run_provenance.json``:

    python studies/util/freeze_run_provenance.py --study studies/06-disrnn-operating-point-at-scale \\
        --project AIND-disRNN/disrnn_data_scaling --group mult-d-grid@20260718-151409

A group is looked up in each ``--project`` in turn. Needs W&B credentials and BEAKER_TOKEN, so
it runs on a compute node from the study's ``freeze-provenance`` target, never in CI. Every
group must have a launch record (check_provenance: artifact-group-unrecorded); record a
launch first, then recover its provenance here.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _meta import build_meta, write_artifact  # noqa: E402

SNAPSHOT_PREFIX = "s3://aind-scratch-data/aind-dynamic-foraging-cache"
SNAPSHOT_HTTPS = "https://aind-scratch-data.s3.amazonaws.com/aind-dynamic-foraging-cache"
COMMITS = {"wrapper": "wrapper_commit", "dispatcher": "dispatcher_commit", "foraging_models": "foraging_models_commit"}


def manifest(snapshot: str) -> tuple[str | None, dict, str]:
    """sha256, body and url of a snapshot's snapshot_manifest.json."""
    url = f"{SNAPSHOT_HTTPS}/snapshots/{snapshot}/snapshot_manifest.json"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            body = resp.read()
        return hashlib.sha256(body).hexdigest(), json.loads(body), url
    except (OSError, ValueError) as exc:
        print(f"  WARNING: {url}: {exc}")
        return None, {}, url


def _values(v) -> list[str]:
    return [str(x) for x in (v if isinstance(v, list) else [v]) if x]


def group_provenance(runs: list, project: str, image_of) -> tuple[dict, list[dict]]:
    """The recovered record for one group's runs, and the sources it read."""
    data, revisions, images, sources = Counter(), defaultdict(Counter), Counter(), []
    unrecovered = Counter()
    for r in runs:
        sources.append({"type": "wandb_run", "id": f"{project}/{r.id}", "state": r.state})
        cfg = r.config or {}
        d = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}
        # an empty data config (e.g. a re-scoring job) says nothing about the data: not "live"
        data[(d.get("type"), str(d["snapshot"]) if d.get("snapshot") else None, bool(d))] += 1
        for repo, key in COMMITS.items():
            for sha in _values(cfg.get(key)):
                revisions[repo][sha] += 1
            if not cfg.get(key):
                unrecovered[f"{repo} commit"] += 1
        job = cfg.get("BEAKER_JOB_ID")
        if job:
            images[image_of(job)] += 1
            sources.append({"type": "url", "id": f"https://beaker.org/job/{job}"})
        else:
            unrecovered["image (no Beaker job: an HPC or local run)"] += 1
    out_data = []
    for (dtype, snap, logged), n in sorted(data.items(), key=lambda kv: -kv[1]):
        if snap:
            digest, m, url = manifest(snap)
            sources.append({"type": "url", "id": url, **({"sha256": digest} if digest else {})})
            out_data.append({"source": "snapshot", "type": dtype, "snapshot": snap,
                             "uri": f"{SNAPSHOT_PREFIX}/snapshots/{snap}/", "manifest_sha256": digest,
                             "build_id": m.get("latest_build_id"), "n_sessions": m.get("n_sessions"), "n_runs": n})
        else:
            # a database config without a snapshot read the live database (before snapshots existed)
            source = ("unrecorded" if not logged else "generated" if str(dtype or "").startswith("synthetic")
                      else "live")
            out_data.append({"source": source, "type": dtype, "snapshot": None, "n_runs": n})
    return {
        "project": project, "n_runs": len(runs),
        "data": out_data,
        "revisions": {repo: dict(c.most_common()) for repo, c in revisions.items()},
        "image": [{"platform": "beaker", "id": i, "n_runs": n} for i, n in images.most_common() if i]
                 + ([{"platform": "beaker", "id": None, "n_runs": images[None]}] if images.get(None) else []),
        "unrecovered": dict(unrecovered),
    }, sources


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--study", required=True, type=Path)
    p.add_argument("--project", action="append", required=True, help="entity/project to look groups up in")
    p.add_argument("--group", action="append", required=True)
    a = p.parse_args(argv)

    import wandb
    api = wandb.Api(timeout=120)
    beaker = None
    if os.environ.get("BEAKER_TOKEN"):
        from beaker import Beaker, Config
        beaker = Beaker(Config(user_token=os.environ["BEAKER_TOKEN"], default_org="ai1"))
    cache: dict[str, str | None] = {}

    def image_of(job: str) -> str | None:
        """The image id a Beaker job ran (its spec names the id it resolved at submit)."""
        if job not in cache:
            try:
                cache[job] = beaker.job.get(job).execution.spec.image.beaker if beaker else None
            except Exception as exc:                 # a purged job: the image is not recoverable
                print(f"  WARNING: Beaker job {job}: {exc}")
                cache[job] = None
        return cache[job]

    groups, sources = {}, []
    for g in a.group:
        for project in a.project:
            runs = list(api.runs(project, filters={"group": g}))
            if runs:
                break
        else:
            sys.exit(f"freeze_run_provenance: no runs in group {g} in {a.project}")
        print(f"{g}: {len(runs)} runs in {project}")
        groups[g], src = group_provenance(runs, project, image_of)
        sources += src
    seen, uniq = set(), []
    for s in sources:
        if s["id"] not in seen:
            seen.add(s["id"])
            uniq.append(s)
    study = a.study.resolve()
    out = study / "analysis" / "provenance" / "run_provenance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = build_meta("../util/freeze_run_provenance.py", list(groups), study_root=study, kind="freeze",
                      code=[__file__], inputs=[], sources=uniq)
    write_artifact(out, {"_meta": meta,
                         "recovered_at_pt": datetime.datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"),
                         "note": "Recovered after the fact from W&B run configs and Beaker jobs (#28); launches "
                                 "from #25 on record data and image in their launch records instead.",
                         "groups": groups})
    print(f"wrote {out.relative_to(study.parents[1])}")


if __name__ == "__main__":
    main()
