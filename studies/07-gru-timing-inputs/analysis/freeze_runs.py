#!/usr/bin/env python
"""Freeze study 07's W&B grid into ``analysis/frozen_runs.json`` (``make freeze``).

The only step of this study that reads W&B (GraphQL; needs ``WANDB_API_KEY``). It keeps
finished runs of the pinned ``WANDB_GROUPS`` and writes one row per run, keyed by its W&B
run id; ``timing_scaling.py`` builds every report artifact offline from that file.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STUDY.parent / "util"))
import wandb_keys as K                             # noqa: E402
from _meta import build_meta                       # noqa: E402
from timing_scaling import DS, ENTITY, FROZEN_RUNS, PROJECT, WANDB_GROUPS  # noqa: E402


def _gql(query, variables):
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        raise SystemExit("WANDB_API_KEY not set (or put wandb creds in ~/.netrc).")
    req = urllib.request.Request(
        "https://api.wandb.ai/graphql",
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Basic " + base64.b64encode(f"api:{key}".encode()).decode()},
    )
    return json.load(urllib.request.urlopen(req))


_RUNS_Q = """query R($e:String!,$p:String!,$c:String){project(name:$p,entityName:$e){
 runs(first:200, after:$c){pageInfo{hasNextPage endCursor}
 edges{node{name group state config summaryMetrics}}}}}"""


def _cfg_get(cfg, *path):
    """Read a possibly-W&B-wrapped ({'value': ...}) nested config field."""
    cur = cfg
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
        if isinstance(cur, dict) and "value" in cur and k != "value":
            cur = cur["value"]
    return cur


def _classify_arm(tf):
    if not tf or not tf.get("enabled"):
        return "OFF"
    if tf.get("shuffle"):
        return "SHUF"
    rt = tf.get("reaction_time", True)
    lk = tf.get("lick_counts", True)
    if rt and lk:
        return "ON"
    return "RT" if rt else "LICK"


def _nearest_D(n):
    return min(DS, key=lambda t: abs(t - n))


def fetch_runs():
    """Finished runs of the pinned groups, classified into (arm, D), in pull order."""
    rows, cursor = [], None
    while True:
        page = _gql(_RUNS_Q, {"e": ENTITY, "p": PROJECT, "c": cursor})["data"]["project"]["runs"]
        for edge in page["edges"]:
            n = edge["node"]
            if n["state"] != "finished" or n.get("group") not in WANDB_GROUPS:
                continue
            cfg = json.loads(n["config"] or "{}")
            sm = json.loads(n["summaryMetrics"] or "{}")
            tf = _cfg_get(cfg, "data", "timing_features") or {}
            subjects = _cfg_get(cfg, "resolved_subject_ids") or []
            D = len(subjects)
            if D == 0:
                continue
            heldout = sm.get(K.HELDOUT) or sm.get(K.HELDOUT_FALLBACK)
            rows.append({
                "wandb_run_id": n["name"], "run": n["name"][-8:], "group": n.get("group") or "",
                "arm": _classify_arm(tf), "D": D, "Db": _nearest_D(D),
                "heldout": heldout,
                "within": sm.get(K.WITHIN), "within_train": sm.get(K.WITHIN_TRAIN),
                "sel_step": sm.get(K.SELECTED_STEP), "final_step": sm.get(K.FINAL_STEP),
            })
        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]
    return rows


def main():
    rows = fetch_runs()
    groups = sorted({r["group"] for r in rows if r["group"]})
    out = {"_meta": build_meta("analysis/freeze_runs.py", groups, study_root=STUDY), "runs": rows}
    FROZEN_RUNS.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {FROZEN_RUNS} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
