#!/usr/bin/env python
"""Freeze the `hattori` runs (``make freeze-hattori``): the production launch into
``analysis/hattori_runs.json``, and the final set (refit + rescore, ADR 0004) into
``analysis/hattori_final_runs.json``.

The only step of r1 that reads live data. One row per run in the pinned W&B group: the held-out
subject it fitted, its pooling totals and gates from the run summary, and the sha256 of the
per-session scores file the run wrote, which is the immutable key for its numbers (W&B run ids
identify the run; the file digest pins what it produced). ``r1_hattori_pooled.py`` is offline.

Needs ``WANDB_API_KEY`` (on HPC: ``export WANDB_API_KEY=$(grep -A2 api.wandb.ai ~/.netrc | grep
password | awk '{print $2}')``) and the run artifact directory on the HPC filesystem, so it runs
on a compute node. Reads W&B over GraphQL: the SDK fails to load runs that belong to a sweep.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

ENTITY, PROJECT = "AIND-disRNN", "mice_data_scaling"
PRODUCTION = ["hattori@20261026-002131"]            # the production launch, pinned
# ADR 0004: every subject's production fit, rescored: the 103 passing the gates, then the 46 that
# were refit (the centred refit, hattori@20261027-181415, was worse on every gate and is not used).
FINAL = ["hattori@20261027-181430", "hattori@20261027-201810"]
WANDB_GROUPS = PRODUCTION + FINAL
ARTIFACT_ROOT = Path("/home/han.hou/outputs/disrnn/hb_artifacts")   # artifact_dir on HPC
OUT = HERE / "hattori_runs.json"
OUT_FINAL = HERE / "hattori_final_runs.json"
N_SUBJECTS = 149

SUMMARY_KEYS = [
    "heldout/eval_log_lik_sum", "heldout/eval_log_lik_sum_se",
    "heldout/eval_log_lik_sum_trial_marginal", "heldout/num_eval_trials",
    "heldout/eval_likelihood", "heldout/n_sessions", "heldout/n_sessions_pareto_k_fail",
    "heldout/pareto_k_threshold", "heldout/min_weight_ess", "hb/n_subjects_converged",
    "hb/max_r_hat", "hb/min_ess_bulk", "hb/divergences", "hb/fit_seconds",
]
# Written only by the final runs (wrapper importance_rescore): the all-plain totals beside the
# headline, and how many sessions were rescored.
FINAL_KEYS = [
    "heldout/eval_log_lik_sum_plain", "heldout/eval_log_lik_sum_plain_se",
    "heldout/eval_likelihood_plain", "heldout/n_sessions_importance",
    "heldout/n_sessions_pareto_k_fail_plain", "heldout/importance_pareto_k_threshold",
]

QUERY = """query($e:String!,$p:String!,$f:JSONString!,$c:String){
  project(name:$p, entityName:$e){ runs(filters:$f, first:200, after:$c){
    pageInfo{ hasNextPage endCursor }
    edges{ node{ name state updatedAt summaryMetrics config }}}}}"""


def _runs(group: str) -> list[dict]:
    """Every run in one W&B group, over GraphQL."""
    key, out, cursor = os.environ["WANDB_API_KEY"], [], None
    while True:
        body = {"query": QUERY, "variables": {"e": ENTITY, "p": PROJECT, "c": cursor,
                                              "f": json.dumps({"group": group})}}
        page = requests.post("https://api.wandb.ai/graphql", auth=("api", key), json=body,
                             timeout=120).json()["data"]["project"]["runs"]
        out += [edge["node"] for edge in page["edges"]]
        if not page["pageInfo"]["hasNextPage"]:
            return out
        cursor = page["pageInfo"]["endCursor"]


def _subject(config: dict) -> str:
    """The single held-out subject a run fitted, from its config (flat or nested keys)."""
    flat = config.get("data.test_subject_ids") or {}
    ids = flat.get("value") if isinstance(flat, dict) else flat
    if ids is None:
        ids = ((config.get("data") or {}).get("value") or {}).get("test_subject_ids")
    if isinstance(ids, str):             # the sweep passes it as the string "[790768]"
        ids = json.loads(ids)
    assert ids is not None and len(ids) == 1, f"expected one subject, got {ids!r}"
    return str(ids[0])


def _sha256(path: Path) -> str:
    """Hex digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze(groups: list[str], out: Path, keys: list[str]) -> list[dict]:
    """Write one row per run in ``groups``, plus its sources, to ``out``."""
    rows, sources = [], []
    for group in groups:
        for run in sorted(_runs(group), key=lambda r: r["name"]):
            summary = json.loads(run["summaryMetrics"] or "{}")
            scores = ARTIFACT_ROOT / run["name"] / "heldout_two_level_scores.npz"
            # A finished run must carry its immutable key; only crashed runs may lack it.
            assert run["state"] != "finished" or scores.exists(), f"{run['name']}: no {scores}"
            row = {"wandb_run_id": run["name"], "wandb_group": group, "state": run["state"],
                   "subject_id": _subject(json.loads(run["config"] or "{}")),
                   "scores_sha256": _sha256(scores) if scores.exists() else None,
                   **{key: summary.get(key) for key in keys}}
            rows.append(row)
            sources.append({"type": "wandb_run", "id": f"{ENTITY}/{PROJECT}/{run['name']}",
                            "state": run["state"], "updated_at": run["updatedAt"]})
            if row["scores_sha256"]:
                sources.append({"type": "file", "id": str(scores), "sha256": row["scores_sha256"]})
    meta = build_meta("analysis/freeze_hattori.py", groups, study_root=HERE.parent,
                      kind="freeze", code=[__file__], sources=sources)
    write_artifact(out, {"_meta": meta, "rows": rows}, outputs=[])
    print(f"froze {len(rows)} runs ({sum(r['state'] == 'finished' for r in rows)} finished) to {out}")
    return rows


def main() -> None:
    """Freeze production and the final set; the final set must cover every subject once."""
    _freeze(PRODUCTION, OUT, SUMMARY_KEYS)
    final = [r for r in _freeze(FINAL, OUT_FINAL, SUMMARY_KEYS + FINAL_KEYS)
             if r["state"] == "finished"]
    subjects = [r["subject_id"] for r in final]
    assert len(subjects) == len(set(subjects)) == N_SUBJECTS, (
        f"final set: {len(set(subjects))} subjects in {len(subjects)} finished runs, "
        f"expected {N_SUBJECTS} once each")


if __name__ == "__main__":
    main()
