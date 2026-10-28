"""Freeze Study 52's source GRUs (Study 07, D=614, H128) into ``source_runs.json``.

Reads each pinned W&B group over GraphQL (no SDK, so it also runs off-cluster), keeps the
D=614 / H128 cells, and records the run id plus the immutable ``gru-output-*`` artifact
id/digest that later extraction must load.

Cross-checks against picking up the wrong runs:
  * every run's ``data.timing_features`` matches the arm in the manifest;
  * every run trains on Study 50's model C cohort (same ``resolved_subject_ids``) and
    snapshot, so C is a valid same-mice baseline;
  * digests already in the manifest must be unchanged;
  * held-out likelihoods must equal Study 07's frozen ``timing_scaling.json`` values.

Needs WANDB_API_KEY (or api.wandb.ai in ~/.netrc).
"""

from __future__ import annotations

import json
import netrc
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

STUDY = Path(__file__).resolve().parents[1]
STUDIES = STUDY.parent
MANIFEST = STUDY / "analysis" / "source_runs.json"
STUDY07_SCALING = STUDIES / "07-gru-timing-inputs" / "analysis" / "timing_scaling.json"
STUDY12_MANIFEST = STUDIES / "50-gru-dynamics-embedding" / "analysis" / "source_runs.json"

# Pinned allowlist: model key -> (W&B group, hidden_size, subject_embedding_size, Study 07 arm).
# Never discovered live. Each group also holds the D~10..300 cells of the same launch.
WANDB_GROUPS = {
    "R": ("h128-dscan-rt-only@20260823-093058", 128, 4, "RT"),
    "L": ("h128-dscan-licks-only@20260823-093110", 128, 4, "LICK"),
}
TIMING_KEYS = ("enabled", "shuffle", "reaction_time", "lick_counts")
SNAPSHOT = "20260603"
STUDY07_TOLERANCE = 1e-6
SEEDS = {0, 1, 2}
HELDOUT_LL_KEYS = ("heldout/final/eval_likelihood", "heldout/eval_likelihood")

RUNS_QUERY = """query Runs($entity:String!,$project:String!,$filters:JSONString){
  project(name:$project,entityName:$entity){
    runs(filters:$filters,first:100){edges{node{
      name state group config summaryMetrics
      outputArtifacts(first:30){edges{node{
        id state digest fileCount size artifactSequence{name}
      }}}
    }}}
  }
}"""


def _wandb_key() -> str:
    if key := os.environ.get("WANDB_API_KEY"):
        return key
    # Explicit path: netrc.netrc() with no argument rejects a ~/.netrc owned by another uid.
    credentials = netrc.netrc(os.path.expanduser("~/.netrc")).authenticators("api.wandb.ai")
    if not credentials or not credentials[2]:
        raise RuntimeError("WANDB_API_KEY is unset and api.wandb.ai is absent from ~/.netrc")
    return credentials[2]


def _wandb_runs(entity: str, project: str, filters: dict) -> list[dict]:
    response = requests.post(
        "https://api.wandb.ai/graphql",
        auth=("api", _wandb_key()),
        json={
            "query": RUNS_QUERY,
            "variables": {"entity": entity, "project": project, "filters": json.dumps(filters)},
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(json.dumps(payload["errors"], indent=2))
    return [edge["node"] for edge in payload["data"]["project"]["runs"]["edges"]]


def _unwrapped(config: dict, key: str):
    value = config.get(key, {})
    return value.get("value", value) if isinstance(value, dict) and "value" in value else value


def _first(summary: dict, keys: tuple[str, ...], run: str) -> float:
    value = next((summary[k] for k in keys if summary.get(k) is not None), None)
    if value is None:
        raise AssertionError(f"{run} has none of {keys}")
    return float(value)


def _gru_artifact(node: dict) -> dict:
    matches = [
        edge["node"]
        for edge in node["outputArtifacts"]["edges"]
        if edge["node"]["artifactSequence"]["name"].startswith("gru-output-")
    ]
    if len(matches) != 1 or matches[0]["state"] != "COMMITTED":
        raise AssertionError(f"Expected one committed gru-output artifact for {node['name']}")
    return matches[0]


def _model_c_subjects() -> list[str]:
    """Resolved subject ids of Study 50's C-s0, read from its own W&B run."""
    manifest = json.loads(STUDY12_MANIFEST.read_text())
    run_id = manifest["models"]["C"]["runs"]["C-s0"]["run_id"]
    (node,) = _wandb_runs(manifest["entity"], manifest["project"], {"name": run_id})
    return sorted(map(str, _unwrapped(json.loads(node["config"]), "resolved_subject_ids")))


def _freeze_group(
    entity: str, project: str, model_key: str, entry: dict, cohort: list[str]
) -> dict[str, dict]:
    group, hidden_size, embedding, _ = WANDB_GROUPS[model_key]
    runs = {}
    for node in _wandb_runs(entity, project, {"group": group}):
        config = json.loads(node["config"] or "{}")
        data = _unwrapped(config, "data")
        architecture = _unwrapped(config, "model")["architecture"]
        if (
            architecture["hidden_size"] != hidden_size
            or architecture["subject_embedding_size"] != embedding
            or float(data["subject_ratio"]) != 1.0
        ):
            continue
        if node["state"] != "finished":
            raise AssertionError(f"{node['name']} is {node['state']}, not finished")
        if architecture["session_encoding_type"] != "scalar":
            raise AssertionError(f"{node['name']} session_encoding_type drifted")
        timing = {k: data["timing_features"].get(k) for k in TIMING_KEYS}
        if timing != entry["timing_features"]:
            raise AssertionError(f"{node['name']} timing_features {timing} != {entry['timing_features']}")
        if str(data["snapshot"]) != SNAPSHOT or data["ignore_policy"] != "exclude":
            raise AssertionError(f"{node['name']} snapshot/ignore_policy drifted")
        if sorted(map(str, _unwrapped(config, "resolved_subject_ids"))) != cohort:
            raise AssertionError(f"{node['name']} does not train on model C's 614 mice")
        seed = int(data["seed"])
        summary = json.loads(node["summaryMetrics"] or "{}")
        artifact = _gru_artifact(node)
        key = f"{model_key}-s{seed}"
        if key in runs:
            raise AssertionError(f"Duplicate cell {key} in {group}")
        runs[key] = {
            "seed": seed,
            "run_id": node["name"],
            "artifact_name": artifact["artifactSequence"]["name"],
            "artifact_id": artifact["id"],
            "artifact_digest": artifact["digest"],
            "source_wrapper_commit": _unwrapped(config, "wrapper_commit"),
            "heldout_likelihood": _first(summary, HELDOUT_LL_KEYS, node["name"]),
            "within_subject_eval_likelihood": float(summary["checkpoint/eval_likelihood"]),
            "checkpoint_step": int(summary["checkpoint/step"]),
        }
    if {r["seed"] for r in runs.values()} != SEEDS or len(runs) != len(SEEDS):
        raise AssertionError(f"{group} {model_key} D=614 cells: {sorted(runs)}")
    return dict(sorted(runs.items()))


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    study07 = json.loads(STUDY07_SCALING.read_text())["grid"]
    cohort = _model_c_subjects()
    if len(cohort) != 614:
        raise AssertionError(f"Model C has {len(cohort)} mice, expected 614")
    for model_key, (group, _, _, arm) in WANDB_GROUPS.items():
        entry = manifest["models"][model_key]
        if entry["group"] != group:
            raise AssertionError(f"{model_key} manifest group {entry['group']} != {group}")
        runs = _freeze_group(manifest["entity"], manifest["project"], model_key, entry, cohort)

        for key, previous in entry["runs"].items():
            if runs[key]["artifact_digest"] != previous["artifact_digest"]:
                raise AssertionError(f"{key} artifact digest changed since it was frozen")
        expected = sorted(study07[arm]["614"]["heldout"]["values"])
        found = sorted(r["heldout_likelihood"] for r in runs.values())
        if any(abs(a - b) > STUDY07_TOLERANCE for a, b in zip(expected, found)):
            raise AssertionError(f"{model_key} held-out {found} != Study 07 {expected}")

        entry["runs"] = runs
        entry["status"] = "complete"
    manifest["frozen_at_pt"] = datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(
        timespec="seconds"
    )
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {MANIFEST}")


if __name__ == "__main__":
    main()
