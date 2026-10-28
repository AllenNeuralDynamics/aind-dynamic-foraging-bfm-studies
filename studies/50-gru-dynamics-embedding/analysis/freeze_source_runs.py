"""Freeze Study 50's source GRUs (D=614; C = H128/E4, B = H256/E4) into ``source_runs.json``.

Reads each pinned W&B group over GraphQL (no SDK, so it also runs off-cluster), keeps the
D=614 cells at the expected hidden size, and records the run id plus the immutable
``gru-output-*`` artifact id/digest that later extraction must load.

Two cross-checks guard against picking up the wrong runs:
  * Digests already in the manifest (C, seeded from Study 09) must be unchanged.
  * B's held-out likelihoods must equal Study 01's frozen ``nxd_scaling.json`` seed scalars.

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
STUDY01_NXD = STUDIES / "01-gru-scaling-law" / "analysis" / "nxd_scaling.json"

# Pinned allowlist: model key -> (W&B group, hidden_size, subject_embedding_size).
# Never discovered live. C's group also holds E8 runs, hence the embedding filter.
WANDB_GROUPS = {
    "C": ("e4-e8-d614-source@20260906-195409", 128, 4),
    "B": ("nxd-grid@20260623-102649", 256, 4),
}
STUDY01_CHECKED = {"B": "N256_D614"}
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


def _wandb_runs(entity: str, project: str, group: str) -> list[dict]:
    response = requests.post(
        "https://api.wandb.ai/graphql",
        auth=("api", _wandb_key()),
        json={
            "query": RUNS_QUERY,
            "variables": {
                "entity": entity,
                "project": project,
                "filters": json.dumps({"group": group}),
            },
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


def _gru_artifact(node: dict) -> dict:
    matches = [
        edge["node"]
        for edge in node["outputArtifacts"]["edges"]
        if edge["node"]["artifactSequence"]["name"].startswith("gru-output-")
    ]
    if len(matches) != 1 or matches[0]["state"] != "COMMITTED":
        raise AssertionError(f"Expected one committed gru-output artifact for {node['name']}")
    return matches[0]


def _freeze_group(
    entity: str, project: str, model_key: str, group: str, hidden_size: int, embedding: int
) -> dict[str, dict]:
    runs = {}
    for node in _wandb_runs(entity, project, group):
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
        seed = int(_unwrapped(config, "seed") if "seed" in config else data["seed"])
        summary = json.loads(node["summaryMetrics"] or "{}")
        heldout = next((summary[k] for k in HELDOUT_LL_KEYS if summary.get(k) is not None), None)
        if heldout is None:
            raise AssertionError(f"{node['name']} has no held-out likelihood")
        artifact = _gru_artifact(node)
        key = f"{model_key}-s{seed}"
        if key in runs:
            raise AssertionError(f"Duplicate cell {key} in {group}")
        runs[key] = {
            "seed": seed,
            "run_id": node["name"],
            "artifact_name": f"{artifact['artifactSequence']['name']}",
            "artifact_id": artifact["id"],
            "artifact_digest": artifact["digest"],
            "heldout_likelihood": float(heldout),
        }
    if {r["seed"] for r in runs.values()} != SEEDS or len(runs) != len(SEEDS):
        raise AssertionError(f"{group} {model_key} D=614 cells: {sorted(runs)}")
    return dict(sorted(runs.items()))


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    study01 = json.loads(STUDY01_NXD.read_text())["seed_scalars"]
    for model_key, (group, hidden_size, embedding) in WANDB_GROUPS.items():
        entry = manifest["models"][model_key]
        if entry["group"] != group:
            raise AssertionError(f"{model_key} manifest group {entry['group']} != {group}")
        runs = _freeze_group(
            manifest["entity"], manifest["project"], model_key, group, hidden_size, embedding
        )

        for key, previous in entry["runs"].items():
            if runs[key]["artifact_digest"] != previous["artifact_digest"]:
                raise AssertionError(f"{key} artifact digest changed since it was frozen")
        if model_key in STUDY01_CHECKED:
            expected = sorted(study01[STUDY01_CHECKED[model_key]])
            found = sorted(r["heldout_likelihood"] for r in runs.values())
            if any(abs(a - b) > 1e-9 for a, b in zip(expected, found)):
                raise AssertionError(f"{model_key} likelihoods {found} != Study 01 {expected}")

        entry["runs"] = runs
        entry["status"] = "complete"
    manifest["frozen_at_pt"] = datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(
        timespec="seconds"
    )
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {MANIFEST}")


if __name__ == "__main__":
    main()
