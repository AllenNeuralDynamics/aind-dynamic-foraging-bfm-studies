"""Freeze the adaptation-budget GRU and Q results (dispatcher #166) from their pinned launches.

Credentialed (W&B GraphQL), like ``freeze_matched_half.py``, whose helpers it reuses. One record per (dataset, K, seed) for the GRU and per (dataset, K) for
Q, keyed by W&B run id and the evaluation artifact's digest, carrying per-subject and
per-session test likelihoods. It refuses to write unless every GRU cell and every Q fit of a
dataset scored exactly the same ordered test trials: a budget changes only what adapts.

Writes ``analysis/adaptation_budget_results.json``; ``report_adaptation_budget.py`` is offline.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta  # noqa: E402
from freeze_matched_half import (ENTITY, PROJECT, _artifact,  # noqa: E402
                                 _cached_wandb_report_files, _prediction_summary,
                                 _report_metrics, _unwrapped, _wandb_runs)

# The GRU grid ran on HPC A100s (launch_record_hpc/hpc_gruk.json): the Beaker
# experiments in launch_record_k-*/half were stopped before any task started.
GRU_LAUNCH = ("gru-adaptation-budget@20261024-060709-hpc", ["27495243", "27495250", "27495483"])
N_GRU_CELLS = 30
Q_LAUNCH = ("q-adaptation-budget@20261024-035709", "27494714")   # q-adaptation-budget/launch_record
CACHE = STUDY / "analysis" / "_cache_adaptation_budget"
COHORT = STUDY / "analysis" / "adaptation_budget_cohort.json"
OUTPUT = STUDY / "analysis" / "adaptation_budget_results.json"


def _budget(target: dict) -> str:
    k = target.get("adapt_sessions_per_subject")
    return "half" if k is None else str(int(k))


def _checked(node: dict, prefix: str, root: Path) -> dict:
    """Report files of one finished run, checked against its W&B summary."""
    if node["state"] != "finished":
        raise AssertionError(f"W&B run {node['name']} is {node['state']}, not finished")
    artifact = _artifact(node, prefix)
    files = _cached_wandb_report_files(artifact, root / artifact["digest"])
    metrics_bytes, predictions_bytes = files["test_metrics.json"], files["test_trial_predictions.csv"]
    trial_digest, n_rows, per_session = _prediction_summary(predictions_bytes)
    metrics = json.loads(metrics_bytes)
    summary = json.loads(node["summaryMetrics"] or "{}")
    if abs(metrics["normalized_likelihood"] - summary["target/test/normalized_likelihood"]) >= 1e-10:
        raise AssertionError(f"W&B/file likelihood mismatch for {node['name']}")
    return {
        "wandb_run_id": node["name"],
        "wandb_url": f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/{node['name']}",
        "artifact": artifact,
        "metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
        "predictions_sha256": hashlib.sha256(predictions_bytes).hexdigest(),
        "ordered_trial_key_sha256": trial_digest,
        "n_prediction_rows": n_rows,
        "metrics": _report_metrics(metrics, per_session),
    }


def _freeze_gru() -> dict:
    group, slurm_array_job_ids = GRU_LAUNCH
    runs = _wandb_runs(group)
    if len(runs) != N_GRU_CELLS:
        raise AssertionError(f"W&B group {group} has {len(runs)} runs, expected {N_GRU_CELLS}")
    out: dict = {}
    for node in runs.values():
        config = json.loads(node["config"] or "{}")
        source, target = _unwrapped(config, "source"), _unwrapped(config, "target")
        dataset, k = target["dataset"], _budget(target)
        record = _checked(node, "external-gru-output-", CACHE / "gru" / dataset)
        record.update(source_key=source["key"], seed=source["seed"], wandb_group=group,
                      slurm_array_job_ids=slurm_array_job_ids)
        cell = out.setdefault(dataset, {}).setdefault(k, {})
        if source["key"] in cell:
            raise AssertionError(f"duplicate GRU cell {dataset}/K={k}/{source['key']}")
        cell[source["key"]] = record
    return out


def _freeze_q() -> dict:
    group, slurm_array_job_id = Q_LAUNCH
    out: dict = {}
    for node in _wandb_runs(group).values():
        target = _unwrapped(json.loads(node["config"] or "{}"), "target")
        dataset, k = target["dataset"], _budget(target)
        if k in out.get(dataset, {}):
            raise AssertionError(f"duplicate Q fit {dataset}/K={k}")
        record = _checked(node, "baseline-rl-output-", CACHE / "q" / dataset)
        record.update(wandb_group=group, slurm_array_job_id=slurm_array_job_id)
        out.setdefault(dataset, {})[k] = record
    return out


def main() -> None:
    cohort = json.loads(COHORT.read_text())
    gru, q = _freeze_gru(), _freeze_q()
    for dataset, budgets in cohort["budgets_launched"].items():
        want_gru = {str(k) for k in budgets["gru"]}
        want_q = {str(k) for k in budgets["q"]}
        if set(gru.get(dataset, {})) != want_gru or set(q.get(dataset, {})) != want_q:
            raise AssertionError(f"{dataset}: budgets GRU {sorted(gru.get(dataset, {}))} / "
                                 f"Q {sorted(q.get(dataset, {}))} differ from the launch plan")
        keys = {r["ordered_trial_key_sha256"] for cell in gru[dataset].values() for r in cell.values()}
        keys |= {r["ordered_trial_key_sha256"] for r in q[dataset].values()}
        if len(keys) != 1:
            raise AssertionError(f"{dataset}: {len(keys)} distinct ordered test-trial key sets")
    output = {
        "_meta": build_meta("analysis/freeze_adaptation_budget.py",
                            [GRU_LAUNCH[0], Q_LAUNCH[0]], study_root=STUDY),
        "contract": {
            "metric": "per-subject mean test log likelihood (nats/trial)",
            "gru": "frozen-core H=128 E4 D=614, seeds 0-2; embedding-only fine-tune, 500 steps; HPC A100",
            "q": "ForagerQLearning_L1F1_CK1_softmax (Bari2019 common Q)",
            "cohorts": str(COHORT.relative_to(STUDY)),
            "exact_ordered_trial_keys": True,
        },
        "wandb_project": f"https://wandb.ai/{ENTITY}/{PROJECT}",
        "gru": gru,
        "q": q,
    }
    OUTPUT.write_text(json.dumps(output, indent=1) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
