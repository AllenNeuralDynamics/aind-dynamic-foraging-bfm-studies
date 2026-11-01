#!/usr/bin/env python
"""Freeze the per-subject literature-model fits (``make freeze-fits``).

The only step of this study that reads live data. For each run in the pinned W&B group (one
run per literature model), it reads the two per-subject tables the trainer wrote to the run
directory on HPC (``outputs/subject_fit_metrics.csv`` for the training pool and
``outputs/heldout_test/subject_fit_metrics.csv`` for the held-out subjects) and writes one row
per (literature model, subject) to ``analysis/per_subject_fits.csv``. The sha256 of each source
table is the immutable key for its numbers; ``analysis/per_subject_fits.json`` records them with
the run ids. Needs ``WANDB_API_KEY`` (the NeuralDynamics key) and ``/allen``, so it runs on a
compute node. Every report producer reads only the two committed files.

    python analysis/freeze_fits.py [--group <group>] [--out-dir <dir>]

``--group`` and ``--out-dir`` exist to test the freeze on a smoke group without touching the
committed files; ``make freeze-fits`` uses neither.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

ENTITY, PROJECT = "NeuralDynamics", "aind-dynamic-foraging-model"
WANDB_GROUPS = ["per-subject-de@20261101-012148"]   # the production launch, pinned
RUN_ROOT = Path("/allen/aind/scratch/han.hou/outputs/disrnn/wandb")   # BFM_OUTPUT_DIR/wandb on HPC
TABLES = {"train": "outputs/subject_fit_metrics.csv",
          "heldout": "outputs/heldout_test/subject_fit_metrics.csv"}
N_SUBJECTS = {"train": 614, "heldout": 149}
# The swept dispatcher config (W&B config `model`) -> (literature model, agent class). The sweep
# parameter is named `model`, so W&B keeps the config name there, not the composed block; and the
# run directory's .hydra/ and inputs.yaml are unreliable copies (smoke-2). The agent class each
# per-subject table records is checked against this map.
CONFIGS = {
    "baseline_rl_lit_hattori2019": ("Hattori2019", "ForagerQLearning"),
    "baseline_rl_lit_bari2019": ("Bari2019", "ForagerQLearning"),
    "baseline_rl_lit_rescorla_wagner": ("Rescorla-Wagner", "ForagerQLearning"),
    "baseline_rl_lit_win_stay_lose_shift": ("Win-Stay-Lose-Shift", "ForagerLossCounting"),
    "baseline_rl_lit_compare_to_threshold": ("CompareToThreshold", "ForagerCompareThreshold"),
    "baseline_rl_lit_actor_critic": ("ActorCritic", "ForagerActorCritic"),
    "baseline_rl_lit_chen_rlck": ("Chen-RLCK", "ForagerRLCK"),
    "baseline_rl_lit_zid_history_kernel2": ("Zid-HistoryKernel2-ForagingRL", "ForagerZidHistoryKernel"),
    "baseline_rl_lit_grossman_meta_learning": ("Grossman-MetaLearning", "ForagerGrossmanMetaLearning"),
    "baseline_rl_lit_lebedeva_pr": ("Lebedeva-PR", "ForagerLebedevaPR"),
    "baseline_rl_lit_beron_rflr": ("Beron-RFLR", "ForagerBeronRFLR"),
    "baseline_rl_lit_miller_rhg": ("Miller-RHG", "ForagerMillerRHG"),
    "baseline_rl_lit_eckstein_rl": ("Eckstein-RL", "ForagerEcksteinRL"),
    "baseline_rl_lit_eckstein_bi": ("Eckstein-BI", "ForagerEcksteinBI"),
    "baseline_rl_lit_feedback_dependent_rl": ("FeedbackDependent-RL", "ForagerFeedbackDependentRL"),
    "baseline_rl_lit_feedback_dependent_rl_bias": ("FeedbackDependent-RL+Bias", "ForagerFeedbackDependentRLBias"),
    "baseline_rl_lit_feedback_dependent_rl_bias_ck1": ("FeedbackDependent-RL+Bias+CK1", "ForagerFeedbackDependentRLBiasCK1"),
    "baseline_rl_lit_alsio_dual_rate_sticky_rl": ("Alsio-DualRateSticky-RL", "ForagerAlsioRL"),
    "baseline_rl_lit_lopez_double_trace": ("Lopez-DoubleTrace", "ForagerLopezDoubleTrace"),
}
# Study 01's GRU split (training pool / test list). Its 763 subjects are ours, but 3 sit in the
# other pool each way (study 13 r2); per-subject fits do not depend on the pool, so the freeze
# checks the union and records the swap.
GRU_SPLIT = HERE.parents[1] / "10-gru-reservoir-ablation" / "reference" / "study01-v2-exact-split.json"
# Columns every table has; any other column after n_free_params is a fitted parameter.
FIXED = ["agent_class", "agent_kwargs", "subject_id", "subject_index", "curriculum_name",
         "num_train_sessions", "num_eval_sessions", "num_train_trials", "num_eval_trials",
         "train_likelihood", "eval_likelihood", "train_total_log_likelihood", "train_total_trials",
         "eval_total_log_likelihood", "eval_total_trials", "log_likelihood_train", "LPT_train",
         "AIC", "BIC", "n_free_params"]
COLUMNS = ["literature_model", "wandb_run_id", "cohort", "subject_id", "curriculum_name",
           "eval_total_log_likelihood", "eval_total_trials", "eval_likelihood",
           "train_total_log_likelihood", "train_total_trials", "n_free_params", "fitted_params"]
QUERY = """query($e:String!,$p:String!,$f:JSONString!,$c:String){
  project(name:$p,entityName:$e){runs(filters:$f,first:50,after:$c){
    edges{node{name state config updatedAt}} pageInfo{hasNextPage endCursor}}}}"""


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


def _value(config: dict, key: str):
    """A run-config value; W&B stores each top-level key as ``{"value": ...}``."""
    node = config[key]
    return node["value"] if isinstance(node, dict) and "value" in node else node


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(model: str, agent_class: str, run_id: str, cohort: str, table: Path) -> list[dict]:
    """One frozen row per subject in one per-subject table."""
    with table.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    classes = {rec["agent_class"] for rec in records}
    assert classes == {agent_class}, f"{run_id} {cohort}: table holds {classes}, expected {agent_class}"
    params = [c for c in records[0] if c not in FIXED]
    rows = []
    for rec in records:
        assert rec["eval_total_trials"] and float(rec["eval_total_trials"]) > 0, (run_id, rec["subject_id"])
        rows.append({"literature_model": model, "wandb_run_id": run_id, "cohort": cohort,
                     "subject_id": rec["subject_id"],
                     "curriculum_name": "" if rec["curriculum_name"] in ("", "None") else rec["curriculum_name"],
                     **{k: rec[k] for k in COLUMNS[5:11]},
                     "fitted_params": json.dumps({p: float(rec[p]) for p in params}, sort_keys=True)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", help="freeze this group instead of WANDB_GROUPS (smoke test)")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    groups = [args.group] if args.group else WANDB_GROUPS
    test = bool(args.group)

    rows, runs, sources = [], [], []
    for group in groups:
        for run in sorted(_runs(group), key=lambda r: r["name"]):
            if run["state"] != "finished":
                print(f"skip {run['name']}: {run['state']}")
                continue
            config_name = _value(json.loads(run["config"] or "{}"), "model")
            model, agent_class = CONFIGS[config_name]
            run_dir, = RUN_ROOT.glob(f"run-*-{run['name']}")
            entry = {"wandb_run_id": run["name"], "wandb_group": group, "model_config": config_name,
                     "literature_model": model, "agent_class": agent_class, "tables": {}}
            sources.append({"type": "wandb_run", "id": f"{ENTITY}/{PROJECT}/{run['name']}",
                            "state": run["state"], "updated_at": run["updatedAt"]})
            for cohort, rel in TABLES.items():
                table = run_dir / "files" / rel
                digest = _sha256(table)
                entry["tables"][cohort] = {"path": str(table), "sha256": digest}
                sources.append({"type": "file", "id": str(table), "sha256": digest})
                rows += _rows(model, agent_class, run["name"], cohort, table)
            runs.append(entry)

    # Every check runs before any write.
    models = [r["literature_model"] for r in runs]
    assert len(models) == len(set(models)), f"a literature model has two runs: {models}"
    if not test:
        assert len(models) == len(CONFIGS), f"{len(models)} finished runs, expected {len(CONFIGS)}"
    subjects = {}
    for row in rows:
        subjects.setdefault((row["literature_model"], row["cohort"]), []).append(row["subject_id"])
    for (model, cohort), ids in subjects.items():
        assert len(ids) == len(set(ids)), f"{model} {cohort}: a subject appears twice"
        if not test:
            assert len(ids) == N_SUBJECTS[cohort], f"{model} {cohort}: {len(ids)} subjects"
    for cohort in N_SUBJECTS:
        sets = {frozenset(ids) for (m, c), ids in subjects.items() if c == cohort}
        assert len(sets) <= 1, f"{cohort}: the models were fit on different subjects"

    split = json.loads(GRU_SPLIT.read_text())
    gru_train = {str(s) for s in split["source_subject_ids"]}
    gru_test = {str(s) for s in split["test_subject_ids"]}
    ours = {c: set(ids) for (m, c), ids in subjects.items()}
    if not test:
        assert ours["train"] | ours["heldout"] == gru_train | gru_test, "cohort is not study 01's 763"
    swap = {"heldout_here_train_in_gru_split": sorted(ours.get("heldout", set()) & gru_train),
            "train_here_test_in_gru_split": sorted(ours.get("train", set()) & gru_test)}

    rows.sort(key=lambda r: (r["literature_model"], r["cohort"], r["subject_id"]))
    out_csv = args.out_dir / "per_subject_fits.csv"
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    meta = build_meta("analysis/freeze_fits.py", groups, study_root=HERE.parent,
                      kind="freeze", code=[__file__], inputs=[GRU_SPLIT], sources=sources)
    write_artifact(args.out_dir / "per_subject_fits.json",
                   {"_meta": meta, "columns": COLUMNS, "gru_split_swap": swap, "runs": runs},
                   outputs=[out_csv])
    print(f"froze {len(rows)} rows from {len(runs)} runs to {out_csv}")


if __name__ == "__main__":
    main()
