#!/usr/bin/env python
"""Freeze the ctt-credit launch's per-subject fits (``make freeze-ctt-credit``).

The ctt-credit group has two runs: CompareToThreshold in stay-credit coordinates and the original
CompareToThreshold, refit on the same foraging-models commit. Same tables, row format and checks
as ``freeze_fits.py`` (whose helpers it reuses, unchanged, so r1's freeze stays current); writes
``analysis/ctt_credit_fits.csv`` and ``analysis/ctt_credit_fits.json``. Needs ``WANDB_API_KEY``
(the NeuralDynamics key) and ``/allen``, so it runs on a compute node.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
sys.path.insert(0, str(HERE))
from _meta import build_meta, write_artifact  # noqa: E402
import freeze_fits as ff  # noqa: E402

WANDB_GROUPS = ["ctt-credit@20261102-132626"]
CONFIGS = {
    "baseline_rl_lit_compare_to_threshold_credit": ("CompareToThreshold-credit",
                                                    "ForagerCompareThresholdCredit"),
    "baseline_rl_lit_compare_to_threshold": ("CompareToThreshold", "ForagerCompareThreshold"),
}


def main() -> None:
    rows, runs, sources = [], [], []
    for group in WANDB_GROUPS:
        for run in sorted(ff._runs(group), key=lambda r: r["name"]):
            assert run["state"] == "finished", f"{run['name']}: {run['state']}"
            config_name = ff._value(json.loads(run["config"] or "{}"), "model")
            model, agent_class = CONFIGS[config_name]
            run_dir, = ff.RUN_ROOT.glob(f"run-*-{run['name']}")
            entry = {"wandb_run_id": run["name"], "wandb_group": group, "model_config": config_name,
                     "literature_model": model, "agent_class": agent_class, "tables": {}}
            sources.append({"type": "wandb_run", "id": f"{ff.ENTITY}/{ff.PROJECT}/{run['name']}",
                            "state": run["state"], "updated_at": run["updatedAt"]})
            for cohort, rel in ff.TABLES.items():
                table = run_dir / "files" / rel
                digest = ff._sha256(table)
                entry["tables"][cohort] = {"path": str(table), "sha256": digest}
                sources.append({"type": "file", "id": str(table), "sha256": digest})
                rows += ff._rows(model, agent_class, run["name"], cohort, table)
            runs.append(entry)

    # Every check runs before any write.
    models = sorted(r["literature_model"] for r in runs)
    assert models == sorted(m for m, _ in CONFIGS.values()), f"runs: {models}"
    subjects = {}
    for row in rows:
        subjects.setdefault((row["literature_model"], row["cohort"]), []).append(row["subject_id"])
    for (model, cohort), ids in subjects.items():
        assert len(ids) == len(set(ids)) == ff.N_SUBJECTS[cohort], f"{model} {cohort}: {len(ids)}"
    # The same subjects, in the same cohorts, as r1's freeze.
    with (HERE / "per_subject_fits.csv").open(newline="") as handle:
        r1 = {(r["cohort"], r["subject_id"]) for r in csv.DictReader(handle)
              if r["literature_model"] == "CompareToThreshold"}
    for model in models:
        assert {(r["cohort"], r["subject_id"]) for r in rows
                if r["literature_model"] == model} == r1, f"{model}: not r1's subjects"

    rows.sort(key=lambda r: (r["literature_model"], r["cohort"], r["subject_id"]))
    out_csv = HERE / "ctt_credit_fits.csv"
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ff.COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    meta = build_meta("analysis/freeze_ctt_credit.py", WANDB_GROUPS, study_root=HERE.parent,
                      kind="freeze", code=[__file__, HERE / "freeze_fits.py"],
                      inputs=[HERE / "per_subject_fits.csv"], sources=sources)
    write_artifact(HERE / "ctt_credit_fits.json",
                   {"_meta": meta, "columns": ff.COLUMNS, "runs": runs}, outputs=[out_csv])
    print(f"froze {len(rows)} rows from {len(runs)} runs to {out_csv}")


if __name__ == "__main__":
    main()
