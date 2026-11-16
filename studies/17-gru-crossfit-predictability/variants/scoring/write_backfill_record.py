#!/usr/bin/env python
"""Write the `backfill` intervention record of backfill_maps.py for one scoring launch.

    python variants/scoring/write_backfill_record.py --label score-28 \\
        --results /allen/aind/scratch/han.hou/s17_scoring/backfill/results_score-28.json \\
        --one-run-check /allen/aind/scratch/han.hou/s17_scoring/backfill/rebuild_one_result.json

Writes launch_record/<label>/backfill_maps.json (launch-record v2, kind backfill, platform none).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
sys.path.insert(0, str(STUDY.parent / "util"))
from launch_record import write_intervention  # noqa: E402

WRAPPER_MERGE = "d52739ec114c53c2a64ba5922aa95e607ad508c9"   # wrapper PR #130 merge: maps + roles in the artifact


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("--label", "--results", "--one-run-check"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    launch = json.loads((HERE / "launch_record" / a.label / "hpc_launch.json").read_text())
    results = json.loads(Path(a.results).read_text())
    # "skipped (v1 exists)" = the one-run test (pool4-3way-full pass A), backfilled by the same script first.
    done = [r for r in results if r["status"] in ("backfilled", "skipped (v1 exists)")]
    check = json.loads(Path(a.one_run_check).read_text())
    studies_sha = subprocess.check_output(["git", "-C", str(STUDY), "rev-parse", "HEAD"], text=True).strip()
    write_intervention(
        HERE / "launch_record" / a.label / "backfill_maps.json",
        kind="backfill", platform="none", wandb_group=launch["wandb_group"], study_root=STUDY,
        label=f"{a.label}-backfill-maps",
        note="Adds heldout-rescore-<run>:v1 to each scoring run: the :v0 files plus the fine-tune's "
             "subject_index_map.json, session_context_map.json and the roles input, so W&B alone "
             "rebuilds each fine-tuned held-out model and its per-trial GRU states (wrapper #129).",
        trigger={"symptom": "heldout-rescore :v0 artifacts lack the fine-tune's subject index map, "
                            "session context map and roles input; the tuned held-out models could only "
                            "be rebuilt from HPC scratch.",
                 "cause": "heldout_rescore before wrapper PR #130 (merge d52739e) logged only scores, "
                          "predictions, embeddings and the manifest.",
                 "evidence": {"wrapper_issue": "AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper#129",
                              "wrapper_pr": "AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper#130",
                              "one_run_rebuild_from_wandb_only": check}},
        tasks=[{"orig_task": r["out"].split("/out/", 1)[1], "orig_wandb_run": r["run_id"],
                "new_wandb_run": r["run_id"], "v0_digest": r.get("v0_digest"), "v1_digest": r["v1_digest"],
                "v1_files": r.get("v1_files"), "status": r["status"]} for r in done],
        deviations={"same_run_ids": "each run was resumed (wandb resume='must') to log :v1; no metric "
                                    "or summary was written, the run returned to finished",
                    "v0_kept": ":v0 is unchanged; :v1 = :v0 files (md5-checked against scratch) + 3 files"},
        revisions={"studies": studies_sha, "wrapper": WRAPPER_MERGE,
                   **{k: v for k, v in launch["revisions"].items() if k not in ("studies", "wrapper")}},
        manifests=["backfill_maps.py", "write_backfill_record.py"],
    )
    print(a.label, len(done), "with :v1;", len(results) - len(done), "other:",
          sorted({r['status'] for r in results} - {"backfilled", "skipped (v1 exists)"}))


if __name__ == "__main__":
    main()
