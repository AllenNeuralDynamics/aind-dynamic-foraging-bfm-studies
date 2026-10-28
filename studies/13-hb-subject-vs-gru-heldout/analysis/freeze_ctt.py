#!/usr/bin/env python
"""Freeze the `ctt` production runs into ``analysis/ctt_runs.json`` (``make freeze-ctt``).

The only step of r5 that reads live data. One row per run in the pinned W&B group: the held-out
subject it fitted, its pooling totals, k̂ rescue counts and gates from the run summary, and the
sha256 of the per-session scores file it wrote (the immutable key for its numbers). Reuses the
GraphQL reader of ``freeze_hattori.py``. Needs ``WANDB_API_KEY`` and the HPC artifact directory,
so it runs on a compute node. ``r5_ctt_pooled.py`` is offline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
sys.path.insert(0, str(HERE))
from _meta import build_meta, write_artifact  # noqa: E402
import freeze_hattori as fh  # noqa: E402

WANDB_GROUPS = ["ctt@20261028-003745"]               # the ctt production launch, pinned
OUT = HERE / "ctt_runs.json"
N_SUBJECTS = 149


def main() -> None:
    """One row per run; exactly one finished run per subject, checked before writing."""
    rows, sources = [], []
    for group in WANDB_GROUPS:
        for run in sorted(fh._runs(group), key=lambda r: r["name"]):
            summary = json.loads(run["summaryMetrics"] or "{}")
            scores = fh.ARTIFACT_ROOT / run["name"] / "heldout_two_level_scores.npz"
            assert run["state"] != "finished" or scores.exists(), f"{run['name']}: no {scores}"
            row = {"wandb_run_id": run["name"], "wandb_group": group, "state": run["state"],
                   "subject_id": fh._subject(json.loads(run["config"] or "{}")),
                   "hb_model": summary.get("hb/model"),
                   "scores_sha256": fh._sha256(scores) if scores.exists() else None,
                   **{key: summary.get(key) for key in fh.SUMMARY_KEYS + fh.FINAL_KEYS}}
            rows.append(row)
            sources.append({"type": "wandb_run", "id": f"{fh.ENTITY}/{fh.PROJECT}/{run['name']}",
                            "state": run["state"], "updated_at": run["updatedAt"]})
            if row["scores_sha256"]:
                sources.append({"type": "file", "id": str(scores), "sha256": row["scores_sha256"]})
    fh._once_each(rows, N_SUBJECTS, "ctt")
    assert all(r["hb_model"] == "ctt" for r in rows if r["state"] == "finished"), "not all ctt"
    meta = build_meta("analysis/freeze_ctt.py", WANDB_GROUPS, study_root=HERE.parent,
                      kind="freeze", code=[__file__, fh.__file__], sources=sources)
    write_artifact(OUT, {"_meta": meta, "rows": rows}, outputs=[])
    print(f"froze {len(rows)} runs ({sum(r['state'] == 'finished' for r in rows)} finished) to {OUT}")


if __name__ == "__main__":
    main()
