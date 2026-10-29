#!/usr/bin/env python
"""Freeze the `ctt` runs (``make freeze-ctt``): production into ``analysis/ctt_runs.json``, the
refit into ``analysis/ctt_refit_runs.json``, and the final set (the refit for the subjects that
failed a gate in production, production for the rest) into ``analysis/ctt_final_runs.json``.

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

PRODUCTION = ["ctt@20261028-003745"]                 # the ctt production launch, pinned
REFIT = ["ctt@20261028-020002"]                      # non-centred refit of the gate-failing subjects
WANDB_GROUPS = PRODUCTION + REFIT
OUT = HERE / "ctt_runs.json"
OUT_REFIT = HERE / "ctt_refit_runs.json"
OUT_FINAL = HERE / "ctt_final_runs.json"
N_SUBJECTS = 149


def _passes(row: dict) -> bool:
    """Every gate: R-hat < 1.01, bulk ESS >= 400, no divergences."""
    return (row["hb/max_r_hat"] < 1.01 and row["hb/min_ess_bulk"] >= 400
            and row["hb/divergences"] == 0)


def _read(groups: list[str]) -> tuple[list[dict], list[dict]]:
    """One row per run in ``groups``, and the sources read."""
    rows, sources = [], []
    for group in groups:
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
    assert all(r["hb_model"] == "ctt" for r in rows if r["state"] == "finished"), "not all ctt"
    return rows, sources


def _write(out: Path, groups: list[str], rows: list[dict], sources: list[dict]) -> None:
    """Write one frozen table."""
    meta = build_meta("analysis/freeze_ctt.py", groups, study_root=HERE.parent,
                      kind="freeze", code=[__file__, fh.__file__], sources=sources)
    write_artifact(out, {"_meta": meta, "rows": rows}, outputs=[])
    print(f"froze {len(rows)} runs ({sum(r['state'] == 'finished' for r in rows)} finished) to {out}")


def main() -> None:
    """Freeze production, the refit and the final set; every check runs before any write."""
    production, p_sources = _read(PRODUCTION)
    refit, r_sources = _read(REFIT)
    fh._once_each(production, N_SUBJECTS, "ctt production")
    failing = {r["subject_id"] for r in production if r["state"] == "finished" and not _passes(r)}
    fh._once_each(refit, len(failing), "ctt refit")
    assert {r["subject_id"] for r in refit if r["state"] == "finished"} == failing, "refit set"
    final = ([r for r in production if r["state"] == "finished" and r["subject_id"] not in failing]
             + [r for r in refit if r["state"] == "finished"])
    fh._once_each(final, N_SUBJECTS, "ctt final")
    _write(OUT, PRODUCTION, production, p_sources)
    _write(OUT_REFIT, REFIT, refit, r_sources)
    _write(OUT_FINAL, WANDB_GROUPS, final, p_sources + r_sources)


if __name__ == "__main__":
    main()
