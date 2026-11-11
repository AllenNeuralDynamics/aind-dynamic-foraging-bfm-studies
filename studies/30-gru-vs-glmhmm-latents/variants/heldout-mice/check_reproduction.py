#!/usr/bin/env python
"""Reproduction gate: the redone fine-tune must match study 02's logged held-out scores.

Reads the source run's row from study 02's committed ``analysis/frozen_runs.json`` (never a
retyped number) and the fine-tune's ``output_summary.json``, compares the metrics both carry, and
writes ``<out_root>/reproduction.json``. Exits 1 if any differs by more than the tolerance:

    python check_reproduction.py <out_root>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY02_FROZEN = HERE.parents[2] / "02-gru-scaling-law-ignore" / "analysis" / "frozen_runs.json"
SOURCE_RUN = "nxd-3way-20260707-075401-80a6d102"
# Deterministic fine-tune (mean-embedding start, full batches): any real difference is a bug.
TOL = 1e-4
METRICS = ("eval_likelihood", "eval_likelihood_LR_engaged", "eval_likelihood_engage",
           "engage_ignore_pr_auc", "engage_ignore_base_rate")


def main() -> None:
    out_root = Path(sys.argv[1])
    (summary_path,) = out_root.glob("heldout/*/outputs/output_summary.json")
    redone = json.loads(summary_path.read_text())["metrics"]
    (logged,) = [r for r in json.loads(STUDY02_FROZEN.read_text())["runs"]
                 if r["wandb_run_id"] == SOURCE_RUN]
    rows = {m: {"study02": logged[m], "redone": redone[m], "diff": redone[m] - logged[m]}
            for m in METRICS}
    ok = all(abs(r["diff"]) <= TOL for r in rows.values())
    result = {"source_run": SOURCE_RUN, "study02_source": str(STUDY02_FROZEN.relative_to(HERE.parents[3])),
              "output_summary": str(summary_path), "tolerance": TOL, "pass": ok, "metrics": rows}
    (out_root / "reproduction.json").write_text(json.dumps(result, indent=1) + "\n")
    for m, r in rows.items():
        print(f"{m:28s} study02 {r['study02']:.7f}  redone {r['redone']:.7f}  diff {r['diff']:+.2e}")
    print("PASS" if ok else f"FAIL: a difference exceeds {TOL}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
