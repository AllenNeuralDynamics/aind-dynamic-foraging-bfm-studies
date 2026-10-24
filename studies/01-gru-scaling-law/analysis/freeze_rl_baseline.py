#!/usr/bin/env python
"""Freeze the classical-RL held-out per-subject tables into ``analysis/rl_baseline_tables.json``
(``make freeze-rl-baseline``).

The only step of r8 (and of the RL overlays on r1/r4/r5/r7) that reads live data: W&B (the
``wandb`` SDK), falling back to the run output dirs on ``/allen`` when the artifact API
fails. One table per run id in ``RL_MODELS`` + the legacy Bari run; ``rl_baseline.py`` is
offline. Each table records where it came from (``origin``): the W&B artifact name and digest,
or the ``/allen`` CSV path and its sha256. The GRU side is ``per_subject_rows.json``
(freeze_per_subject.py).
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _meta import build_meta  # noqa: E402
from rl_baseline import LEGACY_BARI_RUN, PROJECT, RL_FROZEN, RL_MODELS  # noqa: E402

# Per-subject tables live in each run's output dir on /allen (authoritative; the study
# README notes these are independent of W&B). Used when the W&B artifact API is
# unavailable, which it intermittently is.
RUN_OUTPUT_ROOT = Path("/allen/aind/scratch/han.hou/outputs/disrnn/wandb")

COLUMNS = ["subject_id", "eval_likelihood", "eval_total_log_likelihood", "eval_total_trials",
           "curriculum_name"]


def _heldout_table_from_wandb(api, run_id):
    """Held-out per-subject table from W&B.

    The 2026-07-13 suite fits BOTH cohorts, so each run logs `subject_fit_metrics`
    twice — 614 training mice and 149 held-out mice. We must take the HELD-OUT one
    (the trainer logs it through a proxy that prefixes keys with `heldout/`). The
    legacy Bari-only run logged a single table, so a lone table is unambiguous.
    """
    r = api.run(f"{PROJECT}/{run_id}")
    tables = [(a, e) for a in r.logged_artifacts() if a.type == "run_table"
              for e in a.manifest.entries]
    if not tables:
        raise ValueError("no run_table artifact")
    heldout = [(a, e) for a, e in tables if "heldout" in str(e).lower()]
    pick = heldout or ([tables[0]] if len(tables) == 1 else [])
    if not pick:
        raise ValueError(f"{len(tables)} tables and none marked heldout — refusing to guess")
    a, e = pick[0]
    return a.get(e).get_dataframe(), r.group, {"artifact": a.qualified_name, "artifact_digest": a.digest, "entry": str(e)}


def _heldout_table_from_disk(run_id):
    """Fallback: the authoritative per-subject CSV in the run's output dir on /allen."""
    dirs = sorted(RUN_OUTPUT_ROOT.glob(f"run-*-{run_id}"))
    if not dirs:
        raise FileNotFoundError(f"no run dir for {run_id} under {RUN_OUTPUT_ROOT}")
    csv = dirs[-1] / "files" / "outputs" / "heldout_test" / "subject_fit_metrics.csv"
    if not csv.exists():
        raise FileNotFoundError(csv)
    import pandas as pd
    return pd.read_csv(csv), None, {"path": str(csv),
                                    "sha256": hashlib.sha256(csv.read_bytes()).hexdigest()}


def main():
    api = wandb.Api()
    runs = {}
    for run_id in [*(m["run_id"] for m in RL_MODELS.values()), LEGACY_BARI_RUN]:
        try:
            df, group, origin = _heldout_table_from_wandb(api, run_id)
            source = "wandb"
        except Exception as exc:                       # W&B artifact API is flaky
            print(f"  {run_id}: W&B artifact pull failed ({type(exc).__name__}); "
                  f"falling back to run output dir on /allen")
            df, group, origin = _heldout_table_from_disk(run_id)
            source = "disk"
        runs[run_id] = {"group": group, "source": source, "origin": origin,
                        "table": df[COLUMNS].to_dict(orient="records")}
        print(f"  {run_id}: {len(df)} subjects [{source}]")
    groups = sorted({r["group"] for r in runs.values() if r["group"]})
    out = {"_meta": build_meta("analysis/freeze_rl_baseline.py", groups), "runs": runs}
    RL_FROZEN.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {RL_FROZEN}")


if __name__ == "__main__":
    main()
