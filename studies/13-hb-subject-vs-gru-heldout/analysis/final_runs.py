"""The final `hattori` run set (ADR 0004) and where each subject's posterior draws live.

The final set is one finished run per subject (``hattori_final_runs.json``): a rescore of the
production fit for the 103 subjects passing the gates, which saved no draws of its own (so its
draws are the production run's, ``hattori_runs.json``), and the non-centred refit for the other
46, which saved its own. Shared by the freezes that read draws.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRODUCTION_RUNS = HERE / "hattori_runs.json"
FINAL_RUNS = HERE / "hattori_final_runs.json"
PRODUCTION_GROUPS = ["hattori@20261026-002131"]
# ADR 0004: production fits for all 149, rescored (the centred refit, hattori@20261027-181415,
# was rejected): the 103 that passed the gates, then the 46 that did not.
FINAL_GROUPS = ["hattori@20261027-181430", "hattori@20261027-223626"]   # rescore, non-centred refit
ARTIFACT_ROOT = Path("/home/han.hou/outputs/disrnn/hb_artifacts")   # artifact_dir on HPC
DRAWS = "heldout_two_level_draws.npz"


def finished(path: Path) -> list[dict]:
    """Finished rows of a frozen run table."""
    return [r for r in json.loads(path.read_text())["rows"] if r["state"] == "finished"]


def final_rows() -> list[dict]:
    """One finished final run per subject."""
    rows = finished(FINAL_RUNS)
    assert len(rows) == len({r["subject_id"] for r in rows}), "a subject has two final runs"
    return rows


def draws_run(row: dict) -> str:
    """W&B run id whose directory holds this final row's posterior draws (HPC filesystem)."""
    if (ARTIFACT_ROOT / row["wandb_run_id"] / DRAWS).exists():
        return row["wandb_run_id"]                     # the refit: its own draws
    production = {r["subject_id"]: r["wandb_run_id"] for r in finished(PRODUCTION_RUNS)}
    return production[row["subject_id"]]             # a rescore: production's draws
