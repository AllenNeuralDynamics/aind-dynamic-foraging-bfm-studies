#!/usr/bin/env python
"""Freeze r4's rollout scalars from W&B into ``analysis/generative_runs.json`` (``make freeze-generative``).

The only step of r4 that reads W&B (the ``wandb`` SDK; credentials from ``~/.netrc`` or
``WANDB_API_KEY``). One row per (D, seed) cell of the pinned group, keyed by W&B run id;
``generative_match.py`` builds r4 offline from that file.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta  # noqa: E402
from generative_match import (DLAB, FROZEN_ROLLOUTS, GROUP, HIST_N_BACK, HIST_PATTERN,  # noqa: E402
                              PROJECT, STAT)
from wandb_keys import hist_corr, hist_mse, switch_corr, switch_mse  # noqa: E402


def collect() -> list[dict]:
    api = wandb.Api()
    runs = [r for r in api.runs(PROJECT, {"group": GROUP}) if r.state == "finished"]
    by_cell: dict[tuple, object] = {}
    for r in runs:
        meta = r.config.get("meta", {}) or {}
        ratio, seed = meta.get("source_subject_ratio"), meta.get("source_seed")
        if ratio is None:
            continue
        key = (round(float(ratio), 3), seed)
        prev = by_cell.get(key)
        if prev is None or str(getattr(r, "created_at", "")) > str(getattr(prev, "created_at", "")):
            by_cell[key] = r

    rows = []
    for (ratio, seed), r in by_cell.items():
        s = r.summary
        corr, mse = s.get(switch_corr(STAT)), s.get(switch_mse(STAT))
        if corr is None or mse is None:
            print(f"    [skip {r.name[:40]}] missing combined switch scalars")
            continue
        hc = s.get(hist_corr(HIST_PATTERN, HIST_N_BACK))
        hm = s.get(hist_mse(HIST_PATTERN, HIST_N_BACK))
        rows.append(
            dict(
                wandb_run_id=r.id,
                created_at=str(getattr(r, "created_at", "")),
                D=DLAB.get(ratio, ratio),
                seed=seed,
                corr=float(corr),
                rmse=math.sqrt(float(mse)),
                hist_corr=float(hc) if hc is not None else None,
                hist_rmse=math.sqrt(float(hm)) if hm is not None else None,
            )
        )
    # All finished cells missing the scalars means a renamed key, not a few partial runs.
    if by_cell and not rows:
        raise KeyError(
            f"none of {len(by_cell)} finished cells in {GROUP} carry the switch-triggered "
            f"scalars (wrapper schema changed? see analysis/wandb_keys.py)"
        )
    print(f"  {len(runs)} finished rollouts -> {len(rows)} cells")
    return rows


def main() -> None:
    rows = collect()
    out = {"_meta": build_meta("analysis/freeze_generative.py", [GROUP], study_root=STUDY),
           "rows": rows}
    FROZEN_ROLLOUTS.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {FROZEN_ROLLOUTS} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
