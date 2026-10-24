#!/usr/bin/env python
"""Freeze r9's generative-rollout scalars from W&B into ``analysis/generative_runs.json``
(``make freeze-generative``).

The only step of r9 that reads W&B (the ``wandb`` SDK). One row per (variant, ratio, seed)
-- the newest finished rollout -- keyed by its W&B run id; ``generative_match.py`` is offline.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "util"))   # shared _meta (studies + source SHAs), ahead of the local copy
from _meta import build_meta  # noqa: E402
from generative_match import (DLAB, FROZEN, HIST_N_BACKS, HIST_PATTERNS, K_CORR,  # noqa: E402
                              K_CORR_OVERALL, K_MSE, PROJECT, WANDB_GROUPS)
from wandb_keys import hist_corr, hist_mse  # noqa: E402


def _variant(meta):
    v = (meta or {}).get("variant", "")
    return "v1" if v.startswith("v1") else ("v2" if v.startswith("v2") else None)


def collect():
    """Live pull of the generative rollouts, one row per cell."""
    api = wandb.Api()
    runs = [r for r in api.runs(PROJECT)
            if r.group in WANDB_GROUPS and r.state == "finished"]
    groups = sorted({r.group for r in runs if r.group})
    by_cell = {}
    for r in runs:
        meta = r.config.get("meta", {}) or {}
        var = _variant(meta)
        ratio = meta.get("source_subject_ratio"); seed = meta.get("source_seed")
        if var is None or ratio is None:
            continue
        key = (var, round(float(ratio), 3), seed)
        prev = by_cell.get(key)
        if prev is None or str(getattr(r, "created_at", "")) > str(getattr(prev, "created_at", "")):
            by_cell[key] = r
    print(f"  {len(runs)} finished generative runs -> {len(by_cell)} cells")
    rows = []
    for (var, ratio, seed), r in by_cell.items():
        s = r.summary
        corr = s.get(K_CORR); mse = s.get(K_MSE); corr_o = s.get(K_CORR_OVERALL)
        if corr is None or mse is None:
            print(f"    [skip {r.name[:40]}] missing combined scalars")
            continue
        row = dict(wandb_run_id=r.id, variant=var, ratio=ratio, D=DLAB.get(ratio, ratio), seed=seed,
                   corr=float(corr), corr_overall=float(corr_o) if corr_o is not None else None,
                   rmse=math.sqrt(float(mse)))
        # history-dependent metric: corr + sqrt(subject-balanced MSE), per (pattern, n_back).
        for pattern in HIST_PATTERNS:
            for n in HIST_N_BACKS:
                hc = s.get(hist_corr(pattern, n))
                hm = s.get(hist_mse(pattern, n))
                row[f"hist_{pattern}_n{n}_corr"] = float(hc) if hc is not None else None
                row[f"hist_{pattern}_n{n}_rmse"] = math.sqrt(float(hm)) if hm is not None else None
        rows.append(row)
    # All finished cells missing the required scalars means a renamed key, not
    # a few partial runs -> fail loudly rather than emit an empty report.
    if by_cell and not rows:
        raise KeyError(
            f"none of {len(by_cell)} finished generative cells carry {K_CORR!r}/"
            f"{K_MSE!r} (wrapper schema changed? see analysis/wandb_keys.py)"
        )
    return rows, groups


def main():
    rows, groups = collect()
    out = {"_meta": build_meta("analysis/freeze_generative.py", groups, study_root=HERE.parent), "rows": rows}
    FROZEN.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {FROZEN} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
