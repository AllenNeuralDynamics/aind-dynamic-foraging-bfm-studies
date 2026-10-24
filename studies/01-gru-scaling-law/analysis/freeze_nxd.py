#!/usr/bin/env python
"""Freeze r7's N x D grid from W&B into ``analysis/nxd_runs.json`` (``make freeze-nxd``).

The only step of r7 that reads W&B (the ``wandb`` SDK). One row per (N, D, seed) of the
pinned groups -- the newest finished run -- keyed by its W&B run id; ``nxd_scaling.py``
fits and plots offline from that file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "util"))   # shared _meta (studies + source SHAs), ahead of the local copy
from _meta import build_meta  # noqa: E402
from nxd_scaling import (FROZEN, H128_GROUP, NXD_GROUPS, PROJECT, RATIO_D,  # noqa: E402
                         TARGET_HS, TARGET_RATIOS)
from wandb_keys import HELDOUT_LL_KEYS  # noqa: E402


def _ratio_key(x):
    if x is None:
        return None
    f = float(x)
    for r in TARGET_RATIOS:
        if abs(f - r) < 1e-4:
            return r
    return None


def collect_runs():
    """One row per (N, D, seed): the newest finished run and its aggregate held-out LL."""
    api = wandb.Api()
    runs = []
    groups = [*NXD_GROUPS, H128_GROUP]
    for grp in groups:
        runs.extend(api.runs(PROJECT, filters={"group": grp}))
    runs = [r for r in runs if r.state == "finished"]
    print(f"  found {len(runs)} finished runs across {', '.join(groups)}")

    by_cell = {}
    for r in runs:
        h = r.config.get("model", {}).get("architecture", {}).get("hidden_size")
        rk = _ratio_key(r.config.get("data", {}).get("subject_ratio"))
        seed = r.config.get("data", {}).get("subject_sample_seed")
        if h not in TARGET_HS or rk is None or seed is None:
            continue
        D = RATIO_D[rk]
        key = (h, D, int(seed))
        prior = by_cell.get(key)
        if prior is None or str(getattr(r, "created_at", "")) > str(getattr(prior, "created_at", "")):
            by_cell[key] = r

    target = len(TARGET_HS) * len(TARGET_RATIOS) * 3
    print(f"  unique (N, D, seed) cells: {len(by_cell)} "
          f"(rectangular target {target} = {len(TARGET_HS)}x{len(TARGET_RATIOS)}x3; "
          f"H=512 is missing D=614, so the achievable max is "
          f"{target - 1 * 3} = {len(TARGET_HS)-1}x{len(TARGET_RATIOS)}x3 + 1x{len(TARGET_RATIOS)-1}x3)")

    rows = []
    no_scalar = 0
    for (h, D, seed), r in by_cell.items():
        s = next((r.summary.get(k) for k in HELDOUT_LL_KEYS if r.summary.get(k) is not None), None)
        if s is None:
            no_scalar += 1
            continue
        rows.append({"N": h, "D": D, "seed": seed, "wandb_run_id": r.id, "heldout_ll": float(s)})
    if no_scalar:
        print(f"  WARN: skipped {no_scalar} runs missing {HELDOUT_LL_KEYS[0]}")
    if by_cell and no_scalar == len(by_cell):
        raise KeyError(
            f"all {len(by_cell)} cells missing {HELDOUT_LL_KEYS} "
            f"(wrapper schema changed? see analysis/wandb_keys.py)"
        )
    return rows


def main():
    rows = collect_runs()
    out = {"_meta": build_meta("analysis/freeze_nxd.py", [*NXD_GROUPS, H128_GROUP], study_root=HERE.parent), "runs": rows}
    FROZEN.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {FROZEN} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
