#!/usr/bin/env python
"""Freeze r1/r2's per-held-out-subject rows from W&B into ``per_subject_rows.json``
(``make freeze-per-subject``).

The only step of r1/r2 that reads W&B (the ``wandb`` SDK). Keeps one offline held-out
rerun per (variant, ratio, seed), reads its per-subject likelihood table, and records each
run id with the table artifact's name and digest; ``build_report.py`` is offline.
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
from build_report import FROZEN, PROJECT, WANDB_GROUPS  # noqa: E402


def _variant(meta):  # 'v1' / 'v2' from meta.variant (strip -retry)
    v = (meta or {}).get("variant", "")
    return "v1" if v.startswith("v1") else ("v2" if v.startswith("v2") else None)


def _per_subject_table_dataframe(run):
    """(dataframe, artifact) for the run's held-out per-subject likelihood table."""
    for art in run.logged_artifacts():
        if art.type != "run_table":
            continue
        for entry_name in art.manifest.entries:
            if "per_subject_likelihood" not in str(entry_name):
                continue
            return art.get(entry_name).get_dataframe(), art
    raise ValueError("no heldout/per_subject_likelihood table artifact found")

def collect_per_subject():
    """Live pull: rows plus the one deduped run per (variant, ratio, seed) and its table artifact."""
    api = wandb.Api()
    rows = []  # variant, ratio, seed, subject, n_trials, ll
    sources = []  # one per kept run: its id and the immutable table artifact read
    runs = [r for r in api.runs(PROJECT)
            if r.group in WANDB_GROUPS and r.state == "finished"]
    groups = sorted({r.group for r in runs if r.group})
    # DEDUP: validation + mass-launch + retries can produce >1 offline run per
    # (variant, ratio, seed). Keep exactly one (the most recent) so seed-averaging
    # isn't skewed by double-counted cells. (The offline finetune is deterministic
    # from the same checkpoint, so duplicates are ~identical, but partial duplication
    # — e.g. only seed 0 re-run — would over-weight that seed.)
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
    runs = list(by_cell.values())
    print(f"  deduped to {len(runs)} offline runs (one per variant,ratio,seed)")
    for r in runs:
        meta = r.config.get("meta", {}) or {}
        var = _variant(meta)
        ratio = meta.get("source_subject_ratio"); seed = meta.get("source_seed")
        if var is None or ratio is None:
            continue
        try:
            df, art = _per_subject_table_dataframe(r)
        except Exception as e:
            print(f"  [skip {r.name[:40]}] table err: {e}")
            continue
        sources.append(dict(variant=var, ratio=round(float(ratio), 3), seed=seed,
                            wandb_run_id=r.id, artifact=art.qualified_name,
                            artifact_digest=art.digest))
        for _, row in df.iterrows():
            rows.append(dict(variant=var, ratio=round(float(ratio), 3), seed=seed,
                             subject=str(row["heldout_subject_id"]),
                             n_trials=int(row["n_trials"]), ll=float(row["eval_likelihood"])))
    return rows, groups, sources


def main():
    rows, groups, sources = collect_per_subject()
    out = {"_meta": build_meta("analysis/freeze_per_subject.py", groups, study_root=HERE.parent),
           "sources": sources, "rows": rows}
    FROZEN.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {FROZEN} ({len(rows)} rows from {len(sources)} runs)")


if __name__ == "__main__":
    main()
