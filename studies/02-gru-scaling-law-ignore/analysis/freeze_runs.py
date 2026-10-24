#!/usr/bin/env python
"""Freeze the study's 3-way grid from W&B into ``frozen_runs.json`` (``make freeze``).

The only step of this study that reads W&B (GraphQL; needs ``WANDB_API_KEY``). It keeps runs
from the pinned ``WANDB_GROUPS`` only, dedupes to one run per (D, H, seed) with the rule
below, and writes one row per kept run keyed by its W&B run id. ``scaling.py`` then builds
every report artifact offline from that file; the git diff of a refresh is the audit trail.
"""
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta  # noqa: E402
from scaling import (ENTITY, FROZEN_RUNS, HS, METRIC_KEYS, PROJECT, SR_TO_D,  # noqa: E402
                     WANDB_GROUPS, _get_nested)


def _wb(query):
    key = os.environ["WANDB_API_KEY"]
    data = json.dumps({"query": query}).encode()
    req = urllib.request.Request(
        "https://api.wandb.ai/graphql", data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": "Basic " + base64.b64encode(f"api:{key}".encode()).decode()})
    return json.load(urllib.request.urlopen(req))


def fetch_runs(first=500):
    """Live pull of the 3-way grid (HPC + Beaker runs alike): one row per (D,H,seed),
    in the order the pull returned them, deduped by the rule below."""
    q = ('query{project(name:"%s",entityName:"%s"){runs(first:%d){edges{node{'
         'name state group sweepName config summaryMetrics}}}}}' % (PROJECT, ENTITY, first))
    r = _wb(q)
    # first pass: collect one row per run, keyed by (D,H,seed) with a tiebreak on _step
    best = {}  # (D,H,seed) -> (score, row)
    for e in r["data"]["project"]["runs"]["edges"]:
        n = e["node"]
        if n.get("group") not in WANDB_GROUPS:
            continue  # pinned allowlist: never absorb runs this study didn't declare
        cfg = json.loads(n["config"] or "{}")
        sm = json.loads(n["summaryMetrics"] or "{}")
        H = _get_nested(cfg, "model.architecture.hidden_size")
        sr = _get_nested(cfg, "data.subject_ratio")
        seed = _get_nested(cfg, "data.seed", "seed", "data.subject_sample_seed")
        D = SR_TO_D.get(sr)
        if H not in HS or D is None or seed is None:
            continue
        lr = sm.get("heldout/final/eval_likelihood_LR_engaged")
        if not isinstance(lr, (int, float)):
            continue  # held-out not finished for this run
        step = sm.get("_step", 0) or 0
        # Prefer the run that carries ignore metrics, then prefer the LOWEST _step.
        # IMPORTANT (corrected 2026-07-11): the DISRNN_RESTORE_FROM_RUN_ID restore
        # path resumes BASE TRAINING (early-stop patience resets), over-training the
        # model ~30k steps past the original early-stop, so restore/backfill runs
        # (higher _step, e.g. 120505 vs the native 90505) hold a DIFFERENT, degraded
        # model -- both their LR-engaged AND ignore metrics are wrong (e.g. D10/H256
        # native 0.71841 vs restore 0.64816). The 12 affected cells were re-scored
        # exactly via resume_heldout_beaker.py, writing correct ignore metrics INTO
        # the native run at its own (lower) step. So among ignore-bearing runs the
        # native run (lowest _step) is canonical; tie-break lowest step picks it and
        # discards the over-trained restore run. Cells never touched by restore have
        # a single native ignore-bearing run, so the rule is a no-op for them.
        has_ig = isinstance(sm.get("heldout/final/engage_ignore_pr_auc"), (int, float))
        score = (1 if has_ig else 0, -step)  # ignore-bearing first, then LOWEST step
        key = (D, H, seed)
        if key in best and best[key][0] >= score:
            continue  # keep the better run (ignore-bearing, then native/least-trained)
        row = {"D": D, "H": H, "seed": seed, "wandb_run_id": n["name"],
               "wandb_group": n.get("group"), "step": step}
        row.update({k: sm.get("heldout/final/" + k) for k in METRIC_KEYS})
        best[key] = (score, row)
    return [row for _, row in best.values()]


def main():
    runs = fetch_runs()
    groups = sorted({r["wandb_group"] for r in runs if r["wandb_group"]})
    out = {"_meta": build_meta("analysis/freeze_runs.py", groups, study_root=STUDY),
           "runs": runs}
    FROZEN_RUNS.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {FROZEN_RUNS} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
