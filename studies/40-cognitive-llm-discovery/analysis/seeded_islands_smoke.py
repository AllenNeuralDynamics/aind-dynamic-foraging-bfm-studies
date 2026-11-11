#!/usr/bin/env python
"""The seeded-islands-smoke result (``make smoke-result``): the numbers the variant's notes
pre-registered, from the committed freeze ``analysis/seeded_islands_smoke/run.json``. Offline.

Writes ``analysis/seeded_islands_smoke.json``: the pass checks (proposals, dollars, hours, API errors,
archive), dollars and node-minutes per proposal, evaluation seconds, proposals by outcome, 2-level fits
that failed, the best score against the best seed's, by lineage (the seed a program descends from),
and the gain of each new best over the previous one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

FROZEN = HERE / "seeded_islands_smoke" / "run.json"
OUT = HERE / "seeded_islands_smoke.json"
GROUP = "seeded-islands-smoke@20261105-011815"
N_PROPOSALS, MAX_USD, MAX_HOURS = 100, 20.0, 12.0     # notes.md, pre-registered
LINEAGES = ("template", "Miller-RHG", "Lebedeva-PR", "Lopez-DoubleTrace", "Eckstein-BI")


def main() -> None:
    run = json.loads(FROZEN.read_text())
    progs = run["programs"]
    by_id = {p["id"]: p for p in progs}

    def lineage(p):
        while p["parent_id"] in by_id:
            p = by_id[p["parent_id"]]
        return p["seed"] or "template"

    proposals = [p for p in progs if p["generation"] > 0]
    scored = [p for p in proposals if p["correct"]]
    fresh = [p for p in scored if not p["repeat_of"]]
    ev = np.array([p["eval_seconds"] for p in fresh])
    calls, summary = run["calls"], run["summary"]
    seeds = {p["seed"] or "template": p["score"] for p in progs if p["generation"] == 0}
    best_seed = max(v for k, v in seeds.items() if k != "template")
    best = max(scored, key=lambda p: p["score"])
    steps, cur = [], best_seed
    for p in sorted(proposals, key=lambda p: p["generation"]):
        if p["correct"] and p["score"] > cur:
            steps.append({"generation": p["generation"], "score": p["score"], "gain": p["score"] - cur})
            cur = p["score"]
    out = {
        "pass": {"proposals": len(proposals) + len(run["no_program"]), "cost_usd": summary["cost_usd"],
                 "hours": summary["seconds"] / 3600, "api_errors": len(summary["errors"]),
                 "stop_reasons": summary["stop_reasons"], "archive": run["archive"],
                 "passed": (len(proposals) + len(run["no_program"]) == N_PROPOSALS and summary["cost_usd"] <= MAX_USD
                            and summary["seconds"] / 3600 <= MAX_HOURS and not summary["errors"] and bool(run["archive"]["digest"]))},
        "calls": {"n": len(calls), "repair_calls": len(calls) - N_PROPOSALS,
                  "median_seconds": float(np.median([c["seconds"] for c in calls]))},
        "per_proposal": {"usd_mean": summary["cost_usd"] / N_PROPOSALS,
                         "usd_median": float(np.median([p["cost_usd"] for p in proposals])),
                         "node_minutes": summary["seconds"] / 60 / N_PROPOSALS},
        "eval_seconds": {"n": len(fresh), "median": float(np.median(ev)), "p10": float(np.percentile(ev, 10)),
                         "p90": float(np.percentile(ev, 90)), "max": float(ev.max())},
        "outcomes": {"fresh_fit": len(fresh), "repeat": len(scored) - len(fresh),
                     "rejected_by_checks": [{"generation": p["generation"], "reason": p["failure"]} for p in proposals if not p["correct"]],
                     "no_program": run["no_program"]},
        "two_level_failed": sum(bool(p["two_level_failed"]) for p in scored),
        "better_than_parent": sum(1 for p in scored if p["parent_id"] in by_id and by_id[p["parent_id"]]["correct"]
                                  and p["score"] > by_id[p["parent_id"]]["score"]),
        "n_scored": len(scored),
        "best": {"generation": best["generation"], "score": best["score"], "best_seed": best_seed,
                 "gain_over_best_seed": best["score"] - best_seed, "n_params": best["n_params"], "n_latent": best["n_latent"]},
        "new_bests": steps,
        "lineages": {k: {"proposals": sum(1 for p in proposals if lineage(p) == k), "seed": seeds[k],
                         "best": max([p["score"] for p in scored if lineage(p) == k] + [seeds[k]])} for k in LINEAGES},
        "params_7_or_8": sum(1 for p in scored if p["n_params"] in (7, 8)),
    }
    write_artifact(OUT, {"_meta": build_meta("analysis/seeded_islands_smoke.py", [GROUP], study_root=STUDY,
                                             code=[__file__], inputs=[FROZEN]),
                         "about": "seeded-islands-smoke: the pre-registered numbers of its notes.", **out})
    print(json.dumps({k: out[k] for k in ("pass", "best")}, default=float)[:600])


if __name__ == "__main__":
    main()
