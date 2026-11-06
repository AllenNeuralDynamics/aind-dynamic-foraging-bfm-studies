#!/usr/bin/env python
"""The stopping rule's standard error (``make stop-se``), from the committed freeze
``analysis/stop_se/steps.json``. Offline.

Writes ``analysis/stop_se.json``: for each step of `seeded-islands-smoke` and `seeded-islands-cap12` (a new
best against the best before it), the score difference and its paired standard error (×100); their
median, range and count; the pair from each run's best seed to its final best; and ``stop_se_x100``,
the median rounded to 4 decimals: the value `seeded-islands` pre-registers as ``stop_se``.
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

FROZEN = HERE / "stop_se" / "steps.json"
OUT = HERE / "stop_se.json"
GROUPS = ["seeded-islands-smoke@20261105-011815", "seeded-islands-cap12@20261105-100249"]


def paired_se(a: list[dict], b: list[dict]) -> dict:
    """``mevolve.gate2.paired_se``, restated here so this producer runs offline: score difference (×100)
    a − b on the same subjects and its standard error, subjects as the unit, by the delta method."""
    la = {r["subject_id"]: r["log_lik"] for r in a}
    n_i = np.array([r["n_trials"] for r in b], dtype=float)
    ll_a = np.array([la[r["subject_id"]] for r in b])
    ll_b = np.array([r["log_lik"] for r in b])
    n_tot = n_i.sum()
    m_a, m_b = ll_a.sum() / n_tot, ll_b.sum() / n_tot
    infl = np.stack([ll_a - m_a * n_i, ll_b - m_b * n_i]) / n_tot
    cov = len(n_i) * np.cov(infl, ddof=1)
    grad = 100 * np.array([np.exp(m_a), -np.exp(m_b)])
    return {"score_diff_x100": float(100 * (np.exp(m_a) - np.exp(m_b))),
            "se_x100": float(np.sqrt(max(grad @ cov @ grad, 0.0))), "n_subjects": len(n_i)}


def main() -> None:
    frozen = json.loads(FROZEN.read_text())
    steps, whole = [], {}
    for name, run in frozen["runs"].items():
        progs = run["programs"]
        for a, b in zip(progs, progs[1:]):
            pair = paired_se(b["validation"], a["validation"])
            frozen_pair = next(p for p in run["pairs"] if p["from"] == a["generation"] and p["to"] == b["generation"])
            assert abs(pair["se_x100"] - frozen_pair["se_x100"]) < 1e-9, (name, a["generation"], b["generation"])
            steps.append({"run": name, "from": a["generation"], "to": b["generation"], **pair})
        whole[name] = {"from": progs[0]["generation"], "to": progs[-1]["generation"],
                       **paired_se(progs[-1]["validation"], progs[0]["validation"])}
    se = np.array([s["se_x100"] for s in steps])
    out = {"steps": steps, "best_seed_to_best": whole,
           "summary": {"n_steps": len(steps), "median_se_x100": float(np.median(se)),
                       "min_se_x100": float(se.min()), "max_se_x100": float(se.max()),
                       "steps_above_median_se": int(sum(s["score_diff_x100"] > np.median(se) for s in steps))},
           "stop_se_x100": round(float(np.median(se)), 4)}
    write_artifact(OUT, {"_meta": build_meta("analysis/stop_se.py", GROUPS, study_root=STUDY, code=[__file__],
                                             inputs=[FROZEN]),
                         "about": "The paired standard error of real search steps, and the stopping rule's SE.", **out})
    print(json.dumps({"summary": out["summary"], "stop_se_x100": out["stop_se_x100"], "best_seed_to_best": whole}, indent=1))


if __name__ == "__main__":
    main()
