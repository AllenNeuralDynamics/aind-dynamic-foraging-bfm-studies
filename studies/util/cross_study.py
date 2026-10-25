"""Numbers other studies publish, read from their committed JSON so that nobody retypes them.

A producer that compares against another study's result takes the number from here, passes
the matching ``*_SOURCE`` path in ``build_meta(..., inputs=[...])``, and its report names the
source report in ``upstream``. The checker then warns when that report is corrected or
replaced (report-upstream-stale) or when the declaration is missing
(report-upstream-undeclared).

Values come back at the 4 decimals study 01 publishes, which every downstream report was
written against; moving them to full precision shifts quoted gaps in the 4th decimal and is a
result change in its own right (see studies#20).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_S01 = REPO / "studies" / "01-gru-scaling-law" / "analysis"

GRU_CURVE_SOURCE = _S01 / "report_data.json"   # study 01 r1: the fixed-H128 GRU curve
RL_SOURCE = _S01 / "rl_baseline.json"          # study 01 r8: per-mouse RL baselines
GENERATIVE_SOURCE = _S01 / "generative_match.json"   # study 01 r9: GRU generative match

D_NOMINAL = (10, 30, 100, 300, 614)


def gru_h128_curve() -> dict[int, float]:
    """Study 01 r1: the fixed-H128 GRU's held-out likelihood (v1 runs, 3-seed mean) by nominal D."""
    per_ratio = json.loads(GRU_CURVE_SOURCE.read_text())["cell_level"]["per_ratio"]
    return {min(D_NOMINAL, key=lambda d: abs(d - c["D_mean"])): round(c["v1"], 4)
            for c in sorted(per_ratio.values(), key=lambda c: c["D_mean"])}


def rl_baseline(model: str = "ctt") -> float:
    """Study 01 r8: a per-mouse RL baseline's pooled, trial-weighted held-out likelihood.

    ``ctt`` (compare-to-threshold) is the best of the three; ``bari`` and ``hattori`` follow.
    """
    return round(json.loads(RL_SOURCE.read_text())["models"][model]["pooled_likelihood_trial_weighted"], 4)

