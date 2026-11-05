#!/usr/bin/env python
"""The seeds of the seeded-islands searches, by ADR 0006's rule (``make seed-choice``).

Eligible: the literature models study 12 r1 ranks (``literature_ranking()``). Order: each model's
pooled score on study 40's 48 validation mice (``reference/cohorts_seed0.json``), from study 12 r3's
2-level EM fits (``two_level_em_per_subject()``), never study 12's own ranking, so the test mice
cannot influence the search. Seeds: the best three, and Eckstein-BI as the library's model with
discrete states, or the fourth-best model if Eckstein-BI is among the three. Writes
``analysis/seed_choice.json``; the search variants list these seeds in their ``run.json``, and their
job scripts check that the two agree.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
import cross_study_literature as lit  # noqa: E402
from _meta import build_meta, write_artifact  # noqa: E402

COHORTS = STUDY / "reference" / "cohorts_seed0.json"
OUT = HERE / "seed_choice.json"
STATES_MODEL = "Eckstein-BI"


def main() -> None:
    validation = [str(s) for s in json.loads(COHORTS.read_text())["cohorts"]["validation"]]
    ranked = lit.literature_ranking()["ranked"]
    em = lit.two_level_em_per_subject()
    rows = []
    for model in ranked:
        missing = [s for s in validation if s not in em[model]]
        assert not missing, f"{model} has no 2-level EM fit for {missing}"
        log_lik = sum(em[model][s][0] for s in validation)
        n_trials = sum(em[model][s][1] for s in validation)
        rows.append({"literature_model": model, "score": 100 * math.exp(log_lik / n_trials),
                     "n_subjects": len(validation), "n_trials": n_trials})
    rows.sort(key=lambda r: -r["score"])
    order = [r["literature_model"] for r in rows]
    seeds = order[:3] + [STATES_MODEL if STATES_MODEL not in order[:3] else order[3]]
    write_artifact(OUT, {
        "_meta": build_meta("analysis/seed_choice.py", [], study_root=STUDY, code=[__file__],
                            inputs=[COHORTS, lit.LEADERBOARD_SOURCE, lit.TWO_LEVEL_EM_SOURCE]),
        "rule": "ADR 0006: the three literature models with the best pooled 2-level EM score (study 12 r3) on "
                "the 48 validation mice, then Eckstein-BI, or the fourth-best if Eckstein-BI is among them",
        "score": "pooled likelihood x 100 over the validation mice's score sessions",
        "seeds": seeds, "ranking": rows})
    print("seeds:", seeds)


if __name__ == "__main__":
    main()
