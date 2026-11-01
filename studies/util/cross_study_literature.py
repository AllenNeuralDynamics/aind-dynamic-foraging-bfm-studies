"""Per-subject numbers for comparisons with the literature models, read from committed files.

The companion of ``cross_study.py`` (same contract: a producer passes the matching ``*_SOURCE``
path in ``build_meta(..., inputs=[...])`` and names the source in its report's ``upstream``). It
is a separate module because ``build_meta`` digests every ``studies/util`` module a producer
imports: adding readers to ``cross_study.py`` would mark every artifact that imports it stale.

Per-subject values come back at full precision (pooled reference values at 4 decimals, as in
``cross_study.py``): they are inputs that a consumer re-pools over its own subject set (for
example study 40's cohorts).
"""
from __future__ import annotations

import csv
import json

from cross_study import REPO

_S01 = REPO / "studies" / "01-gru-scaling-law" / "analysis"
_S12 = REPO / "studies" / "12-cognitive-literature-leaderboard" / "analysis"
_S13 = REPO / "studies" / "13-hb-subject-vs-gru-heldout" / "analysis"

GRU_PER_SUBJECT_SOURCE = _S01 / "per_subject_rows.json"   # study 01 r2: per-held-out-subject GRU
HB_CTT_SOURCE = _S13 / "r5_ctt_pooled.json"    # study 13 r5: HB-CTT (and HB-Hattori) pooled
LITERATURE_SOURCE = _S12 / "per_subject_fits.csv"   # study 12 freeze: literature models per subject
GRU_SPLIT_SOURCE = (REPO / "studies" / "10-gru-reservoir-ablation" / "reference"
                    / "study01-v2-exact-split.json")   # study 01's GRU training pool / test list

_RATIO_TO_D = {0.016: 10, 0.049: 30, 0.163: 100, 0.489: 300, 1.0: 614}   # study 01's subject_ratio


def gru_per_subject(d: int = 614, variant: str = "v1") -> dict[str, list[tuple[float, int]]]:
    """Study 01 r2: the H128 GRU per held-out subject at nominal ``d``, full precision.

    ``{subject: [(likelihood, n_scored_trials) per seed]}``. Study 01's GRU held-out list differs
    from the 149 per-subject-fit subjects by 3 subjects each way (study 13 r2).
    """
    out: dict[str, list[tuple[float, int]]] = {}
    for row in json.loads(GRU_PER_SUBJECT_SOURCE.read_text())["rows"]:
        if row["variant"] == variant and _RATIO_TO_D[row["ratio"]] == d:
            out.setdefault(str(row["subject"]), []).append((float(row["ll"]), int(row["n_trials"])))
    return out


def hb_pooled(model: str = "ctt") -> float:
    """Study 13 r5: a subject-level hierarchical model's pooled held-out likelihood on the 149.

    ``ctt`` is HB-CTT (r5's headline); ``hattori`` is HB-Hattori2019 as r5 quotes it from r1.
    """
    return round(json.loads(HB_CTT_SOURCE.read_text())[{"ctt": "hb_ctt", "hattori": "hb_hattori"}[model]], 4)


def literature_per_subject(cohort: str | None = None) -> dict[str, dict[str, tuple[float, int]]]:
    """Study 12: each literature model's score per subject, full precision.

    ``{literature_model: {subject: (sum_log_p, n_scored_trials)}}`` over odd (eval) sessions, for
    ``cohort`` ``"train"`` (614 training pool), ``"heldout"`` (149) or ``None`` (all 763). The
    likelihood of any set of subjects is ``exp(Σ sum_log_p / Σ n_scored_trials)``.
    """
    out: dict[str, dict[str, tuple[float, int]]] = {}
    with LITERATURE_SOURCE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if cohort is None or row["cohort"] == cohort:
                out.setdefault(row["literature_model"], {})[row["subject_id"]] = (
                    float(row["eval_total_log_likelihood"]), int(row["eval_total_trials"]))
    return out



def literature_fitted_params(model: str) -> dict[str, dict[str, float]]:
    """Study 12: one literature model's fitted parameters per subject, ``{subject: {name: value}}``.

    All 763 subjects; each set was fit on the subject's even sessions (MEvolve's parity reference).
    """
    with LITERATURE_SOURCE.open(newline="") as handle:
        return {row["subject_id"]: json.loads(row["fitted_params"])
                for row in csv.DictReader(handle) if row["literature_model"] == model}


def gru_split() -> dict[str, list[str]]:
    """Study 01's GRU split of the 763 subjects: ``{"train": 614 ids, "test": 149 ids}``.

    3 subjects sit in the other pool each way relative to the per-subject fits' cohorts (study 13
    r2); per-subject scores do not depend on the pool, so any of these lists can be re-pooled.
    """
    split = json.loads(GRU_SPLIT_SOURCE.read_text())
    return {"train": [str(s) for s in split["source_subject_ids"]],
            "test": [str(s) for s in split["test_subject_ids"]]}
