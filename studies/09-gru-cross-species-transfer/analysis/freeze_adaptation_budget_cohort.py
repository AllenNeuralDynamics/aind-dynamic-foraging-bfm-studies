"""Freeze the adaptation-budget cohort and per-K adaptation sessions (dispatcher #166).

Run on a compute node before any budget result exists, so the cohort is fixed in advance:

    python -m external_bandit_datasets.cli grossman --cache-root "$DATA_ROOT"   # and chen
    BFM_WRAPPER_CODE=.../aind-dynamic-foraging-bfm-wrapper/code \\
        python analysis/freeze_adaptation_budget_cohort.py --data-root "$DATA_ROOT/canonical"

For Grossman (mouse) and Chen (mouse) it loads the canonical table and frozen split with the
wrapper's own ``ExternalBanditDatasetLoader`` and applies the wrapper's own
``apply_external_adaptation_budget`` at every K, so the recorded sessions are exactly the
ones the GRU and Q runs adapt on. Target test sessions are identical at every K; the file
records them once per subject.

Two cohorts, each a set of subjects for whom every K it covers is a genuine K-session fit
(no subject runs out of adaptation sessions):

  * ``primary`` -- >= 4 adaptation sessions; the shared K in {0, 1, 2, 4} curve. Every Chen
    (mouse) subject has exactly 4 adaptation sessions, so K=8 cannot exist there and Chen's
    K=4 is its full adaptation half.
  * ``k8`` -- >= 8 adaptation sessions; Grossman (mouse) only, for the K=8 point.

Writes ``analysis/adaptation_budget_cohort.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta  # noqa: E402

DATASETS = {"grossman": "grossman-bari-cohen-2021", "chen": "chen-et-al-2021"}
BUDGETS = (1, 2, 4, 8)   # K=0 adapts nothing; "half" adapts every adaptation session
PRIMARY_MIN = 4          # the primary cohort's minimum adaptation sessions
OUT = STUDY / "analysis" / "adaptation_budget_cohort.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sessions_by_subject(bundle) -> dict[str, list[str]]:
    """subject -> its adaptation session ids, in the packed (chronological) order."""
    subject_by_session = {
        str(session_id): str(row["subject_id"])
        for row in bundle.metadata["session_context"]["per_subject"]
        for session_id in row["ordered_session_ids"]
    }
    out: dict[str, list[str]] = {}
    for session_id in bundle.metadata["train_session_ids"]:
        out.setdefault(subject_by_session[str(session_id)], []).append(str(session_id))
    return out


def _freeze(dataset: str, data_root: Path) -> dict:
    from data_loaders.external_bandit import (ExternalBanditDatasetLoader,
                                              apply_external_adaptation_budget)

    table = data_root / f"{dataset}.parquet"
    manifest = data_root / f"{dataset}.split.json"
    bundle = ExternalBanditDatasetLoader(
        file_path=table, split_manifest_path=manifest, dataset_id=DATASETS[dataset],
        batch_size=None, batch_mode="single", adapt_sessions_per_subject=None, seed=0,
    ).load()
    adapt = _sessions_by_subject(bundle)
    split = json.loads(manifest.read_text())
    test = {str(row["subject_id"]): [str(s) for s in row["test_session_ids"]]
            for row in split["subjects"]}
    selected = {}
    for k in BUDGETS:
        capped = apply_external_adaptation_budget(bundle, adapt_sessions_per_subject=k)
        ids = set(capped.metadata["target_adaptation_session_ids"])
        selected[k] = {s: [x for x in sessions if x in ids] for s, sessions in adapt.items()}
        for s, sessions in adapt.items():   # the budget is a prefix of the adaptation sequence
            assert selected[k][s] == sessions[:k], (dataset, s, k)
    subjects = []
    for s, sessions in sorted(adapt.items()):
        subjects.append({
            "subject_id": s,
            "n_adapt_sessions": len(sessions),
            "n_test_sessions": len(test[s]),
            "in_primary_cohort": len(sessions) >= PRIMARY_MIN,
            "in_k8_cohort": len(sessions) >= 8,
            "adapt_session_ids": sessions,
            "adapt_session_ids_by_k": {str(k): selected[k][s] for k in BUDGETS},
            "test_session_ids": test[s],
        })
    cohorts = {"primary": [r["subject_id"] for r in subjects if r["in_primary_cohort"]],
               "k8": [r["subject_id"] for r in subjects if r["in_k8_cohort"]]}
    print(f"{dataset}: primary {len(cohorts['primary'])}/{len(subjects)}, "
          f"k8 {len(cohorts['k8'])}/{len(subjects)}")
    return {
        "dataset_id": DATASETS[dataset],
        "table_sha256": _sha256(table),
        "split_manifest_sha256": _sha256(manifest),
        "cohorts": cohorts,
        "subjects": subjects,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="the CLI's canonical/ directory holding <dataset>.parquet/.split.json")
    args = parser.parse_args()
    sys.path.insert(0, os.environ["BFM_WRAPPER_CODE"])
    out = {
        "_meta": build_meta("analysis/freeze_adaptation_budget_cohort.py", [], study_root=STUDY),
        "budgets_launched": {
            "grossman": {"gru": [0, 1, 2, 4, 8, "half"], "q": [1, 2, 4, 8, "half"]},
            "chen": {"gru": [0, 1, 2, "half"], "q": [1, 2, "half"]},
        },
        "cohort_rule": {"primary": f">= {PRIMARY_MIN} adaptation sessions; K in {{0,1,2,4}}",
                        "k8": ">= 8 adaptation sessions; K=8 (Grossman only)"},
        "datasets": {name: _freeze(name, args.data_root) for name in DATASETS},
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
