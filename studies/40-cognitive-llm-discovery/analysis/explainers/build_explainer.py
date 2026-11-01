"""Fill the explainer's reference numbers from committed JSON, so that nobody retypes them.

Rewrites the block between ``// BEGIN REFS`` and ``// END REFS`` in
``study40_animated_explainer.html`` in place. Run from anywhere; reads only committed files.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parents[3]   # analysis/explainers/
sys.path.insert(0, str(REPO))
from studies.util import cross_study  # noqa: E402

HTML = HERE / "study40_animated_explainer.html"
S13_CTT = REPO / "studies" / "13-hb-subject-vs-gru-heldout" / "analysis" / "r5_ctt_pooled.json"
SPLIT = REPO / "studies" / "10-gru-reservoir-ablation" / "reference" / "study01-v2-exact-split.json"


def refs() -> dict:
    s13 = json.loads(S13_CTT.read_text())
    per_ratio = json.loads(cross_study.GRU_CURVE_SOURCE.read_text())["cell_level"]["per_ratio"]
    full = max(per_ratio.values(), key=lambda c: c["D_mean"])          # D = 614, the full pool
    split = json.loads(SPLIT.read_text())
    rl_meta = json.loads(cross_study.RL_SOURCE.read_text())["_meta"]
    gru_meta = json.loads(cross_study.GRU_CURVE_SOURCE.read_text())["_meta"]
    return {
        "chance": 0.5,
        "mle_ctt": cross_study.rl_baseline("ctt"),
        "hb_ctt": round(s13["hb_ctt"], 4),
        "gru_v2_h128": round(full["v2"], 4),
        "n_train_mice": len(split["source_subject_ids"]),
        "n_test_mice": len(split["test_subject_ids"]),
        "n_test_sessions": s13["n_sessions"],
        "n_test_trials": s13["n_trials"],
        "sources": [
            {"value": "mle_ctt", "file": str(cross_study.RL_SOURCE.relative_to(REPO)),
             "produced_at_pt": rl_meta.get("produced_at_pt")},
            {"value": "hb_ctt, n_test_sessions, n_test_trials", "file": str(S13_CTT.relative_to(REPO)),
             "produced_at_pt": s13["_meta"].get("produced_at_pt")},
            {"value": "gru_v2_h128", "file": str(cross_study.GRU_CURVE_SOURCE.relative_to(REPO)),
             "produced_at_pt": gru_meta.get("produced_at_pt")},
            {"value": "n_train_mice, n_test_mice", "file": str(SPLIT.relative_to(REPO))},
        ],
    }


def main() -> None:
    block = "// BEGIN REFS\nconst REFS = " + json.dumps(refs(), indent=1) + ";\n// END REFS"
    text = HTML.read_text()
    new, n = re.subn(r"// BEGIN REFS.*?// END REFS", lambda _: block, text, flags=re.S)
    if n != 1:
        raise SystemExit("REFS markers not found exactly once")
    HTML.write_text(new)
    print(block)


if __name__ == "__main__":
    main()
