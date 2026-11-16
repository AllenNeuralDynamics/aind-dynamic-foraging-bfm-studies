#!/usr/bin/env python
"""The GRU held-out map's index: where each model's per-trial predictions live, and two headline values.

Every scored model x pass x session-lambda mode (74) has one W&B artifact on its own scoring run,
`heldout-rescore-<run>:v1` (project NeuralDynamics/dfbfm_17-gru-crossfit-predictability), holding
`trial_predictions.csv.gz` (freeze trial numbering; `ll_lr` = log p(choice) renormalised over left
and right), `session_scores.csv`, `mouse_scores.csv`, the tuned subject embeddings, the roles and the
maps. Their digests are committed in the three `variants/scoring/launch_record/*/backfill_maps.json`
records; the artifact name, type and files were checked on W&B for all 74 (2026-11-16).

This producer is offline: it reads those records, `analysis/scores.json` and the fold and tier maps,
and writes `analysis/gru_map_index.json`:
- `artifacts`: one row per artifact (model, fold, D, way, seed, pass, scored role, lambda mode,
  checkpoint, run, group, artifact, digest), with `comparison: true` on the 10 that a model compared
  with MEvolve programs should use: the 5 fold models (full size, seed 0) per head, pass A (scores the
  freeze's `score` sessions, the sessions MEvolve scores), each checkpoint at its own lambda.
- `mice`: each of the 763 freeze mice's fold and study 40 tier (`reference/pool_strata.json`;
  the 149 freeze test mice are tier `test`). Study 41 may re-deal tiers; join its own on read.
- `reference`: per head, the held-out likelihood, trial-pooled, left/right on engaged trials, pass A,
  over the 5 fold models: exp(sum ll_lr / n engaged trials). The registry entries
  `gru_cv_map_3way` and `gru_cv_map_2way` (studies/util/references.py) point here.

    python analysis/gru_map_index.py   # writes analysis/gru_map_index.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

SCORES = HERE / "scores.json"
FOLDS = STUDY / "reference" / "folds.json"
STRATA = STUDY / "reference" / "pool_strata.json"
RECORDS = sorted((STUDY / "variants" / "scoring" / "launch_record").glob("*/backfill_maps.json"))
OUT = HERE / "gru_map_index.json"
ENTITY, PROJECT = "NeuralDynamics", "dfbfm_17-gru-crossfit-predictability"
# the fold models: full size, seed 0, one per fold and head (the test fold reuses studies 02 and 01)
FOLD_MODELS = {"3way": {"test": "test-3way-full-s0", **{f"pool_{k}": f"pool{k}-3way-full" for k in range(1, 5)}},
               "2way": {"test": "test-2way-full-s0", **{f"pool_{k}": f"pool{k}-2way-full" for k in range(1, 5)}}}
# freeze test mice in study 01's 2-way training set: no held-out 2-way prediction (r1, 760 mice)
NO_2WAY = ["722683", "795395", "820243"]


def main() -> None:
    scores = json.loads(SCORES.read_text())
    folds = json.loads(FOLDS.read_text())
    strata = json.loads(STRATA.read_text())["strata"]
    models = scores["models"]
    fold_models = {m for per in FOLD_MODELS.values() for m in per.values()}

    rows = []
    for rec_path in RECORDS:
        rec = json.loads(rec_path.read_text())
        for t in rec["tasks"]:
            assert t["status"] in ("backfilled", "skipped (v1 exists)"), t
            model, mode = t["orig_task"].split("/")
            pass_, lam = mode.split("_", 1)                      # "passA", "lam-checkpoint"
            m = models[model]
            rows.append({
                "model": model, "fold": "test" if m["heldout"] == "test" else m["heldout"],
                "D": str(m["D"]), "way": m["way"], "seed": m["seed"], "pass": pass_[-1],
                "scored_role": "score" if pass_ == "passA" else "fit",
                "session_lambda": lam.removeprefix("lam-"),        # "checkpoint" or "1"
                "checkpoint_step": m["best_eval_step"], "checkpoint_lambda": m["checkpoint_lambda"],
                "fold_model": model in fold_models,
                "comparison": model in fold_models and pass_ == "passA" and lam == "lam-checkpoint",
                "run": t["new_wandb_run"], "group": rec["wandb_group"],
                "artifact": f"heldout-rescore-{t['new_wandb_run']}:v1", "digest": t["v1_digest"],
                "record": rec_path.relative_to(STUDY).as_posix()})
    rows.sort(key=lambda r: (r["model"], r["pass"], r["session_lambda"]))
    keys = [(r["model"], r["pass"], r["session_lambda"]) for r in rows]
    assert len(rows) == 74 and len(set(keys)) == 74, len(rows)
    assert sum(r["session_lambda"] == "1" for r in rows) == 10
    assert sum(r["comparison"] for r in rows) == 10

    mice = {s: {"fold": f, "tier": "test" if f == "test" else strata[s]["tier"]}
            for s, f in sorted(folds["fold_of"].items())}
    assert len(mice) == 763

    reference = {}
    for way, per_fold in FOLD_MODELS.items():
        ll = n = 0.0
        seen = set()
        for fold, model in per_fold.items():
            m = models[model]
            assert (m["way"], str(m["D"]), m["seed"]) == (way, "full", 0), model
            for s, (ll_lr, n_eng, _, _) in m["scores"]["lam-checkpoint"]["A"]["mice"].items():
                assert mice[s]["fold"] == fold and s not in seen, (model, s)
                seen.add(s)
                ll += ll_lr
                n += n_eng
        missing = sorted(set(mice) - seen)
        assert missing == ([] if way == "3way" else NO_2WAY), (way, missing)
        reference[way] = {"pooled_lr_likelihood": math.exp(ll / n), "n_mice": len(seen), "n_models": len(per_fold),
                          "n_engaged_trials": int(n), "models": per_fold, "mice_without_prediction": missing}

    meta = build_meta("analysis/gru_map_index.py", [], study_root=STUDY, code=[__file__],
                      inputs=[SCORES, FOLDS, STRATA, *RECORDS])
    write_artifact(OUT, {
        "_meta": meta,
        "wandb": {"entity": ENTITY, "project": PROJECT, "artifact_type": "heldout-rescore",
                  "files": ["manifest.json", "mouse_scores.csv", "roles.csv", "session_context_map.json",
                            "session_scores.csv", "subject_embeddings.csv", "subject_index_map.json",
                            "trial_predictions.csv.gz"]},
        "freeze": {"snapshot": folds["snapshot"], "manifest_sha256": folds["freeze_manifest_sha256"]},
        "estimand": "held-out likelihood, trial-pooled: exp(sum ll_lr / n) over engaged trials of the "
                    "scored sessions; ll_lr = log p(choice) renormalised over left and right",
        "reference": reference,
        "artifacts": rows,
        "mice": mice,
    })


if __name__ == "__main__":
    main()
