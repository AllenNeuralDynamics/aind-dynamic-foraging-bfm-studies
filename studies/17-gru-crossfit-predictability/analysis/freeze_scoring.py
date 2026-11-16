#!/usr/bin/env python
"""Per-mouse held-out scores of every scored model (variant `scoring`), frozen for the reports.

Reads each launch's ``jobs.csv`` (variants/scoring/launch_record/*/jobs.csv) and the scorer's outputs
under ``$S17_SCORING/out/<model>/pass<A|B>_lam-<mode>/`` (wrapper ``heldout_rescore``:
``mouse_scores.csv``, ``trial_predictions.csv.gz``, ``manifest.json``) and writes
``analysis/scores.json``: per model, its fold, D, way, selected checkpoint and lambda; per pass and
lambda mode, every held-out mouse's scored left/right (engaged) and 3-way log-likelihood sums and
trial counts, plus the ignore-class PR-AUC and recall (p_ignore >= 0.5) over the scored trials (3-way models). Each
job's ``manifest.json`` sha256 is recorded as a source. A job whose output is missing is listed
under ``missing``. Needs pandas; run on a compute node:

    make -C studies/17-gru-crossfit-predictability freeze-scoring PY=<mevolve-search env>/bin/python
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, file_digest, write_artifact  # noqa: E402

VARIANT = STUDY / "variants" / "scoring"
ROOT = Path(os.environ.get("S17_SCORING", "/allen/aind/scratch/han.hou/s17_scoring"))
OUT = HERE / "scores.json"
IGNORE = 2  # response code of an ignored trial in the per-trial export


def ignore_metrics(trials: pd.DataFrame) -> dict:
    """Ignore-class PR-AUC (average precision of p_ignore) and recall at p_ignore >= 0.5, as the
    wrapper's checkpoint metrics (gru_trainer engage_ignore_*), so studies 01/02 compare directly."""
    t = trials[trials["use"] == "scored"]
    y = (t["response"].to_numpy() == IGNORE).astype(int)
    p = t["p_ignore"].to_numpy()
    order = np.argsort(-p, kind="mergesort")
    hits = np.cumsum(y[order])
    precision = hits / np.arange(1, len(y) + 1)
    pr_auc = float(np.sum(precision[y[order] == 1]) / max(y.sum(), 1))
    pred = p >= 0.5
    recall = float((pred & (y == 1)).sum() / max(y.sum(), 1))
    return {"ignore_pr_auc": pr_auc, "ignore_recall": recall, "ignore_base_rate": float(y.mean())}


def main() -> None:
    models = {m["name"]: m for m in yaml.safe_load((VARIANT / "models.yaml").read_text())["models"]}
    jobs_files = sorted(VARIANT.glob("launch_record/*/jobs.csv"))
    out: dict = {}
    missing, sources = [], []
    for jf in jobs_files:
        for j in csv.DictReader(open(jf)):
            d = ROOT / "out" / j["out"]
            manifest = d / "manifest.json"
            if not manifest.exists():
                missing.append(j["out"])
                continue
            sources.append({"type": "file", "id": str(manifest), "sha256": file_digest(manifest)})
            m = models[j["name"]]
            entry = out.setdefault(j["name"], {
                "heldout": m["heldout"], "D": m["D"], "way": m["way"], "seed": m["seed"],
                "source": f"{m['entity']}/{m['project']}/{m['run_id']}",
                "best_eval_step": int(j["best_eval_step"]), "checkpoint_lambda": float(j["checkpoint_lambda"]),
                "scores": {}})
            ms = pd.read_csv(d / "mouse_scores.csv", dtype={"subject_id": str})
            ms = ms[ms["use"] == "scored"]
            rec = {"mice": {r.subject_id: [r.ll_lr_sum, int(r.n_engaged), r.ll_3way_sum, int(r.n_trials_3way)]
                            for r in ms.itertuples()}}
            if m["way"] == "3way":
                rec.update(ignore_metrics(pd.read_csv(d / "trial_predictions.csv.gz")))
            entry["scores"].setdefault(f"lam-{j['session_lambda']}", {})[j["pass"]] = rec
    meta = build_meta("analysis/freeze_scoring.py", [], study_root=STUDY, kind="freeze", code=[__file__],
                      inputs=[VARIANT / "models.yaml", *jobs_files], sources=sources)
    write_artifact(OUT, {
        "_meta": meta,
        "columns": ["ll_lr_sum", "n_engaged", "ll_3way_sum", "n_trials_3way"],
        "note": "scored half only; pass A tunes on fit and scores score, pass B the reverse",
        "missing": missing,
        "models": out,
    })


if __name__ == "__main__":
    main()
