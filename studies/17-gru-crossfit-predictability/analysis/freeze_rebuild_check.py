#!/usr/bin/env python
"""Freeze: the held-out model rebuild check (wrapper #129/#130) into a small committed JSON.

Reads, from /allen scratch:
  - backfill/rebuild_one_result.json: pool4-3way-full pass A rebuilt from W&B downloads only
    (heldout-rescore-ecuwfnwx:v1 + the source gru-output artifact) vs its logged predictions;
  - rebuild_check/fig_data_passA.npz: histogram of |rebuilt - logged p(right)| over every logged trial;
  - rebuild_check/fig_sample_passA.parquet: 20,000 random (logged, rebuilt) p pairs;
  - rebuild_check/fig_session_passA.parquet: the per-trial GRU states of one scored session.
The session's states are reduced here to their first two principal components (over the whole
session); the JSON keeps trials 900-1199 and the variance they explain. Each file's sha256 is a source.
Needs pandas + pyarrow: the mevolve-search env, on a compute node.

    python analysis/freeze_rebuild_check.py   # writes analysis/rebuild_check.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, file_digest, write_artifact  # noqa: E402

S = Path("/allen/aind/scratch/han.hou/s17_scoring")
FILES = {"result": S / "backfill/rebuild_one_result.json", "hist": S / "rebuild_check/fig_data_passA.npz",
         "sample": S / "rebuild_check/fig_sample_passA.parquet", "session": S / "rebuild_check/fig_session_passA.parquet"}
OUT = HERE / "rebuild_check.json"
N_SAMPLE, WINDOW = 2000, (900, 1200)


def main() -> None:
    result = json.loads(FILES["result"].read_text())
    h = np.load(FILES["hist"])
    smp = pd.read_parquet(FILES["sample"]).iloc[:N_SAMPLE]
    one = pd.read_parquet(FILES["session"])
    states = one[[c for c in one if c.startswith("rnn_state_")]].to_numpy()
    states = states - states.mean(0)
    _, s, vt = np.linalg.svd(states, full_matrices=False)
    pc = states @ vt[:2].T
    t = one["trial"].to_numpy()
    w = (t >= WINDOW[0]) & (t < WINDOW[1])
    out = {
        "model": "pool4-3way-full", "pass": "A", "scoring_run": "NeuralDynamics/dfbfm_17-gru-crossfit-predictability/ecuwfnwx",
        "artifact": "heldout-rescore-ecuwfnwx:v1",
        "wandb_only_rebuild": result,
        "abs_diff_p_right_hist": {"edges": h["edges"].tolist(), "counts": h["counts"].tolist(), "n_exact_zero": int(h["n_zero"])},
        "sample_p_right": {"logged": smp["p_right"].tolist(), "rebuilt": smp["p_model1_right"].tolist()},
        "session": {"subject_id": str(one["subject_id"].iloc[0]), "session_id": str(one["session_id"].iloc[0]),
                    "n_trials": int(len(one)), "pc12_variance_share": float((s[:2] ** 2).sum() / (s ** 2).sum()),
                    "window": list(WINDOW), "trial": t[w].tolist(), "action": one["action"].to_numpy()[w].astype(int).tolist(),
                    "pc1": pc[w, 0].round(5).tolist(), "pc2": pc[w, 1].round(5).tolist()},
    }
    meta = build_meta("analysis/freeze_rebuild_check.py", [], study_root=STUDY, kind="freeze", code=[__file__], inputs=[],
                      sources=[{"type": "file", "id": str(p), "sha256": file_digest(p)} for p in FILES.values()])
    write_artifact(OUT, {"_meta": meta, **out})


if __name__ == "__main__":
    main()
