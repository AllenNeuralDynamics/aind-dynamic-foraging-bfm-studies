#!/usr/bin/env python
"""The test programs' mouse-average curves on study 15's headline sets and subjects (ADR 0008 amendment).

Study 15 draws, per statistic, the subjects' mean per bin against each model's (its Figure 5; per-bin
RMSE in Figure 6), on the subjects every model holds out: replay on the ``non-warm-up`` set over its
146 common subjects (``p1_replay_gaps.json``), rollout on the ``headline`` set over its 122
(``p2_rollout_gaps.json``). This recomputes the same curves for the programs scored by
``freeze_test.py`` from their kept tallies, with study 15's rule (its ``_per_bin``: each subject's
value where its own tally keeps the bin, then the mean over subjects), and requires the subjects' own
curves to equal study 15's, so the comparison is on the same trials. Writes ``test/curves.json``.

Needs the fingerprint library (foraging-models 392c504) on PYTHONPATH; reads files only:

    python analysis/freeze_test_curves.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
S15 = STUDY.parent / "15-behavioral-fingerprints" / "analysis"
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

P1, P2 = S15 / "p1_replay_gaps.json", S15 / "p2_rollout_gaps.json"
PROGRAMS = {"rank3_e3819885": HERE / "test" / "test" / "rank3_e3819885.json",
            "reference_miller_rhg": HERE / "test" / "test" / "reference_miller_rhg.json"}
OUT = HERE / "test" / "curves.json"
MODELS_SHA = "392c504406094b33407321b168173981b8af8291"


def curve(subjects: list[dict], mode: str, set_name: str, stat: str) -> dict:
    from aind_dynamic_foraging_models.fingerprints import OBS, PRED, from_json, kept_bins, values

    model, obs = [], []
    for s in subjects:
        t = s["tallies"]
        if mode == "replay":
            own = t["replay"].get(set_name)
            if own is None:
                continue
            own = from_json(own)
            m = values(own, stat, PRED)
        else:
            if "rollout" not in t or t["rollout_subject"].get(set_name) is None or t["rollout"].get(set_name) is None:
                continue
            own, rolled = from_json(t["rollout_subject"][set_name]), from_json(t["rollout"][set_name])
            m = values(rolled, stat)
        keep = kept_bins(own, stat)
        obs.append(np.where(keep, values(own, stat, OBS), np.nan))
        model.append(np.where(keep, m, np.nan))
    obs, model = np.asarray(obs), np.asarray(model)
    n = np.isfinite(obs).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mo, so = np.nanmean(model, axis=0), np.nanmean(obs, axis=0)
    return {"n_subjects": n.tolist(), "model": [None if k == 0 or not np.isfinite(x) else float(x) for k, x in zip(n, mo)],
            "subject": [None if k == 0 else float(x) for k, x in zip(n, so)]}


def main() -> None:
    ref = {"replay": json.loads(P1.read_text()), "rollout": json.loads(P2.read_text())}
    common = {"replay": None, "rollout": ref["rollout"]["common_subjects"]}
    out = {"sets": {m: ref[m]["headline_set"] for m in ref}, "programs": {}, "subject_check": {}}
    for name, path in PROGRAMS.items():
        run = json.loads(path.read_text())
        by_id = {s["subject_id"]: s for s in run["subjects"]}
        out["programs"][name] = {}
        for mode, r in ref.items():
            set_name = r["headline_set"]
            cells = r["models"]["Miller-RHG"]["two_level_em"][set_name]
            cells = cells.get("statistics", cells)
            n_common = cells[next(iter(cells))]["per_bin_common"]["n_subjects"]
            if mode == "replay":   # study 15's replay common set: the test subjects the GRU holds out
                gru_rows = json.loads((STUDY.parent / "01-gru-scaling-law" / "analysis" / "per_subject_rows.json")
                                      .read_text())["rows"]
                gru = {x["subject"] for x in gru_rows if x["variant"] == "v2" and x["ratio"] == 1.0}
                ids = [sid for sid in r["subjects"] if sid in gru]
            else:
                ids = common[mode]
            subs = [by_id[s] for s in ids]
            out["programs"][name][mode] = {}
            for stat, cell in cells.items():
                c = curve(subs, mode, set_name, stat)
                theirs = cell["per_bin_common"]
                diff = max((abs(a - b) for a, b in zip(c["subject"], theirs["subject"]) if a is not None and b is not None),
                           default=0.0)
                same_n = c["n_subjects"] == theirs["n_subjects"]
                out["subject_check"][f"{name}|{mode}|{stat}"] = {"max_abs_diff": diff, "same_n_subjects": same_n}
                assert diff < 1e-9 and same_n, f"{name} {mode} {stat}: subject curve differs from study 15 ({diff})"
                out["programs"][name][mode][stat] = c
            out["programs"][name][mode + "_n_common"] = len(subs)
        print(name, "ok", {m: out["programs"][name][m + "_n_common"] for m in ref})
    meta = build_meta(f"analysis/{Path(__file__).name}", [], study_root=STUDY, kind="freeze", code=[__file__],
                      inputs=[P1, P2, *PROGRAMS.values(),
                              STUDY.parent / "01-gru-scaling-law" / "analysis" / "per_subject_rows.json"],
                      sources=[{"type": "url", "id": "https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/commit/"
                                + MODELS_SHA}])
    write_artifact(OUT, {"_meta": meta, **out})


if __name__ == "__main__":
    main()
