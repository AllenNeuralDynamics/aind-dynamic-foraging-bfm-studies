#!/usr/bin/env python
"""P1: replay fingerprints of study 01's v1 GRU at D = 614, from its held-out per-trial predictions.

Variant ``gru-heldout-replay`` re-fine-tuned the GRU's three seeds on the held-out subjects with
study 01's own held-out config, pinned to the MEvolve freeze's cohort (snapshot 20260603), and saved
each scored trial's choice probabilities (``test_trial_predictions.csv``). The GRU's p(right) on a
trial is ``P(right) / (P(left) + P(right))``, as ``fingerprints.replay`` renormalises.

Before a subject's tallies count, per seed:

- its scored sessions are the freeze's score sessions, in order, with the same choices (exact);
- the CSV's log likelihood is the log of its own probability of the chosen side (``TOL_SELF``);
- its likelihood, ``exp(mean log likelihood)``, equals study 01 r2's stored value for that seed
  (``per_subject_rows.json``, which ``cross_study_literature.gru_per_subject`` reads without the
  seed) to ``TOL_LIK``, with the same number of trials.
  Study 01 scored on GPUs and this fine-tune ran on CPUs, so the bound is not 1e-6.

Study 01's held-out list differs from the freeze's 149 by 3 subjects each way (study 13 r2); those
3 have no stored value to check against (``subjects_unreferenced``). They are among the 614
subjects the source checkpoint was trained on (its ``subject_index_map.json``), so they are not
held out for the GRU: ``subjects_in_training`` lists them, and the gaps freeze leaves them out of
the GRU's numbers.

Writes ``p1_gru.json`` (the gates) and ``p1_replay/GRU-D614.parquet`` (the PRED tallies in the
format of ``freeze_p1_replay.py``, one regime per seed: ``seed0``, ``seed1``, ``seed2``).

Needs /allen and the mevolve env; on a compute node:

    sbatch -p aind --ntasks=1 --cpus-per-task=56 --mem=128G --time=01:00:00 --wrap \\
      "env PYTHONPATH=<foraging-models>/src:<mevolve>/src <mevolve env>/bin/python \\
       analysis/freeze_p1_gru.py --models-repo <foraging-models> --outputs <launch outputs> \\
       --prior-root <source runs>"
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
sys.path.insert(0, str(HERE))
import cross_study_literature as lit  # noqa: E402
from _meta import build_meta, write_artifact  # noqa: E402
from freeze_p0_golden_mevolve import FREEZE, FREEZE_MANIFEST, _git_sha  # noqa: E402
from freeze_p1_replay import _pred, _sha256, pred_frame  # noqa: E402

MODEL = "GRU-D614"
SEEDS = (0, 1, 2)
TOL_LIK, TOL_SELF = 1e-4, 1e-6
GROUPS = HERE / "p1_session_groups.json"
LAUNCH = (HERE.parent / "variants" / "gru-heldout-replay" / "launch_record" / "gru-replay-hpc-snap"
          / "hpc_20261104-150009.json")
OUT = HERE / "p1_gru.json"
OUT_TALLIES = HERE / "p1_replay" / f"{MODEL}.parquet"
COLUMNS = ["subject_id", "source_ses_idx", "trial", "choice", "probability_choice_0",
           "probability_choice_1", "log_likelihood_nats"]


def _csv(outputs: Path, seed: int) -> Path:
    found = sorted((outputs / f"s{seed}" / "heldout").rglob("test_trial_predictions.csv"))
    assert len(found) == 1, f"seed {seed}: {found}"
    return found[0]


def _stored(seed: int) -> dict[str, tuple[float, int]]:
    """Study 01 r2: ``{subject: (likelihood, n_trials)}`` of the v1 D = 614 GRU, one seed."""
    rows = json.loads(lit.GRU_PER_SUBJECT_SOURCE.read_text())["rows"]
    return {str(r["subject"]): (float(r["ll"]), int(r["n_trials"])) for r in rows
            if r["variant"] == "v1" and r["ratio"] == 1.0 and r["seed"] == str(seed)}


def _trained_on(prior_root: Path, seed: int) -> tuple[Path, set[str]]:
    """The subjects the source checkpoint of ``seed`` was trained on."""
    found = sorted((prior_root / f"s{seed}" / "run").rglob("subject_index_map.json"))
    assert len(found) == 1, f"seed {seed}: {found}"
    return found[0], {str(k) for k in json.loads(found[0].read_text())["subject_id_to_index"]}


def _subject_job(args) -> dict:
    """One subject, one seed: the session and likelihood checks, and the tallies."""
    seed, sid, rows, sessions, labels, stored, in_training = args
    from aind_dynamic_foraging_models.fingerprints import tally_groups

    order = list(dict.fromkeys(rows["source_ses_idx"]))
    same_sessions = order == [s.session_id for s in sessions]
    p_right, same_choices, self_diff = [], same_sessions, 0.0
    for s in sessions if same_sessions else []:
        r = rows[rows["source_ses_idx"] == s.session_id]
        same_choices &= bool(np.all(np.diff(r["trial"].to_numpy()) > 0)) and np.array_equal(
            r["choice"].to_numpy(), s.choices)
        p0, p1 = r["probability_choice_0"].to_numpy(), r["probability_choice_1"].to_numpy()
        chosen = np.where(r["choice"].to_numpy() == 1, p1, p0)
        d = np.abs(np.log(chosen) - r["log_likelihood_nats"].to_numpy())
        self_diff = max(self_diff, float(np.max(np.where(np.isfinite(d), d, np.inf))))
        p_right.append(p1 / (p0 + p1))
    out = {"subject_id": sid, "seed": seed, "same_sessions": same_sessions,
           "same_choices": bool(same_choices), "max_self_diff": self_diff,
           "likelihood": float(np.exp(rows["log_likelihood_nats"].mean())), "n_trials": int(len(rows)),
           "in_training": in_training}
    if stored is not None:
        out["likelihood_diff"] = out["likelihood"] - stored[0]
        out["n_trials_equal"] = out["n_trials"] == stored[1]
    if same_sessions and same_choices:
        out["pred"] = _pred(tally_groups([s.choices for s in sessions], [s.rewards for s in sessions],
                                         labels, p_right))
    return out


def _gate(rows: list[dict]) -> dict:
    """Per seed: how many subjects pass every check, and the largest misses."""
    def differs(r):
        if not (r["same_sessions"] and r["same_choices"] and r["max_self_diff"] <= TOL_SELF):
            return True
        if "likelihood_diff" not in r:
            return False   # no stored value (unreferenced), checked on its sessions only
        return not (abs(r["likelihood_diff"]) <= TOL_LIK and r["n_trials_equal"])

    ref = [r for r in rows if "likelihood_diff" in r]
    out = {"n_subjects": len(rows), "n_subjects_referenced": len(ref),
           "subjects_unreferenced": sorted(r["subject_id"] for r in rows if "likelihood_diff" not in r),
           "subjects_in_training": sorted(r["subject_id"] for r in rows if r["in_training"]),
           "max_abs_likelihood_diff": float(max(abs(r["likelihood_diff"]) for r in ref)),
           "median_abs_likelihood_diff": float(np.median([abs(r["likelihood_diff"]) for r in ref])),
           "max_self_diff": float(max(r["max_self_diff"] for r in rows)),
           "subjects_replay_differs": sorted(r["subject_id"] for r in rows if differs(r))}
    out["n_subjects_replay_differs"] = len(out["subjects_replay_differs"])
    out["passed"] = out["n_subjects_replay_differs"] == 0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-repo", type=Path, required=True)
    ap.add_argument("--outputs", type=Path, required=True, help="the launch's output root")
    ap.add_argument("--prior-root", type=Path, required=True, help="the source runs, s<seed>/run")
    ap.add_argument("--workers", type=int, default=56)
    a = ap.parse_args()
    import aind_dynamic_foraging_models
    import pandas as pd

    lib_path = Path(aind_dynamic_foraging_models.__file__).resolve()
    assert (a.models_repo / "src").resolve() in lib_path.parents, f"library from {lib_path}"
    models_sha = _git_sha(a.models_repo)
    from mevolve.data import Freeze
    from mevolve.workers import WORKER_ENV

    from aind_dynamic_foraging_models.fingerprints import SPEC_VERSION

    launch = json.loads(LAUNCH.read_text())
    launch_id = launch["wandb_group"].split("@")[1]
    assert a.outputs.resolve().name == launch_id, "outputs of another launch"
    freeze = Freeze(FREEZE)
    assert _sha256(FREEZE / "manifest.json") == FREEZE_MANIFEST, "another data freeze"
    groups = json.loads(GROUPS.read_text())
    assert groups["freeze_manifest_sha256"] == FREEZE_MANIFEST
    subjects = freeze.subjects("heldout")
    sessions = {sid: freeze.subject_sessions(sid, "score") for sid in subjects}
    labels = {sid: [groups["groups"][s.session_id] for s in ss] for sid, ss in sessions.items()}

    os.environ.update(WORKER_ENV)
    gates, sources, by_seed = {}, [{"type": "file", "id": str(FREEZE / "manifest.json"),
                                    "sha256": FREEZE_MANIFEST}], {}
    for seed in SEEDS:
        path = _csv(a.outputs, seed)
        sources.append({"type": "file", "id": str(path), "sha256": _sha256(path)})
        df = pd.read_csv(path, usecols=COLUMNS, dtype={"subject_id": str})
        assert set(df["subject_id"]) == set(subjects), f"seed {seed}: not the freeze's held-out subjects"
        per = dict(tuple(df.groupby("subject_id", sort=False)))
        stored = _stored(seed)
        index, trained = _trained_on(a.prior_root, seed)
        sources.append({"type": "file", "id": str(index), "sha256": _sha256(index)})
        jobs = [(seed, sid, per[sid], sessions[sid], labels[sid], stored.get(sid), sid in trained)
                for sid in subjects]
        with mp.get_context("spawn").Pool(min(a.workers, len(jobs))) as pool:
            rows = pool.map(_subject_job, jobs, chunksize=1)
        gates[f"seed{seed}"] = _gate(rows)
        by_seed[f"seed{seed}"] = rows
        print(seed, {k: v for k, v in gates[f"seed{seed}"].items() if not k.startswith("subjects")},
              flush=True)

    passed = all(g["passed"] for g in gates.values())
    outputs = []
    if all("pred" in r for rows in by_seed.values() for r in rows):   # else nothing to tally
        preds = {reg: {r["subject_id"]: r["pred"] for r in rows} for reg, rows in by_seed.items()}
        OUT_TALLIES.parent.mkdir(exist_ok=True)
        pred_frame([{"subject_id": sid, **{reg: {"pred": p[sid]} for reg, p in preds.items()}}
                    for sid in subjects], list(preds)).to_parquet(OUT_TALLIES, index=False)
        outputs.append(OUT_TALLIES)
    meta = build_meta("analysis/freeze_p1_gru.py", [], study_root=HERE.parent, kind="freeze",
                      code=[__file__, HERE / "freeze_p0_golden_mevolve.py", HERE / "freeze_p1_replay.py"],
                      inputs=[GROUPS, LAUNCH, lit.GRU_PER_SUBJECT_SOURCE], sources=sources)
    write_artifact(OUT, {
        "_meta": meta,
        "fingerprints_library": {"repo": "AllenNeuralDynamics/aind-dynamic-foraging-models",
                                 "git_sha": models_sha},
        "spec": SPEC_VERSION,
        "model": MODEL,
        "launch": {"wandb_group": launch["wandb_group"], "launch_id": launch_id},
        "tolerance": {"likelihood": TOL_LIK, "self": TOL_SELF},
        "passed": passed,
        "seeds": gates,
        "subjects": [{"subject_id": r["subject_id"], "seed": r["seed"], "likelihood": r["likelihood"],
                      "n_trials": r["n_trials"], "in_training": r["in_training"]}
                     for rows in by_seed.values() for r in rows],
    }, outputs=outputs)
    print(f"{'PASS' if passed else 'FAIL'} -> {OUT}")


if __name__ == "__main__":
    main()
