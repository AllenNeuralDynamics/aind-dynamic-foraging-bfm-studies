#!/usr/bin/env python
"""Session groups and gym task specs for a cohort's score sessions (study 40 ADR 0008; study 15 ADR 0006).

Study 15 built both for the 149 test subjects only. This applies its rules, unchanged, to any cohort of
``reference/cohorts_seed0.json`` (``generation`` for the pilot, ``selection`` for ADR 0008's selection),
importing its helpers rather than copying them:

- **group**: ``warm-up`` if every trial has a reward probability of 0 on one side or only 0 and 1, else
  ``{stage_group}|{task}`` (study 15's ``freeze_p1_session_groups``);
- **task spec**: the session's most common block parameters, reward pairs pooled over the subject's
  sessions of the same task and stage, engaged trials; sessions with no block parameters, coupled pools
  with no unequal pair and uncoupled sessions with one probability are excluded (study 15's
  ``freeze_p2_tasks``, its gym traps).

``--check-study15`` builds the 149 test subjects' groups and specs and requires them to equal study 15's
committed ``p1_session_groups.json`` and ``p2_task_specs.json``, so the rules are the same code path.

Needs /allen (database snapshot) and duckdb, so the dynamic-foraging-bfm-cpu env; on a compute node:

    sbatch -p aind --ntasks=1 --cpus-per-task=4 --mem=32G --time=01:00:00 --wrap \\
      "<dynamic-foraging-bfm-cpu env>/bin/python analysis/freeze_selection_tasks.py --cohort selection"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
S15 = STUDY.parent / "15-behavioral-fingerprints" / "analysis"
sys.path.insert(0, str(STUDY.parent / "util"))
sys.path.insert(0, str(S15))
from _meta import build_meta, write_artifact  # noqa: E402
from freeze_p0_task_checks import FREEZE, FREEZE_MANIFEST, SNAPSHOT, load_trials  # noqa: E402
from freeze_p1_session_groups import stage_group  # noqa: E402
from freeze_p2_tasks import _mode, _shares  # noqa: E402
from p2_task import MAX_BLOCK_TALLY, PERSEVERATIVE_LIMIT  # noqa: E402

COHORTS = STUDY / "reference" / "cohorts_seed0.json"
OUT_DIR = HERE / "selection"


def groups_and_specs(sessions: pd.DataFrame) -> tuple[dict, list, list, str]:
    """Study 15's group rule and task-spec rule for these subjects' sessions (fit and score rows)."""
    trials, db_path = load_trials(sessions)
    by_session = dict(tuple(trials.groupby("session_id", sort=False)))
    score = sessions[sessions["role"] == "score"]
    groups = {}
    for r in score.itertuples(index=False):
        p = by_session[r.session_id][["reward_probabilityL", "reward_probabilityR"]].to_numpy(float)
        warm_up = bool(np.isin(p, (0.0, 1.0)).all()) or bool((p.min(axis=1) == 0).all())
        groups[r.session_id] = "warm-up" if warm_up else f"{stage_group(r.stage)}|{r.task}"
    pools = defaultdict(set)
    for r in sessions.itertuples(index=False):
        if r.task.startswith("Coupled") and groups.get(r.session_id) != "warm-up":
            t = by_session[r.session_id]
            pools[(r.subject_id, r.task, str(r.stage))] |= _shares(
                t[["reward_probabilityL", "reward_probabilityR"]].to_numpy(float))
    specs, excluded = [], []
    for r in score.itertuples(index=False):
        t = by_session[r.session_id]
        engaged = int((t["animal_response"].to_numpy(float) != 2).sum())
        group = groups[r.session_id]
        bmin, v1 = _mode(t["block_min"].to_numpy(float))
        bmax, v2 = _mode(t["block_max"].to_numpy(float))
        beta, v3 = _mode(t["block_beta"].to_numpy(float))
        min_reward, _ = _mode(t["min_reward_each_block"].to_numpy(float))
        base, _ = _mode(t["base_reward_probability_sum"].to_numpy(float))
        spec = {"subject_id": r.subject_id, "session_id": r.session_id, "task": r.task, "group": group,
                "stage": None if r.stage is None or pd.isna(r.stage) else str(r.stage), "index": int(r.index),
                "baited": "Baiting" in r.task and "Without" not in r.task, "n_trials": engaged,
                "block_min": bmin, "block_max": bmax, "block_beta": beta,
                "min_reward_each_block": min_reward, "parameters_vary": v1 or v2 or v3}
        p = t[["reward_probabilityL", "reward_probabilityR"]].to_numpy(float)
        head = {k: spec[k] for k in ("subject_id", "session_id", "task", "group")}
        if bmin is None or bmax is None or (beta is None and r.task.startswith("Coupled")):
            excluded.append({**head, "reason": "no recorded block parameters"})
            continue
        if group == "warm-up":
            spec.update(kind="warm-up", reward_pairs=[[0.0, base if base else 1.0]])
        elif r.task.startswith("Coupled"):
            shares = sorted(pools[(r.subject_id, r.task, str(r.stage))])
            if not shares or base is None or shares == [0.5]:
                excluded.append({**head, "reason": "no recorded unequal reward pair or base sum"})
                continue
            spec.update(kind="coupled", base_reward_probability_sum=base,
                        reward_pairs=[[round((1 - s) * base, 6), round(s * base, 6)] for s in shares])
        elif r.task.startswith("Uncoupled"):
            probs = sorted(np.round(np.unique(p), 3).tolist())
            if len(probs) < 2:
                excluded.append({**head, "reason": f"one reward probability ({probs}), so no uncoupled schedule"})
                continue
            spec.update(kind="uncoupled", reward_probabilities=probs)
        else:
            excluded.append({**head, "reason": f"no gym task for {r.task}"})
            continue
        specs.append(spec)
    return groups, specs, excluded, db_path


def _check_study15(sessions: pd.DataFrame) -> dict:
    held = sessions[sessions["pool"] == "heldout"]
    groups, specs, excluded, _ = groups_and_specs(held)
    ref_groups = json.loads((S15 / "p1_session_groups.json").read_text())["groups"]
    ref_specs = json.loads((S15 / "p2_task_specs.json").read_text())
    same_specs = json.dumps(specs, sort_keys=True) == json.dumps(ref_specs["sessions"], sort_keys=True)
    same_excl = json.dumps(excluded, sort_keys=True) == json.dumps(ref_specs["excluded"], sort_keys=True)
    out = {"groups_equal": groups == ref_groups, "specs_equal": same_specs, "excluded_equal": same_excl,
           "n_specs": len(specs), "n_excluded": len(excluded)}
    print(out)
    assert all(out[k] for k in ("groups_equal", "specs_equal", "excluded_equal")), "differs from study 15"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["generation", "selection"])
    ap.add_argument("--check-study15", action="store_true")
    a = ap.parse_args()
    manifest = FREEZE / "manifest.json"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == FREEZE_MANIFEST
    sessions = pd.read_parquet(FREEZE / "sessions.parquet")
    if a.check_study15:
        _check_study15(sessions)
        return
    cohorts = json.loads(COHORTS.read_text())["cohorts"]
    subjects = set(cohorts[a.cohort])
    mine = sessions[sessions["subject_id"].isin(subjects)].copy()
    assert mine["subject_id"].nunique() == len(subjects)
    groups, specs, excluded, db_path = groups_and_specs(mine)
    counts = Counter(s["kind"] for s in specs)
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"tasks_{a.cohort}.json"
    meta = build_meta(f"analysis/{Path(__file__).name}", [], study_root=STUDY, kind="freeze",
                      code=[__file__, S15 / "freeze_p0_task_checks.py", S15 / "freeze_p1_session_groups.py",
                            S15 / "freeze_p2_tasks.py", S15 / "p2_task.py"],
                      inputs=[COHORTS],
                      sources=[{"type": "file", "id": str(manifest), "sha256": FREEZE_MANIFEST},
                               {"type": "file", "id": db_path, "state": f"database snapshot {SNAPSHOT}"}])
    write_artifact(out, {
        "_meta": meta, "cohort": a.cohort, "n_subjects": len(subjects),
        "freeze_manifest_sha256": FREEZE_MANIFEST,
        "rules": {"perseverative_limit": PERSEVERATIVE_LIMIT, "max_block_tally": MAX_BLOCK_TALLY,
                  "source": "study 15 freeze_p1_session_groups and freeze_p2_tasks, applied unchanged"},
        "n_score_sessions": len(groups), "n_specs": len(specs), "n_excluded": len(excluded),
        "n_subjects_with_specs": len({s["subject_id"] for s in specs}), "by_kind": dict(counts),
        "groups": dict(sorted(groups.items())), "excluded": excluded, "sessions": specs,
    })
    print({k: v for k, v in json.loads(out.read_text()).items() if k not in ("groups", "excluded", "sessions", "_meta")})


if __name__ == "__main__":
    main()
