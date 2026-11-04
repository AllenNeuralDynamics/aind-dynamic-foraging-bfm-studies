#!/usr/bin/env python
"""P0 task checks for parameter-matched rollouts (ADR 0006), on the test subjects' sessions.

1. Held blocks. The real coupled task draws a block length as ``block_min + Exp(block_beta)``,
   capped at ``block_max`` (dynamic-foraging-task ``MyFunctions._generate_next_coupled_block``),
   then can hold the switch: ``BlockMinReward`` and ``AdvancedBlockAuto``
   (``_check_coupled_block_transition``). With ``AddOneTrialForNoresponse`` (not recorded) it
   also adds one trial to the block for each ignored trial (``_add_one_trial``). So a block's
   count of *engaged* trials exceeds ``block_max`` only if the block was held, whatever that
   setting was; that count is a lower bound on holds, and the count over all trials, also kept,
   an upper one. The gym has neither hold, so such a block is one a rollout cannot produce. This
   counts them in the score sessions of study 12's 149 test subjects (ADR 0007), by task and stage,
   and splits them by whether the session set a minimum reward (``min_reward_each_block`` > 0):
   where it did not, the unrecorded auto-block rule or an experimenter's change of settings
   held the block; the data cannot tell which.
2. Reward-pair pools. The pairs of a coupled session are ``RewardFamilies[family][:N]`` scaled to
   ``base_reward_probability_sum`` (``Foraging.py``); family and N are not recorded. ADR 0006 pools
   the pairs seen in the subject's sessions (fit and score) with the same task and stage. This
   checks each pool against the GUI's family prefixes, how much of its pool a session shows, and
   that every block's pair sums to the recorded base sum. Uncoupled sessions draw each side from
   a list of probabilities; their pooled lists are counted.
3. Coverage: score sessions missing ``block_min``, ``block_max``, ``block_beta``, and sessions
   whose recorded probabilities are only 0 and 1 (older sessions record which side is better, not
   the schedule); those carry no usable pairs and are left out of the pools.

Needs /allen (the database snapshot); on a compute node:

    srun --ntasks=1 --cpus-per-task=8 --mem=32G -p aind --time=00:30:00 \\
      <dynamic-foraging-bfm-cpu env>/bin/python analysis/freeze_p0_task_checks.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from aind_dynamic_foraging_database import trial_db

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

SNAPSHOT = "20260603"
FREEZE = Path("/allen/aind/scratch/han.hou/mevolve/freeze/20260603")
FREEZE_MANIFEST = "cfea3d1e21f49ed7baa42f11e1fdda66fd5901c519a47773c08716e062e7eef0"
FINAL_STAGES = ("STAGE_FINAL", "GRADUATED")
# Foraging.py RewardFamilies (dynamic-foraging-task main, read 2026-11-03), as pairs [high, low].
REWARD_FAMILIES = {
    1: [[8, 1], [6, 1], [3, 1], [1, 1]],
    2: [[8, 1], [1, 1]],
    3: [[1, 0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]],
    4: [[6, 1], [3, 1], [1, 1]],
}
COLUMNS = ["session_id", "trial", "animal_response", "reward_probabilityL", "reward_probabilityR", "block_min",
           "block_max", "block_beta", "min_reward_each_block", "base_reward_probability_sum"]
OUT = HERE / "p0_task_checks.json"


def _share(pair_hi_lo) -> float:
    hi, lo = pair_hi_lo
    return round(float(hi / (hi + lo)), 2)


FAMILY_PREFIXES = {(fam, n): frozenset(_share(p) for p in pairs[:n])
                   for fam, pairs in REWARD_FAMILIES.items() for n in range(1, len(pairs) + 1)}


def _stage_group(stage) -> str:
    if stage is None or stage == "None" or pd.isna(stage):
        return "unknown"
    return "final" if stage in FINAL_STAGES else "earlier"


def _blocks(p: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of each block of constant (pL, pR); the last one is incomplete, so dropped."""
    change = np.r_[True, np.any(p[1:] != p[:-1], axis=1)]
    starts = np.flatnonzero(change)
    return [(int(s), int(e - s)) for s, e in zip(starts[:-1], starts[1:])]


def load_trials(sessions: pd.DataFrame) -> pd.DataFrame:
    subjects = sorted(sessions["subject_id"].unique())
    path = trial_db(snapshot=SNAPSHOT)
    query = (f"SELECT {', '.join(COLUMNS)} FROM read_parquet('{path}/**/*.parquet', "
             "hive_partitioning=true, union_by_name=true) "
             f"WHERE CAST(subject_id AS VARCHAR) IN ({', '.join(repr(s) for s in subjects)})")
    trials = duckdb.sql(query).df()
    trials = trials[trials["session_id"].isin(set(sessions["session_id"]))]
    return trials.sort_values(["session_id", "trial"]), str(path)


def main() -> None:
    manifest = FREEZE / "manifest.json"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == FREEZE_MANIFEST
    expected = json.loads(manifest.read_text())["files"]["sessions.parquet"]
    actual = hashlib.sha256((FREEZE / "sessions.parquet").read_bytes()).hexdigest()
    assert actual == expected, f"sessions.parquet {actual} differs from the manifest's {expected}"
    meta_sessions = pd.read_parquet(FREEZE / "sessions.parquet")
    sessions = meta_sessions[meta_sessions["pool"] == "heldout"].copy()
    assert sessions["subject_id"].nunique() == 149
    trials, db_path = load_trials(sessions)
    by_session = dict(tuple(trials.groupby("session_id", sort=False)))
    missing = sorted(set(sessions["session_id"]) - set(by_session))
    assert not missing, f"{len(missing)} freeze sessions have no trials in the snapshot"

    holds = defaultdict(lambda: Counter())
    coverage = Counter()
    coupled_shares, sum_mismatch = {}, Counter()
    uncoupled_values = {}
    for row in sessions.itertuples(index=False):
        t = by_session[row.session_id]
        p = t[["reward_probabilityL", "reward_probabilityR"]].to_numpy(float)
        key = (row.task, row.stage if isinstance(row.stage, str) else "None")
        binary = bool(np.isin(p, (0.0, 1.0)).all())
        if row.role == "score":
            coverage["sessions"] += 1
            coverage["binary_probabilities"] += int(binary)
            for col in ("block_min", "block_max", "block_beta", "min_reward_each_block"):
                coverage[f"missing_{col}"] += int(t[col].isna().all())
        if row.task.startswith("Coupled"):
            blocks = _blocks(p)
            hi_lo = [(max(p[s]), min(p[s])) for s, _ in blocks] + [(max(p[-1]), min(p[-1]))]
            if not binary:
                coupled_shares[row.session_id] = frozenset(_share(x) for x in hi_lo if sum(x) > 0)
            base = t["base_reward_probability_sum"].to_numpy(float)
            if row.role == "score" and np.isfinite(base).all():
                sum_mismatch["checked"] += 1
                sum_mismatch["mismatched"] += int(np.any(np.abs(p.sum(1) - base) > 1e-6))
            if row.role != "score":
                continue
            bmax = t["block_max"].to_numpy(float)
            min_reward = t["min_reward_each_block"].to_numpy(float)
            sets_min = bool(np.nanmax(min_reward, initial=0) > 0)
            c = holds[key]
            c["sessions"] += 1
            c["sessions_with_min_reward"] += int(sets_min)
            engaged = (t["animal_response"].to_numpy(float) != 2).astype(int)
            for start, length in blocks:
                if not np.isfinite(bmax[start]):
                    c["blocks_no_block_max"] += 1
                    continue
                over = int(engaged[start:start + length].sum() > bmax[start])  # held, for sure
                c["blocks"] += 1
                c["over_max"] += over
                c["over_max_all_trials"] += int(length > bmax[start])
                c["over_max_with_min_reward" if sets_min else "over_max_without_min_reward"] += over
        elif not binary:
            uncoupled_values[row.session_id] = frozenset(np.round(np.unique(p), 3).tolist())

    hold_rows = []
    for (task, stage), c in sorted(holds.items()):
        hold_rows.append({"task": task, "stage": stage, "stage_group": _stage_group(stage),
                          **{k: int(v) for k, v in c.items()},
                          "share_over_max": c["over_max"] / c["blocks"] if c["blocks"] else None})
    totals = Counter()
    for c in holds.values():
        totals.update(c)

    # pools per (subject, task, stage) over all of the subject's sessions
    pools = defaultdict(frozenset)
    for row in sessions.itertuples(index=False):
        if row.session_id in coupled_shares:
            g = (row.subject_id, row.task, str(row.stage))
            pools[g] = pools[g] | coupled_shares[row.session_id]
    matches = Counter(next((f"family {f}, first {n}" for (f, n), s in FAMILY_PREFIXES.items()
                            if s == pool), "no family prefix") for pool in pools.values())
    unmatched = Counter(str(sorted(pool)) for pool in pools.values()
                        if pool not in set(FAMILY_PREFIXES.values()))
    families = [FAMILY_PREFIXES[(f, len(pairs))] for f, pairs in REWARD_FAMILIES.items()]
    score_sets = [coupled_shares[r.session_id] for r in sessions.itertuples(index=False)
                  if r.role == "score" and r.session_id in coupled_shares]
    within = [any(shares <= fam for fam in families) for shares in score_sets]
    seen = [len(coupled_shares[r.session_id]) / len(pools[(r.subject_id, r.task, str(r.stage))])
            for r in sessions.itertuples(index=False)
            if r.role == "score" and r.session_id in coupled_shares]
    uncoupled_pools = Counter(str(sorted(v)) for v in uncoupled_values.values())

    sources = [{"type": "file", "id": str(manifest), "sha256": FREEZE_MANIFEST},
               {"type": "file", "id": db_path, "state": f"database snapshot {SNAPSHOT}"}]
    meta = build_meta("analysis/freeze_p0_task_checks.py", [], study_root=HERE.parent,
                      kind="freeze", code=[__file__], sources=sources)
    write_artifact(OUT, {
        "_meta": meta,
        "cohort": {"subjects": 149, "score_sessions": int((sessions["role"] == "score").sum()),
                   "pool": "heldout", "freeze_manifest_sha256": FREEZE_MANIFEST},
        "held_blocks": {"rows": hold_rows,
                        "coupled_total": {k: int(v) for k, v in totals.items()},
                        "share_over_max": totals["over_max"] / totals["blocks"],
                        "share_over_max_all_trials": totals["over_max_all_trials"] / totals["blocks"]},
        "reward_pairs": {
            "coupled_pools": len(pools),
            "pool_matches": dict(matches.most_common()),
            "score_sessions_within_one_family": {"n": len(within), "within": int(sum(within))},
            "unmatched_pools": dict(unmatched.most_common(10)),
            "score_session_share_of_pool_seen": {
                "n": len(seen), "mean": float(np.mean(seen)), "share_seeing_all": float(np.mean(
                    np.isclose(seen, 1.0)))},
            "pair_sum_vs_base": dict(sum_mismatch),
            "uncoupled_probability_sets": dict(uncoupled_pools.most_common(10)),
        },
        "coverage": dict(coverage),
    })
    print(json.dumps({"held": hold_rows, "total": dict(totals), "pairs": dict(matches)}, indent=1))


if __name__ == "__main__":
    main()
