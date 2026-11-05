#!/usr/bin/env python
"""P2: one gym task per score session, from the session's recorded task parameters (ADR 0006).

For each score session of the 149 test subjects (ADR 0007), the task a rollout plays:

- **coupled** (``Coupled Baiting``, ``Coupled Without Baiting``): ``CoupledBlockTask`` with the
  session's ``block_min``, ``block_max`` and ``block_beta`` (each the session's most common value),
  baiting from the task name, and as reward pairs the pool of pairs seen in the subject's sessions
  (fit and score) of the same task and stage, each pair's share of the high side
  (high ÷ (high + low)) scaled to this session's ``base_reward_probability_sum``.
- **uncoupled** (``Uncoupled Baiting``, ``Uncoupled Without Baiting``): ``UncoupledBlockTask`` with
  the session's ``block_min`` and ``block_max``, the set of reward probabilities the session used,
  ``perseverative_limit=4`` and ``max_block_tally=3`` (the real task's values; the gym's default
  tally is 4).
- **warm-up** (``p1_session_groups.json``'s ``warm-up``): ``CoupledBlockTask`` with the single pair
  [1, 0] scaled to the recorded base sum (ADR 0006).

Each task has the session's engaged trial count: the agents never ignore, and the floor is built
from the subject's engaged trials (decided with Han, 2026-11-05). A session that records no block
parameters is left out (Han, 2026-11-05), and so is an uncoupled session with one reward
probability, which has no schedule to draw: ``excluded`` lists them with the reason. The
minimum-reward hold (``min_reward_each_block``) is kept for the holds check (README, P2).

Check: each task is built and played by a coin-flip agent; for coupled sessions the gym's mean
completed block length is recorded beside the real session's (``_blocks``), a held block (ADR 0006)
making the real one longer.

Needs /allen (the database snapshot and the MEvolve freeze); on a compute node:

    srun --ntasks=1 --cpus-per-task=4 --mem=32G -p aind --time=00:30:00 \\
      <dynamic-foraging-bfm-cpu env>/bin/python analysis/freeze_p2_tasks.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
sys.path.insert(0, str(HERE))
from _meta import build_meta, write_artifact  # noqa: E402
from freeze_p0_task_checks import FREEZE, FREEZE_MANIFEST, SNAPSHOT, _blocks, load_trials  # noqa: E402
from p2_task import MAX_BLOCK_TALLY, PERSEVERATIVE_LIMIT, build_task  # noqa: E402

GROUPS = HERE / "p1_session_groups.json"
OUT = HERE / "p2_task_specs.json"
CHECK_SEED = 0


def _mode(values: np.ndarray):
    v = values[np.isfinite(values)]
    if v.size == 0:
        return None, False
    vals, counts = np.unique(v, return_counts=True)
    return float(vals[np.argmax(counts)]), bool(vals.size > 1)


def _shares(p: np.ndarray) -> set[float]:
    """The high side's share of each block's pair (equal pairs give 0.5); 0/1-only sessions none."""
    if np.isin(p, (0.0, 1.0)).all():
        return set()
    out = set()
    for start, _ in _blocks(p) + [(len(p) - 1, 1)]:
        hi, lo = max(p[start]), min(p[start])
        if hi + lo > 0:
            out.add(round(float(hi / (hi + lo)), 3))
    return out


def main() -> None:
    manifest = FREEZE / "manifest.json"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == FREEZE_MANIFEST
    sessions = pd.read_parquet(FREEZE / "sessions.parquet")
    sessions = sessions[sessions["pool"] == "heldout"].copy()
    assert sessions["subject_id"].nunique() == 149
    groups = json.loads(GROUPS.read_text())
    assert groups["freeze_manifest_sha256"] == FREEZE_MANIFEST
    trials, db_path = load_trials(sessions)
    by_session = dict(tuple(trials.groupby("session_id", sort=False)))

    pools = defaultdict(set)   # (subject, task, stage) -> shares, over fit and score sessions
    for r in sessions.itertuples(index=False):
        if r.task.startswith("Coupled") and groups["groups"].get(r.session_id) != "warm-up":
            t = by_session[r.session_id]
            pools[(r.subject_id, r.task, str(r.stage))] |= _shares(
                t[["reward_probabilityL", "reward_probabilityR"]].to_numpy(float))

    specs, excluded, check = [], [], defaultdict(list)
    rng = np.random.default_rng(CHECK_SEED)
    for r in sessions[sessions["role"] == "score"].itertuples(index=False):
        t = by_session[r.session_id]
        engaged = int((t["animal_response"].to_numpy(float) != 2).sum())
        group = groups["groups"][r.session_id]
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
        if bmin is None or bmax is None or (beta is None and r.task.startswith("Coupled")):
            excluded.append({**{k: spec[k] for k in ("subject_id", "session_id", "task", "group")},
                             "reason": "no recorded block parameters"})
            continue
        if group == "warm-up":
            spec.update(kind="warm-up", reward_pairs=[[0.0, base if base else 1.0]])
        elif r.task.startswith("Coupled"):
            shares = sorted(pools[(r.subject_id, r.task, str(r.stage))])
            if not shares or base is None or shares == [0.5]:   # the gym needs an unequal pair
                excluded.append({**{k: spec[k] for k in ("subject_id", "session_id", "task", "group")},
                                 "reason": "no recorded unequal reward pair or base sum"})
                continue
            spec.update(kind="coupled", base_reward_probability_sum=base,
                        reward_pairs=[[round((1 - s) * base, 6), round(s * base, 6)] for s in shares])
        elif r.task.startswith("Uncoupled"):
            probs = sorted(np.round(np.unique(p), 3).tolist())
            if len(probs) < 2:   # the gym's first block redraws until the sides differ: no task
                excluded.append({**{k: spec[k] for k in ("subject_id", "session_id", "task", "group")},
                                 "reason": f"one reward probability ({probs}), so no uncoupled schedule"})
                continue
            spec.update(kind="uncoupled", reward_probabilities=probs)
        else:
            excluded.append({**{k: spec[k] for k in ("subject_id", "session_id", "task", "group")},
                             "reason": f"no gym task for {r.task}"})
            continue
        if len(specs) % 200 == 0:
            print(len(specs), "sessions built", flush=True)
        task = build_task(spec, seed=int(rng.integers(2**31)))   # built and played once
        task.reset()
        for _ in range(spec["n_trials"]):
            task.step(int(rng.integers(2)))
        if spec["kind"] != "uncoupled":
            real = [n for _, n in _blocks(p)]
            sim = task.block_lens[:-1]   # completed blocks
            check[spec["kind"]].append((np.mean(real) if real else np.nan, np.mean(sim) if sim else np.nan))
        specs.append(spec)

    counts = Counter((s["kind"], s["group"]) for s in specs)
    meta = build_meta("analysis/freeze_p2_tasks.py", [], study_root=HERE.parent, kind="freeze",
                      code=[__file__, HERE / "freeze_p0_task_checks.py", HERE / "p2_task.py"], inputs=[GROUPS],
                      sources=[{"type": "file", "id": str(manifest), "sha256": FREEZE_MANIFEST},
                               {"type": "file", "id": db_path, "state": f"database snapshot {SNAPSHOT}"}])
    write_artifact(OUT, {
        "_meta": meta,
        "cohort": {"subjects": 149, "freeze_manifest_sha256": FREEZE_MANIFEST},
        "rules": {"perseverative_limit": PERSEVERATIVE_LIMIT, "max_block_tally": MAX_BLOCK_TALLY,
                  "n_trials": "engaged trials", "block_parameters": "the session's most common value",
                  "missing_block_parameters": "excluded"},
        "n_sessions": len(specs), "n_excluded": len(excluded),
        "by_kind_and_group": {f"{k}|{g}": n for (k, g), n in sorted(counts.items())},
        "n_parameters_vary": sum(s["parameters_vary"] for s in specs),
        "block_length_check": {k: {"n_sessions": len(v), "real_mean": float(np.nanmean([a for a, _ in v])),
                                   "gym_mean": float(np.nanmean([b for _, b in v]))} for k, v in check.items()},
        "excluded": excluded, "sessions": specs,
    })
    print(json.dumps({k: v for k, v in json.loads(OUT.read_text()).items()
                      if k not in ("sessions", "excluded", "_meta")}, indent=1))


if __name__ == "__main__":
    main()
