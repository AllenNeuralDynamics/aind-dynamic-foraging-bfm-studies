#!/usr/bin/env python
"""P2 check: does the coupled task's minimum-reward hold move the rollout fingerprints?

ADR 0006's rollouts have neither coupled hold. Only one is recorded: ``BlockMinReward``
(``min_reward_each_block``); the auto-block hold is not, so it cannot be tested (decided with Han,
2026-11-05). The recorded minimum is 0 in most coupled sessions (``n_sessions_by_minimum``). On
every coupled or warm-up score session that records a positive minimum, each
model plays ``N_ROLLOUTS`` rollouts with the hold off and on (``p2_task.build_task(hold=...)``),
with the same seeds, so the pair differs only by the hold. Models: the five literature models with
the best 2-level EM likelihood (as 2-level EM; r2's scatter models) and the GRU (seed 0).

Per model, subject and statistic: the gap between the two rollouts' tallies, over the subject's
kept bins (``rollout_gap``'s bins and weights), against the split-half floor of the subject's same sessions
(``noise_floor``); a ratio of the subject-balanced means below 1 means the hold moves the
fingerprints less than the subject's own sessions differ from each other (README, P2: then the
headline keeps the simulator as it is; else a new ADR turns the hold on).

Needs /allen and the mevolve env; on a compute node:

    sbatch -p aind --ntasks=1 --cpus-per-task=56 --mem=128G --time=02:00:00 --wrap \\
      "env PYTHONPATH=<foraging-models>/src:<mevolve>/src <mevolve env>/bin/python \\
       analysis/freeze_p2_holds.py --models-repo <foraging-models> --mevolve-repo <mevolve>"
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
sys.path.insert(0, str(HERE))
import cross_study_literature as lit  # noqa: E402
from _meta import build_meta, write_artifact  # noqa: E402
from freeze_p0_golden_mevolve import FREEZE, FREEZE_MANIFEST, _git_sha, _sha256  # noqa: E402
from freeze_p2_rollouts import (CONTEXTS, GRU, GRU_MODEL, N_ROLLOUTS, SPECS, TWO_LEVEL_INDEX, GRUAgent,  # noqa: E402
                                _gru_params, _program_agents, _streams)

OUT, REPLAY_GAPS = HERE / "p2_holds.json", HERE / "p1_replay_gaps.json"
N_TOP = 5


def _job(args) -> dict:
    """One model on one subject: hold off and on, the gap between them and the subject's floor."""
    model, sid, specs, facts, make, real = args
    import jax.numpy as jnp

    from aind_dynamic_foraging_models.fingerprints import STATISTICS, kept_bins, noise_floor, rollout, session_tally, tally_sessions, values
    from aind_dynamic_foraging_models.fingerprints.tally import N
    from p2_task import build_task

    tallies = {}
    for hold in (False, True):
        choices, rewards = [], []
        for spec in specs:
            for k in range(N_ROLLOUTS):
                task_seed, rng = _streams(model, "holds", spec["session_id"], k)   # the same for off and on
                c, r, _ = rollout(make(rng, jnp, spec["session_id"]), build_task(spec, task_seed, hold=hold), rng,
                                  session=facts[spec["session_id"]])
                choices.append(c)
                rewards.append(r)
        tallies[hold] = tally_sessions(choices, rewards)
    own = [session_tally(c, r) for c, r in real]
    subject = tally_sessions([c for c, _ in real], [r for _, r in real])
    out = {}
    for st in STATISTICS:   # rollout_gap's bins and weights (the subject's), the hold-off rollout as reference
        on, off = values(tallies[True], st), values(tallies[False], st)
        ok = kept_bins(subject, st) & np.isfinite(on) & np.isfinite(off)
        w = subject[st][:, N]
        gap = float(np.sum(w[ok] * np.abs(on[ok] - off[ok])) / np.sum(w[ok])) if ok.any() else np.nan
        out[st] = {"gap": gap, "floor": float(noise_floor(own, st))}
    return {"model": model, "subject_id": sid, "statistics": out}


def _literature_job(args):
    name, sid, specs, facts, params, rec, real = args
    agents = _program_agents(name, params, rec)
    return _job((name, sid, specs, facts, agents["two_level_em"], real))


def _gru_job(args):
    name, sid, specs, facts, path, contexts, real = args

    def make(rng, jnp, session_id):
        return GRUAgent(_gru_params(path), contexts[session_id])
    return _job((name, sid, specs, facts, make, real))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-repo", type=Path, required=True)
    ap.add_argument("--mevolve-repo", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=56)
    ap.add_argument("--subjects", type=int, help="smoke test: the first N subjects, nothing written")
    a = ap.parse_args()
    from mevolve import fitting
    from mevolve.data import Freeze
    from mevolve.workers import WORKER_ENV

    from aind_dynamic_foraging_models.fingerprints import STATISTICS, floor_ratio

    freeze = Freeze(FREEZE)
    assert _sha256(FREEZE / "manifest.json") == FREEZE_MANIFEST, "another data freeze"
    all_specs = json.loads(SPECS.read_text())["sessions"]
    specs = [s for s in all_specs
             if s["kind"] in ("coupled", "warm-up") and (s["min_reward_each_block"] or 0) > 0]
    by_subject: dict = {}
    for s in specs:
        by_subject.setdefault(s["subject_id"], []).append(s)
    if a.subjects:
        by_subject = {k: by_subject[k] for k in sorted(by_subject)[: a.subjects]}
    facts, real = {}, {}
    for sid, ss in by_subject.items():
        xs = {x.session_id: x for x in freeze.subject_sessions(sid, "score")}
        for s in ss:
            facts[s["session_id"]] = fitting.session_facts(xs[s["session_id"]])
            real[s["session_id"]] = (xs[s["session_id"]].choices, xs[s["session_id"]].rewards)
    # r2's scatter models, from the frozen inputs r2 picks them from: the five best by pooled 2-level EM
    # likelihood whose 2-level fit did not fail (study 12 r3) and whose replay reproduced (P1)
    failed = {r["literature_model"] for r in lit.literature_headline_ranking() if r["source"] != "2-level EM"}
    skip = {e["model"] for e in json.loads(REPLAY_GAPS.read_text())["excluded"] if e["regime"] == "two_level_em"}
    pooled = {m: sum(v[0] for v in t.values()) / sum(v[1] for v in t.values())
              for m, t in lit.two_level_em_per_subject("heldout").items() if m not in failed | skip}
    top = sorted(pooled, key=pooled.get)[-N_TOP:]   # worst to best
    index = {r["literature_model"]: r for r in json.loads(TWO_LEVEL_INDEX.read_text())["runs"]}
    sources = [{"type": "file", "id": str(FREEZE / "manifest.json"), "sha256": FREEZE_MANIFEST}]
    jobs = []
    for name in top:
        run = index[name]
        assert _sha256(Path(run["path"])) == run["sha256"]
        sources.append({"type": "file", "id": run["path"], "sha256": run["sha256"]})
        recs = {str(r["subject_id"]): r for r in json.loads(Path(run["path"]).read_text())["subjects"]}
        params = lit.literature_fitted_params(name)
        jobs += [(_literature_job, (name, sid, ss, {s["session_id"]: facts[s["session_id"]] for s in ss}, params[sid],
                                    recs[sid], [real[s["session_id"]] for s in ss])) for sid, ss in by_subject.items()]
    ctx = json.loads(CONTEXTS.read_text())["seeds"]["seed0"]
    assert _sha256(Path(ctx["params"])) == ctx["params_sha256"], "the GRU's seed-0 params differ"
    sources.append({"type": "file", "id": ctx["params"], "sha256": ctx["params_sha256"]})
    trained = set(json.loads(GRU.read_text())["seeds"]["seed0"]["subjects_in_training"])
    jobs += [(_gru_job, (GRU_MODEL, sid, ss, {s["session_id"]: facts[s["session_id"]] for s in ss}, ctx["params"],
                         {s["session_id"]: ctx["contexts"][s["session_id"]] for s in ss}, [real[s["session_id"]] for s in ss]))
             for sid, ss in by_subject.items() if sid not in trained]
    os.environ.update(WORKER_ENV)
    with mp.get_context("spawn").Pool(min(a.workers, len(jobs))) as pool:
        res = pool.starmap(_call, jobs, chunksize=1)
    models = {}
    for m in top + [GRU_MODEL]:
        rows = [r for r in res if r["model"] == m]
        models[m] = {"n_subjects": len(rows), "statistics": {}}
        for st in STATISTICS:
            g = np.array([r["statistics"][st]["gap"] for r in rows], float)
            f = np.array([r["statistics"][st]["floor"] for r in rows], float)
            ok = np.isfinite(g) & np.isfinite(f)   # the subjects floor_ratio uses, for the means too
            models[m]["statistics"][st] = {"hold_gap": float(g[ok].mean()), "floor": float(f[ok].mean()),
                                           "ratio": floor_ratio(g, f), "n_subjects": int(ok.sum())}
        print(m, {st: round(c["ratio"], 2) for st, c in models[m]["statistics"].items()}, flush=True)
    worst = max(c["ratio"] for m in models.values() for c in m["statistics"].values())
    if a.subjects:
        return
    meta = build_meta("analysis/freeze_p2_holds.py", [], study_root=HERE.parent, kind="freeze",
                      code=[__file__, HERE / "freeze_p2_rollouts.py", HERE / "p2_task.py",
                            HERE / "freeze_p0_golden_mevolve.py"],
                      inputs=[SPECS, CONTEXTS, GRU, TWO_LEVEL_INDEX, REPLAY_GAPS, lit.TWO_LEVEL_EM_SOURCE,
                              lit.HEADLINE_SOURCE, lit.LITERATURE_SOURCE], sources=sources)
    write_artifact(OUT, {
        "_meta": meta,
        "fingerprints_library": {"repo": "AllenNeuralDynamics/aind-dynamic-foraging-models",
                                 "git_sha": _git_sha(a.models_repo)},
        "n_rollouts": N_ROLLOUTS, "n_sessions": len(specs), "n_subjects": len(by_subject),
        "sessions_by_group": {g: sum(s["group"] == g for s in specs) for g in sorted({s["group"] for s in specs})},
        "n_sessions_by_minimum": {f"{k}|{m:g}": n for (k, m), n in sorted(Counter(
            (s["kind"], s["min_reward_each_block"] or 0.0) for s in all_specs if s["kind"] != "uncoupled").items())},
        "models": models, "max_ratio": worst, "holds_move_less_than_floor": bool(worst < 1),
    })
    print(f"max ratio {worst:.2f} -> {OUT}")


def _call(fn, args):
    return fn(args)


if __name__ == "__main__":
    main()
