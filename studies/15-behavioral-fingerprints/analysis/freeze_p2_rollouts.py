#!/usr/bin/env python
"""P2: rollouts of the literature models and the GLM in each score session's own task (ADR 0005, 0006).

Each model plays, ``N_ROLLOUTS`` times, the gym task built from each score session's recorded
parameters (``p2_task_specs.json``; ``p2_task.build_task``), through the fingerprint
library's ``rollout``, and the library tallies the rollouts by the session's group:

- **per subject**: study 12's fitted parameters, the MEvolve program receiving the real session's
  facts (``fitting.session_facts``: stage group, coupled, baited, index), as in its fit;
- **2-level EM**: each rollout draws its own session parameters from the subject's distribution,
  theta = mu + exp(log sigma) * eps in the fit's unconstrained space (ADR 0005);
- **GLM**: the per-subject Su2022 regression of ``p1_glm.json`` (``GLMAgent``);
- **GRU**: study 01's v1 GRU at D = 614, fine-tuned on the held-out subjects (``gru-heldout-replay``),
  one regime per seed, on the 146 subjects its checkpoint did not train on: a numpy GRU step
  (``GRUAgent``) with each session's context from ``p2_gru_contexts.json``. Gate before its
  rollouts count: ``replay`` through ``GRUAgent`` on every real score session equals the held-out
  CSV's per-trial ``probability_choice_1`` (``TOL_GRU``).

Every rollout is seeded from (``SEED``, model, regime, session, rollout), the task by one stream and
the agent's choices (and draws) by another. A 2-level cell that P1 left out (``p1_replay_gaps.json``
``excluded``) is not rolled out.

Check, per model and subject: ``replay`` on one rollout's own choices and rewards gives back its
p(right) (``TOL_SELF``), so the agent plays the model it replays.

Writes ``p2_rollouts/<model>.parquet`` (regime, subject, group, statistic, bin, n, obs: the summed
rollout tallies) and ``p2_rollouts/animal_sessions.parquet`` (each real score session's own tally,
for the split-half floor), and ``p2_rollouts.json`` (the run and its checks). Needs /allen and the
mevolve env; on a compute node:

    sbatch -p aind --ntasks=1 --cpus-per-task=56 --mem=200G --time=04:00:00 --wrap \\
      "env PYTHONPATH=<foraging-models>/src:<mevolve>/src <mevolve env>/bin/python \\
       analysis/freeze_p2_rollouts.py --models-repo <foraging-models> --mevolve-repo <mevolve>"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
sys.path.insert(0, str(HERE))
import cross_study_literature as lit  # noqa: E402
from _meta import build_meta, write_artifact  # noqa: E402
from freeze_p0_golden_mevolve import FREEZE, FREEZE_MANIFEST, _git_sha, _sha256  # noqa: E402

SPECS, GLM, GAPS = HERE / "p2_task_specs.json", HERE / "p1_glm.json", HERE / "p1_replay_gaps.json"
CONTEXTS, GRU = HERE / "p2_gru_contexts.json", HERE / "p1_gru.json"
GRU_MODEL, TOL_GRU = "GRU-D614", 1e-4
TWO_LEVEL_INDEX = HERE.parents[1] / "12-cognitive-literature-leaderboard" / "analysis" / "two_level_em_fits.json"
OUT, OUT_DIR = HERE / "p2_rollouts.json", HERE / "p2_rollouts"
N_ROLLOUTS, SEED, TOL_SELF = 10, 0, 1e-9
REGIMES = ("per_subject", "two_level_em")


def _streams(model: str, regime: str, session_id: str, k: int):
    """Independent task and agent generators for one rollout."""
    key = [SEED, zlib.crc32(model.encode()), zlib.crc32(regime.encode()), zlib.crc32(session_id.encode()), k]
    task_ss, agent_ss = np.random.SeedSequence(key).spawn(2)
    return int(task_ss.generate_state(1)[0]), np.random.default_rng(agent_ss)


class _Jitted:
    """A MEvolve program with jitted predict and update (one compile per worker)."""

    def __init__(self, prog):
        import jax

        self.init_latents, self.predict, self.update = prog.init_latents, jax.jit(prog.predict), jax.jit(prog.update)


_PROGRAMS: dict = {}


def _program(name: str):
    if name not in _PROGRAMS:
        from mevolve import literature, program

        prog = program.literature(literature.LITERATURE[name][0])
        _PROGRAMS[name] = (prog, _Jitted(prog))
    return _PROGRAMS[name]


def _tally_rows(regime, sid, tallies):
    from aind_dynamic_foraging_models.fingerprints import BINS, N, OBS, STATISTICS

    return [(regime, sid, g, st, lab, float(n), float(o)) for g, t in tallies.items() if g != "all"
            for st in STATISTICS for lab, n, o in zip(BINS[st], t[st][:, N], t[st][:, OBS])]


class GRUAgent:
    """The fine-tuned GRU as an ``Agent``: the latent is the state the next prediction reads.

    ``init_latents`` steps a zero state on [-1, -1] (no previous trial); ``predict`` reads out the
    2 logits; ``update`` steps on (choice, reward). The update is Haiku's GRU
    (``post_training_analysis/gru_step.py``) on [obs, context], in float32 as the network ran.
    """

    def __init__(self, params: dict, context):
        g, r = params["multisubject_gru/~/gru"], params["multisubject_gru/~/readout"]
        f = np.float32
        self.w_i, self.w_h, self.b = (np.asarray(g[k], f) for k in ("w_i", "w_h", "b"))
        self.w_o, self.b_o = np.asarray(r["w"], f), np.asarray(r["b"], f)
        self.c = np.asarray(context, f)
        self.n = self.w_h.shape[0]

    def _step(self, h, obs):
        n = self.n
        x = np.concatenate([np.asarray(obs, np.float32), self.c])
        zr_x, a_x = np.split(x @ self.w_i, [2 * n])
        b_z, b_a = np.split(self.b, [2 * n])
        z, r = np.split(1 / (1 + np.exp(-(zr_x + h @ self.w_h[:, :2 * n] + b_z))), 2)
        a = np.tanh(a_x + (r * h) @ self.w_h[:, 2 * n:] + b_a)
        return (1 - z) * h + z * a

    def init_latents(self, n_trials, session=None):
        return self._step(np.zeros(self.n, np.float32), (-1.0, -1.0))

    def predict(self, h):
        logits = (h @ self.w_o + self.b_o).astype(float)
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def update(self, h, choice, reward):
        return self._step(h, (float(choice), float(reward)))


_GRU_PARAMS: dict = {}


def _gru_params(path: str) -> dict:
    if path not in _GRU_PARAMS:
        _GRU_PARAMS[path] = json.loads(Path(path).read_text())
    return _GRU_PARAMS[path]


def _gru_job(args):
    """The GRU on one subject: the replay gate on its real sessions, then its rollouts per seed."""
    from aind_dynamic_foraging_models.fingerprints import replay

    name, sid, specs, sessions, real, seeds = args   # seeds: {regime: (params path, {session: context}, {session: csv p})}
    gate = 0.0
    for regime, (path, contexts, csv) in seeds.items():
        for spec in specs:
            x = real[spec["session_id"]]
            p = replay(GRUAgent(_gru_params(path), contexts[spec["session_id"]]), x[0], x[1])
            d = np.abs(p - csv[spec["session_id"]])
            gate = max(gate, float(np.max(np.where(np.isfinite(d), d, np.inf))))

    def maker(path, contexts):
        def make(rng, jnp, session_id):
            return GRUAgent(_gru_params(path), contexts[session_id])
        return make
    res = _subject_job((name, sid, specs, sessions,
                        {regime: maker(path, contexts) for regime, (path, contexts, _) in seeds.items()}))
    res["max_gate_diff"] = gate
    return res


def _subject_job(args) -> dict:
    """One model on one subject: every regime, every session, ``N_ROLLOUTS`` rollouts each."""
    model, sid, specs, sessions, agents = args
    import jax.numpy as jnp

    from aind_dynamic_foraging_models.fingerprints import replay, rollout, tally_groups
    from p2_task import build_task

    rows, self_diff = [], 0.0
    for regime, make in agents.items():
        choices, rewards, groups = [], [], []
        for spec in specs:
            facts = sessions[spec["session_id"]]
            for k in range(N_ROLLOUTS):
                task_seed, rng = _streams(model, regime, spec["session_id"], k)
                agent = make(rng, jnp, spec["session_id"])
                c, r, p = rollout(agent, build_task(spec, task_seed), rng, session=facts)
                if k == 0 and spec is specs[0]:   # the agent plays the model it replays
                    d = np.abs(replay(agent, c, r, session=facts) - p)
                    self_diff = max(self_diff, float(np.max(d)))
                choices.append(c)
                rewards.append(r)
                groups.append(spec["group"])
        rows += _tally_rows(regime, sid, tally_groups(choices, rewards, groups))
    return {"subject_id": sid, "rows": rows, "max_self_diff": self_diff,
            "n_rollout_sessions": len(specs) * N_ROLLOUTS}


def _program_agents(name, params, rec):
    """Agent factories per regime for a literature model and one subject: ``make(rng, jnp)``."""
    from aind_dynamic_foraging_models.fingerprints import ProgramAgent
    from mevolve import fitting

    prog, jitted = _program(name)
    agents = {"per_subject": lambda rng, jnp, sid: ProgramAgent(jitted, {k: params[k] for k in prog.params})}
    if rec is not None:
        mu = np.asarray(fitting.from_params(prog, rec["typical_params"]), float)
        sigma = np.exp(np.asarray(rec["log_sigma"], float))

        def two_level(rng, jnp, sid):   # ADR 0005: each rollout draws its session's parameters
            theta = mu + sigma * rng.standard_normal(mu.shape)
            p = fitting.to_params(prog, jnp.asarray(theta))
            return ProgramAgent(jitted, {k: float(v) for k, v in p.items()})
        agents["two_level_em"] = two_level
    return agents


def _literature_job(args):
    name, sid, specs, sessions, params, rec = args
    return _subject_job((name, sid, specs, sessions, _program_agents(name, params, rec)))


def _glm_job(args):
    from aind_dynamic_foraging_models.fingerprints import GLMAgent

    name, sid, specs, sessions, r = args
    return _subject_job((name, sid, specs, sessions,
                         {"per_subject": lambda rng, jnp, s: GLMAgent(r["coef_rewc"], r["coef_unrc"], r["bias"])}))


def _gru_jobs(name, subjects, by_subject, facts, real_sessions, sources) -> list:
    """Per held-out subject the GRU's seeds: params, each session's context and the CSV's p(right)."""
    import pandas as pd

    ctx = json.loads(CONTEXTS.read_text())
    assert ctx["passed"], "the GRU contexts' first-trial check failed"
    trained = set(json.loads(GRU.read_text())["seeds"]["seed0"]["subjects_in_training"])
    seeds = {}
    for regime, e in ctx["seeds"].items():
        assert _sha256(Path(e["params"])) == e["params_sha256"], f"{regime}: params differ"
        csv = Path(e["params"]).parent / "test_trial_predictions.csv"
        sources += [{"type": "file", "id": e["params"], "sha256": e["params_sha256"]},
                    {"type": "file", "id": str(csv), "sha256": _sha256(csv)}]
        df = pd.read_csv(csv, usecols=["source_ses_idx", "trial", "probability_choice_1"])
        probs = {k: g.sort_values("trial").probability_choice_1.to_numpy() for k, g in df.groupby("source_ses_idx")}
        seeds[regime] = (e["params"], e["contexts"], probs)
    jobs = []
    for sid in subjects:
        if sid in trained:
            continue
        sp = by_subject[sid]
        ids = [x["session_id"] for x in sp]
        jobs.append((name, sid, sp, {i: facts[i] for i in ids}, {i: real_sessions[i] for i in ids},
                     {r: (path, {i: c[i] for i in ids}, {i: pr[i] for i in ids}) for r, (path, c, pr) in seeds.items()}))
    return jobs


def _write(name: str, rows: list) -> Path:
    import pandas as pd

    path = OUT_DIR / f"{name}.parquet"
    pd.DataFrame(rows, columns=["regime", "subject_id", "group", "statistic", "bin", "n", "obs"]).to_parquet(
        path, index=False)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-repo", type=Path, required=True)
    ap.add_argument("--mevolve-repo", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=56)
    ap.add_argument("--models", nargs="*", help="only these (a smoke test; writes nothing)")
    ap.add_argument("--subjects", type=int, help="only the first N subjects (a smoke test)")
    a = ap.parse_args()
    import aind_dynamic_foraging_models

    lib_path = Path(aind_dynamic_foraging_models.__file__).resolve()
    assert (a.models_repo / "src").resolve() in lib_path.parents, f"library from {lib_path}"
    from mevolve import fitting, literature
    from mevolve.data import Freeze
    from mevolve.workers import WORKER_ENV

    from aind_dynamic_foraging_models.fingerprints import BINS, N, OBS, SPEC_VERSION, session_tally

    freeze = Freeze(FREEZE)
    assert _sha256(FREEZE / "manifest.json") == FREEZE_MANIFEST, "another data freeze"
    specs_all = json.loads(SPECS.read_text())
    assert specs_all["cohort"]["freeze_manifest_sha256"] == FREEZE_MANIFEST
    by_subject: dict = {}
    for s in specs_all["sessions"]:
        by_subject.setdefault(s["subject_id"], []).append(s)
    subjects = sorted(by_subject)[: a.subjects] if a.subjects else sorted(by_subject)
    facts, animal, real_sessions = {}, [], {}
    for sid in subjects:
        real = {x.session_id: x for x in freeze.subject_sessions(sid, "score")}
        for spec in by_subject[sid]:
            x = real[spec["session_id"]]
            real_sessions[spec["session_id"]] = (x.choices, x.rewards)
            assert len(x.choices) == spec["n_trials"], f"{spec['session_id']}: engaged trials differ"
            facts[spec["session_id"]] = fitting.session_facts(x)
            t = session_tally(x.choices, x.rewards)
            animal += [(sid, spec["session_id"], spec["group"], st, lab, float(n), float(o))
                       for st, tab in t.items() for lab, n, o in zip(BINS[st], tab[:, N], tab[:, OBS])]

    gaps = json.loads(GAPS.read_text())
    skip = {(e["model"], e["regime"]) for e in gaps["excluded"]}
    index = {r["literature_model"]: r for r in json.loads(TWO_LEVEL_INDEX.read_text())["runs"]}
    ranked = lit.literature_ranking()["ranked"]
    names = a.models or ranked + [json.loads(GLM.read_text())["model"], GRU_MODEL]
    os.environ.update(WORKER_ENV)
    OUT_DIR.mkdir(exist_ok=True)
    sources = [{"type": "file", "id": str(FREEZE / "manifest.json"), "sha256": FREEZE_MANIFEST}]
    runs, outputs = {}, []
    glm = {r["subject_id"]: r for r in json.loads(GLM.read_text())["subjects"]}
    for name in names:
        if name == GRU_MODEL:
            jobs, fn = _gru_jobs(name, subjects, by_subject, facts, real_sessions, sources), _gru_job
        elif name in literature.LITERATURE:
            params = lit.literature_fitted_params(name)
            recs = None
            if (name, "two_level_em") not in skip:
                run = index[name]
                assert _sha256(Path(run["path"])) == run["sha256"], f"{name}'s 2-level file differs"
                recs = {str(r["subject_id"]): r for r in json.loads(Path(run["path"]).read_text())["subjects"]}
                sources.append({"type": "file", "id": run["path"], "sha256": run["sha256"]})
            jobs = [(name, sid, by_subject[sid], {s["session_id"]: facts[s["session_id"]] for s in by_subject[sid]},
                     params[sid], None if recs is None else recs[sid]) for sid in subjects]
            fn = _literature_job
        else:
            jobs = [(name, sid, by_subject[sid], {s["session_id"]: facts[s["session_id"]] for s in by_subject[sid]},
                     glm[sid]) for sid in subjects]
            fn = _glm_job
        with mp.get_context("spawn").Pool(min(a.workers, len(jobs))) as pool:   # fresh workers per model
            res = pool.map(fn, jobs, chunksize=1)
        runs[name] = {"n_subjects": len(res), "max_self_diff": max(r["max_self_diff"] for r in res),
                      **({"max_gate_diff": max(r["max_gate_diff"] for r in res), "tolerance_gate": TOL_GRU}
                         if name == GRU_MODEL else {}),
                      "n_rollout_sessions": sum(r["n_rollout_sessions"] for r in res),
                      "regimes": sorted({row[0] for r in res for row in r["rows"]})}
        print(name, runs[name], flush=True)
        if not a.models and not a.subjects:
            outputs.append(_write(name, [row for r in res for row in r["rows"]]))
    if a.models or a.subjects:
        return

    import pandas as pd

    path = OUT_DIR / "animal_sessions.parquet"
    pd.DataFrame(animal, columns=["subject_id", "session_id", "group", "statistic", "bin", "n", "obs"]).to_parquet(
        path, index=False)
    outputs.append(path)
    passed = all(r["max_self_diff"] <= TOL_SELF and r.get("max_gate_diff", 0.0) <= TOL_GRU for r in runs.values())
    meta = build_meta("analysis/freeze_p2_rollouts.py", [], study_root=HERE.parent, kind="freeze",
                      code=[__file__, HERE / "p2_task.py", HERE / "freeze_p0_golden_mevolve.py"],
                      inputs=[SPECS, GLM, GAPS, CONTEXTS, GRU, TWO_LEVEL_INDEX, lit.LITERATURE_SOURCE,
                              lit.LEADERBOARD_SOURCE],
                      sources=sources)
    write_artifact(OUT, {
        "_meta": meta,
        "fingerprints_library": {"repo": "AllenNeuralDynamics/aind-dynamic-foraging-models",
                                 "git_sha": _git_sha(a.models_repo)},
        "mevolve_git_sha": (a.mevolve_repo / "MEVOLVE_GIT_SHA").read_text().strip(),
        "spec": SPEC_VERSION, "n_rollouts": N_ROLLOUTS, "seed": SEED, "tolerance_self": TOL_SELF,
        "n_subjects": len(subjects), "n_sessions": sum(len(by_subject[s]) for s in subjects),
        "excluded_cells": sorted(f"{m}|{r}" for m, r in skip), "passed": passed, "models": runs,
    }, outputs=outputs)
    print(f"{'PASS' if passed else 'FAIL'} -> {OUT}")


if __name__ == "__main__":
    main()
