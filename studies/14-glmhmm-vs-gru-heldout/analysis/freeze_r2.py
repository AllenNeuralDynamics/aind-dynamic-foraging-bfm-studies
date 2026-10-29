"""Freeze for r2: one GLM per {subject, state} under each K-grid run's global fit.

For every hurdle and reference K-grid run (K = 1..10) and both cohorts (the 149 held-out mice
and the 614 training mice, every session of each):
1. Viterbi path of every session under the run's global parameters, so state k is the same
   state in every mouse (per-mouse refits reorder states; r1 result-3).
2. For each subject and state, the subject's trials on that state's Viterbi path. Lags are
   built on the whole session before the rows are taken, so no history crosses a cut.
3. One MAP fit of that state's emission GLM to those trials: the package's own Newton M-step
   with the Viterbi path as 0/1 weights, the per-mouse refit's prior (Gaussian, SD 2, centred
   at 0), started from the global state's weights. A second M-step from the result measures
   convergence.

Writes ``frozen_r2.json`` (_meta, runs, subjects, cell tables) and one weight file per
emission, ``frozen_r2_weights_<emission>.npz`` (float16), both read by ``r2_state_spread.py``.
Cells with fewer than ``MIN_FROZEN_TRIALS`` trials are not frozen; the producer applies its own,
higher threshold.

Needs the /allen mount and the snapshot, so it runs on an HPC compute node:

    WRAPPER_ROOT=<wrapper checkout> DISPATCHER_ROOT=<dispatcher checkout> \\
        make -C studies/14-glmhmm-vs-gru-heldout freeze-r2

``R2_SMOKE=1`` runs the K = 3 run of each emission (or ``R2_RUNS``) on ``R2_SMOKE_MICE`` (20)
mice per cohort and writes nothing unless ``R2_PARTS`` is set; a smoke merge writes to the
ignored ``analysis/_cache_smoke/``. A SLURM array runs one run per task (``R2_RUNS``, ``R2_PARTS``), then
``R2_MERGE=1`` writes the frozen files (``main``).
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
sys.path.insert(0, str(HERE))
from _meta import build_meta, write_artifact  # noqa: E402
from freeze_round1 import (  # noqa: E402
    DATA_OVERRIDES,
    PROJECT,
    SNAPSHOT,
    find_runs,
    sha256,
    snapshot_source,
)

WANDB_GROUPS = ["hurdle@20261027-212905", "reference@20261027-213006"]
OUT = HERE / "frozen_r2.json"
MIN_FROZEN_TRIALS = 50
PRIOR_SIGMA = 2.0                # the per-mouse refit's weight prior (SUBJECT_PRIOR)
NEWTON_STEPS = 30                # per fit; a second fit from the result checks convergence
SUBJECTS_PER_CHUNK = 32          # mice per posterior pass; compiled code is released after each


def load_cohorts():
    """Every session of both cohorts, as the trainer's session records, keyed by cohort.

    The loader and the session builder are the trainer's own (``_sessions``), at the wrapper
    and dispatcher commits the K-grid runs used.
    """
    wrapper, dispatcher = Path(os.environ["WRAPPER_ROOT"]), Path(os.environ["DISPATCHER_ROOT"])
    sys.path.insert(0, str(wrapper / "code"))
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from model_trainers.glmhmm_trainer import _sessions

    with initialize_config_dir(config_dir=str(dispatcher / "code" / "config"), version_base=None):
        cfg = compose(config_name="config", overrides=DATA_OVERRIDES + ["model=glmhmm"])
    bundles = {"training": instantiate(cfg.data).load(),
               "heldout": instantiate(cfg.data, split="heldout", multisubject=True,
                                      subject_ids=None).load()}
    cohorts = {}
    for name, bundle in bundles.items():
        ids = list(bundle.metadata["train_session_ids"]) + list(bundle.metadata["eval_session_ids"])
        records = _sessions(bundle.raw, ids)
        by_subject = {}
        for r in records:
            by_subject.setdefault(r["subject_id"], []).append(r)
        cohorts[name] = {s: sorted(rs, key=lambda r: str(r["ses_idx"]))
                         for s, rs in sorted(by_subject.items())}
    return cohorts


def flat_weights(emission, e):
    """One {subject, state} weight vector per state: (K, D) from an emission parameter block."""
    if emission == "hurdle":
        return np.concatenate([np.asarray(e["side_weights"] if isinstance(e, dict) else e.side_weights),
                               np.asarray(e["engage_weights"] if isinstance(e, dict) else e.engage_weights)], 1)
    w = np.asarray(e["weights"] if isinstance(e, dict) else e.weights)
    return w.reshape(w.shape[0], -1)


def feature_names(emission, n_lags):
    """Names of the columns of ``flat_weights``."""
    from aind_dynamic_foraging_models.glm_hmm import column_names

    names = column_names(n_lags)                       # 4L lag columns + bias
    L = n_lags
    if emission == "hurdle":
        return ([f"side:{n}" for n in names[:2 * L]] + ["side:bias"]
                + [f"engage:{n}" for n in names[2 * L:4 * L]] + ["engage:bias"])
    return [f"left:{n}" for n in names] + [f"right:{n}" for n in names]


def viterbi_paths(model, params, subjects, cfg):
    """Viterbi path of every session of ``subjects`` (subject -> list of records)."""
    from aind_dynamic_foraging_models.glm_hmm import make_buckets, state_posteriors

    records = [r for s in subjects for r in subjects[s]]
    buckets = make_buckets([(r["choices"], r["rewards"]) for r in records], model.n_lags,
                           cfg["design"], cfg["bucket_size"])
    posts = state_posteriors(model, params, buckets)
    return [np.asarray(posts[i]["viterbi"]) for i in range(len(records))]


def fit_cells(model, params, props, records, paths, n_lags, design):
    """MAP emission weights of every state on one subject's Viterbi trials.

    Returns (weights (K, D), n_trials (K,), n_engaged (K,), n_sessions (K,), max |change| of a
    second M-step started from the result).
    """
    import jax
    from aind_dynamic_foraging_models.glm_hmm import build_inputs
    from aind_dynamic_foraging_models.glm_hmm.emissions import aggregate
    from jax.experimental import enable_x64

    K = model.num_states
    labels = np.concatenate([r["choices"] for r in records])
    inputs = np.concatenate([build_inputs(r["choices"], r["rewards"], n_lags, design) for r in records])
    states = np.concatenate(paths)
    session = np.concatenate([np.full(len(r["choices"]), i) for i, r in enumerate(records)])
    onehot = np.eye(K)[states]
    component = model.emission_component
    with enable_x64():
        blocks = component.prepare_m_step(labels, inputs)
        stats = tuple((b.X, b.targets, aggregate(onehot, b)) for b in blocks)
        start = jax.tree_util.tree_map(lambda x: np.asarray(x, np.float64), params.emissions)
        fitted, _ = component.m_step(start, props.emissions, stats, None, PRIOR_SIGMA)
        again, _ = component.m_step(fitted, props.emissions, stats, None, PRIOR_SIGMA)
    w = flat_weights(model.emission, fitted)
    delta = float(np.abs(flat_weights(model.emission, again) - w).max())
    n_trials = np.bincount(states, minlength=K)
    n_engaged = np.bincount(states[labels != 2], minlength=K)
    n_sessions = np.array([len(np.unique(session[states == k])) for k in range(K)])
    return w, n_trials, n_engaged, n_sessions, delta


def release_compiled(where):
    """Drop JAX's compiled code so its memory maps return to the kernel (wrapper#119/#120).

    XLA's CPU JIT maps memory for every compiled executable and never unmaps it; one cohort's
    worth of bucket and M-step shapes reaches vm.max_map_count (65,530 on the HPC).
    """
    import jax

    jax.clear_caches()
    try:
        with open("/proc/self/maps") as maps:
            print(f"    released compiled code after {where} ({sum(1 for _ in maps)} maps)", flush=True)
    except OSError:
        pass


def freeze_run(run, cohorts, subject_index):
    """Every {subject, state} fit of one run, both cohorts."""
    from aind_dynamic_foraging_models.glm_hmm import GLMHMM

    files = run["dir"] / "files" / "outputs" / "glmhmm"
    fit = json.loads((files / "fit.json").read_text())
    cfg, g = fit["config"], fit["global"]
    K, L, emission = int(cfg["num_states"]), int(cfg["n_lags"]), cfg["emission"]
    model = GLMHMM(K, L, emission, NEWTON_STEPS)
    params, props = model.initialize(initial_probs=np.asarray(g["initial_probs"]),
                                     transition_matrix=np.asarray(g["transition_matrix"]),
                                     **{k: np.asarray(v) for k, v in g["emissions"].items()})
    cells = {"subject": [], "state": [], "n_trials": [], "n_engaged": [], "n_sessions": []}
    weights, deltas, not_frozen = [], [], {}
    for cohort, subjects in cohorts.items():
        not_frozen[cohort] = 0
        names = list(subjects)
        for start in range(0, len(names), SUBJECTS_PER_CHUNK):
            chunk = {s: subjects[s] for s in names[start:start + SUBJECTS_PER_CHUNK]}
            paths = viterbi_paths(model, params, chunk, cfg)
            i = 0
            for s, records in chunk.items():
                p = paths[i:i + len(records)]
                i += len(records)
                w, n, n_eng, n_ses, delta = fit_cells(model, params, props, records, p, L, cfg["design"])
                for k in range(K):
                    if n[k] < MIN_FROZEN_TRIALS:
                        not_frozen[cohort] += 1
                        continue
                    cells["subject"].append(subject_index[s])
                    cells["state"].append(k)
                    cells["n_trials"].append(int(n[k]))
                    cells["n_engaged"].append(int(n_eng[k]))
                    cells["n_sessions"].append(int(n_ses[k]))
                    weights.append(w[k])
                    deltas.append(delta)
            release_compiled(f"{cohort} {start + len(chunk)}/{len(names)}")
    record = {
        "group": run["group"], "emission": emission, "num_states": K, "n_lags": L,
        "wandb_run_id": run["wandb_run_id"], "fit_json_sha256": sha256(files / "fit.json"),
        "global_weights": flat_weights(emission, g["emissions"]).round(6).tolist(),
        "global_transition_matrix": np.asarray(g["transition_matrix"]).round(6).tolist(),
        "max_second_step_change": float(max(deltas)) if deltas else None,
        "cells": cells,
        "n_cells_not_frozen": not_frozen,     # {subject, state} cells under MIN_FROZEN_TRIALS
    }
    return record, np.asarray(weights, dtype=np.float32)


def write_outputs(out_dir, subjects, frozen, arrays, runs):
    """frozen_r2.json and the weight files, from every run's record and weights."""
    weight_files = []
    for emission, by_k in arrays.items():
        if by_k:
            path = out_dir / f"frozen_r2_weights_{emission}.npz"
            # float16 (about 3 significant digits) keeps each file under the repo's 10 MiB cap;
            # the float32 parts stay on /allen.
            np.savez_compressed(path, **{k: w.astype(np.float16) for k, w in by_k.items()})
            weight_files.append(path)
    snapshot, build_id = snapshot_source()     # refuses a snapshot other than the runs read
    sources = [snapshot]
    for run in runs.values():
        fit_path = run["dir"] / "files" / "outputs" / "glmhmm" / "fit.json"
        sources += [{"type": "wandb_run", "id": f"{PROJECT}/{run['wandb_run_id']}"},
                    {"type": "file", "id": str(fit_path), "sha256": sha256(fit_path)}]
    meta = build_meta("analysis/freeze_r2.py", WANDB_GROUPS, study_root=HERE.parent, kind="freeze",
                      code=[__file__, HERE / "freeze_round1.py"], sources=sources)
    write_artifact(out_dir / OUT.name, {
        "_meta": meta,
        "data": {"config": "mice_snapshot_scaling", "snapshot": SNAPSHOT,
                 "manifest_sha256": snapshot["sha256"], "build_id": build_id, "ignore_policy": "include",
                 "sessions": "every session (train and eval) of each mouse",
                 "posteriors": "Viterbi under each run's global fit",
                 "prior_sigma": PRIOR_SIGMA, "newton_steps": NEWTON_STEPS,
                 "min_frozen_trials": MIN_FROZEN_TRIALS},
        "subjects": subjects,
        "runs": sorted(frozen, key=lambda r: (r["emission"], r["num_states"])),
    }, outputs=weight_files)
    print(f"wrote {out_dir / OUT.name} and {[p.name for p in weight_files]}")


def main():
    """Freeze every K-grid run, or (for a SLURM array) some runs into part files, or merge parts.

    ``R2_RUNS=hurdle:3,reference:5`` limits the runs; with ``R2_PARTS=<dir>`` each run is written
    there as a part (record JSON + weights .npy) instead of to the frozen files; ``R2_MERGE=1``
    then reads every part in ``R2_PARTS`` and writes the frozen files.
    """
    smoke = os.environ.get("R2_SMOKE") == "1"
    parts = Path(os.environ["R2_PARTS"]) if "R2_PARTS" in os.environ else None
    runs = {key: run for key, run in find_runs().items() if key[0] in WANDB_GROUPS}
    for (group, *_), run in runs.items():
        run["group"] = group
    if os.environ.get("R2_MERGE") == "1":
        frozen, arrays, subjects = [], {"hurdle": {}, "reference": {}}, None
        for path in sorted(parts.glob("*.json")):
            part = json.loads(path.read_text())
            if subjects is not None and part["subjects"] != subjects:
                raise RuntimeError(f"{path.name}: subjects differ from the other parts")
            subjects = part["subjects"]
            frozen.append(part["record"])
            arrays[part["record"]["emission"]][part["record"]["weights_key"]] = np.load(path.with_suffix(".npy"))
        if len(frozen) != len(runs) and not smoke:
            raise RuntimeError(f"{len(frozen)} parts for {len(runs)} runs")
        # A smoke merge writes to an ignored directory, never over the committed files.
        if smoke:
            (HERE / "_cache_smoke").mkdir(exist_ok=True)
            done = {(r["emission"], r["num_states"]) for r in frozen}
            runs = {k: r for k, r in runs.items() if (k[1], k[2]) in done}
        write_outputs(HERE / "_cache_smoke" if smoke else HERE, subjects, frozen, arrays, runs)
        return
    if smoke and "R2_RUNS" not in os.environ:
        runs = {key: run for key, run in runs.items() if key[2] == 3}
    if "R2_RUNS" in os.environ:
        wanted = {(e, int(k)) for e, k in (x.split(":") for x in os.environ["R2_RUNS"].split(","))}
        runs = {key: run for key, run in runs.items() if (key[1], key[2]) in wanted}
    print(f"{len(runs)} runs", flush=True)
    cohorts = load_cohorts()
    if smoke:
        n = int(os.environ.get("R2_SMOKE_MICE", 20))
        cohorts = {c: dict(list(s.items())[:n]) for c, s in cohorts.items()}
    subjects = [{"subject_id": s, "cohort": c, "n_sessions": len(rs),
                 "n_trials": int(sum(len(r["choices"]) for r in rs))}
                for c, ss in cohorts.items() for s, rs in ss.items()]
    subject_index = {s["subject_id"]: i for i, s in enumerate(subjects)}
    print({c: len(s) for c, s in cohorts.items()}, flush=True)

    frozen, arrays = [], {"hurdle": {}, "reference": {}}
    for (group, emission, K, L), run in sorted(runs.items(), key=lambda kv: kv[0]):
        t0 = time.time()
        record, w = freeze_run(run, cohorts, subject_index)
        record["weights_key"] = f"K{K}"
        record["feature_names"] = feature_names(emission, L)
        print(f"  {group} K={K}: {len(w)} cells, max second-step change "
              f"{record['max_second_step_change']:.2e}, {time.time() - t0:.0f} s", flush=True)
        if parts is not None:
            parts.mkdir(parents=True, exist_ok=True)
            np.save(parts / f"{emission}_K{K:02d}.npy", w)
            (parts / f"{emission}_K{K:02d}.json").write_text(json.dumps({"subjects": subjects, "record": record}))
            continue
        arrays[emission][f"K{K}"] = w
        frozen.append(record)
    if parts is None and not smoke:
        write_outputs(HERE, subjects, frozen, arrays, runs)


if __name__ == "__main__":
    main()
