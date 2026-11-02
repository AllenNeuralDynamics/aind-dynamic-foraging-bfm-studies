"""Freeze for r3: one GLM per {subject, state}, under 18 ways of cutting trials into states.

For every hurdle and reference K-grid run (K = 1..10) and both cohorts (the 149 held-out mice
and the 614 training mice, every session of each):
1. Each mouse's round-2 refit (r2): EM from the global fit on the mouse's train sessions, with
   the prior round 2 picked at that K (weights centred on the global fit, SD sigma; transitions
   centred on the global matrix with c pseudo-transitions per row, or round 1's Dirichlet when
   c is off). The run's own code path (``fit_em`` with the trainer's settings), so for the
   held-out mice it must reproduce the refits round 2 saved; the freeze checks that, and that
   every refit's states match the global states one to one (``match_by_weights``).
2. Two labellings of every session: the Viterbi path and smoothed posteriors under the global
   fit (``global``, the same model for every mouse) or under the mouse's own refit (``refit``).
3. Nine cuts per labelling: a trial counts for its Viterbi state only if that state's smoothed
   posterior is at least ``p`` and the trial lies in a run of the same Viterbi state at least
   ``m`` trials long within its session; p in POSTERIOR_MIN, m in RUN_MIN.
4. For each of the 18 settings, subject and state, one MAP fit of that state's emission GLM to
   the kept trials: the package's own Newton M-step with 0/1 trial weights, prior SD 2 centred
   at 0, started from the global state's weights. A state's nine cuts share one M-step call on
   that state's Viterbi trials (a 9-state emission whose states are fitted independently). A second M-step
   from the result measures convergence. Lags are built on the whole session before any trial
   is dropped, so no history crosses a cut.

Setting ``global_p0_m1`` (every trial, global Viterbi) is the first version of r3.

A SLURM array (``freeze_r3.sbatch``) runs one K-grid run per task and writes a part per run to
``R3_PARTS``: the record (JSON) and every setting's float32 weights plus the refit weights
(npz). ``R3_SCORE=1`` then scores each part with the producer's own functions (cluster scores
and plotted coordinates of every setting, on the weights rounded as committed), and
``R3_MERGE=1`` writes ``frozen_r3.json`` and its npz files from the parts. Needs the
/allen mount and the snapshot, so it runs on an HPC compute node:

    WRAPPER_ROOT=<wrapper checkout> DISPATCHER_ROOT=<dispatcher checkout> \\
        make -C studies/14-glmhmm-vs-gru-heldout freeze-r3

``R3_SMOKE=1`` runs the K = 3 run of each emission (or ``R3_RUNS``) on ``R3_SMOKE_MICE`` (20)
mice per cohort; it writes nothing unless ``R3_PARTS`` is set, and a smoke merge writes to the
ignored ``analysis/_cache_smoke/``.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
from freeze_round2 import cell_key  # noqa: E402
from freeze_round2 import find_runs as find_round2_runs  # noqa: E402

WANDB_GROUPS = ["hurdle@20261027-212905", "reference@20261027-213006"]
ROUND2_GROUP = "refit-prior@20261101-130226"
ROUND2 = HERE / "r2_refit_prior.json"    # which round-2 prior the selection score picks per K
OUT = HERE / "frozen_r3.json"
MIN_FROZEN_TRIALS = 50
PRIOR_SIGMA = 2.0                # the {subject, state} GLM's weight prior (round 1's SUBJECT_PRIOR)
NEWTON_STEPS = 12                # per fit (from the global weights, 12 match 30 to < 1e-8 on 16 mice);
                                 # a second fit from the result checks convergence
SUBJECTS_PER_CHUNK = 32          # mice per pass; compiled code is released after each
REFIT_WORKERS = 8
EMISSIONS = ("hurdle", "reference")
LABELLINGS = ("global", "refit")
POSTERIOR_MIN = (0.0, 0.8, 0.95)
RUN_MIN = (1, 10, 30)
CUTS = [(p, m) for p in POSTERIOR_MIN for m in RUN_MIN]
SETTINGS = [(lab, p, m) for lab in LABELLINGS for p, m in CUTS]
REFIT_TOL = 1e-3                 # held-out refits against round 2's saved ones (max |diff|)
N_PCS = 3                        # plotted coordinates: PC1-3 of the anchor's training cells


def setting_key(labelling, p, m):
    return f"{labelling}_p{round(100 * p)}_m{m}"


ANCHOR = setting_key("global", 0.0, 1)   # every trial, global Viterbi: r3's first version


def load_cohorts():
    """Every session of both cohorts, as the trainer's session records, keyed by cohort.

    The loader and the session builder are the trainer's own (``_sessions``), at the wrapper
    and dispatcher commits the K-grid runs used. Each record carries ``split`` (train or eval):
    the refits use the train sessions, as round 2's did.
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
        by_subject = {}
        for split in ("train", "eval"):
            for r in _sessions(bundle.raw, list(bundle.metadata[f"{split}_session_ids"])):
                by_subject.setdefault(r["subject_id"], []).append({**r, "split": split})
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


def load_params(model, block):
    """Exact float64 parameters from a fit.json block (global or one subject's), as the trainer loads them."""
    from jax.experimental import enable_x64

    with enable_x64():   # without it, jnp rounds the saved float64 parameters to float32
        params, props = model.initialize(
            initial_probs=np.asarray(block["initial_probs"], np.float64),
            transition_matrix=np.asarray(block["transition_matrix"], np.float64),
            **{k: np.asarray(v, np.float64) for k, v in block["emissions"].items()})
    return params, props


def buckets_of(records, n_lags, cfg):
    from aind_dynamic_foraging_models.glm_hmm import make_buckets

    return make_buckets([(r["choices"], r["rewards"]) for r in records], n_lags, cfg["design"],
                        cfg["bucket_size"])


def labels_of(model, params, records, cfg):
    """(Viterbi path, smoothed posterior of the Viterbi state) of every session of ``records``."""
    from aind_dynamic_foraging_models.glm_hmm import state_posteriors

    posts = state_posteriors(model, params, buckets_of(records, model.n_lags, cfg))
    out = []
    for i in range(len(records)):
        path = np.asarray(posts[i]["viterbi"])
        out.append((path, np.asarray(posts[i]["smoothed"])[np.arange(len(path)), path]))
    return out


def run_length_per_trial(path):
    """Length of the same-state run each trial of one session lies in."""
    starts = np.flatnonzero(np.r_[True, path[1:] != path[:-1]])
    lengths = np.diff(np.r_[starts, len(path)])
    return np.repeat(lengths, lengths)


def refit_prior(cfg2, global_params):
    """The round-2 run's subject prior, exactly as its trainer builds it."""
    from aind_dynamic_foraging_models.glm_hmm import Prior

    prior = Prior(cfg2["prior_sigma_subject"], cfg2["transition_alpha_subject"], cfg2["stickiness"])
    if cfg2["subject_prior_center"] == "global":
        prior = prior._replace(mean=global_params.emissions)
    if cfg2["transition_strength_subject"] is not None:
        A = np.asarray(global_params.transitions.transition_matrix, np.float64)
        prior = prior._replace(transition_concentration=1.0 + cfg2["transition_strength_subject"] * A)
    return prior


def compiled_m_step(component, props):
    """The emission M-step, compiled once per padded data shape.

    ``m_step`` builds its Newton loop anew on every call, so JAX would trace and compile it
    for every subject and state; under ``jax.jit`` the compiled code is reused for every call
    with the same row-grid sizes (``compress_rows`` pads to a fixed grid).
    """
    import jax

    @jax.jit
    def step(start, stats):
        return component.m_step(start, props, stats, None, PRIOR_SIGMA)[0]

    return step


def pow2_rows(X, targets, weights):
    """Zero-weight rows appended up to a power of two, so few row counts reach the compiler.

    ``compress_rows`` pads to four sizes per octave; with two blocks (hurdle) the combinations
    still compiled anew for most subjects. Zero-weight rows add nothing to the fit.
    """
    n = len(X)
    size = 1 << max(8, (n - 1).bit_length())
    pad = size - n
    return (np.pad(np.asarray(X), ((0, pad), (0, 0))), np.pad(np.asarray(targets), (0, pad)),
            np.pad(np.asarray(weights), ((0, pad), (0, 0))))


def fit_cells(cut_model, starts, step, records, labels, emission, n_lags, design):
    """MAP emission weights of every (cut, state) on one subject's kept trials, for one labelling.

    State by state: the trials on state k's Viterbi path are the only ones with weight in any
    of its cells, so each state's nine cuts are fitted together on those rows alone, as the
    nine states of ``cut_model`` (fitted independently, each started from the global state's
    weights, ``starts[k]``), with ``step`` from ``compiled_m_step``. Returns (weights (9K, D)
    with cell ``j * K + k`` = cut j, state k; n_trials, n_engaged, n_sessions (9K,); max |change|
    of a second M-step from the result).
    """
    from aind_dynamic_foraging_models.glm_hmm import build_inputs
    from aind_dynamic_foraging_models.glm_hmm.emissions import aggregate
    from jax.experimental import enable_x64

    K, J = len(starts), len(CUTS)
    y = np.concatenate([r["choices"] for r in records])
    inputs = np.concatenate([build_inputs(r["choices"], r["rewards"], n_lags, design) for r in records])
    path = np.concatenate([p for p, _ in labels])
    confidence = np.concatenate([c for _, c in labels])
    run = np.concatenate([run_length_per_trial(p) for p, _ in labels])
    session = np.concatenate([np.full(len(r["choices"]), i) for i, r in enumerate(records)])
    cut = np.stack([(confidence >= p) & (run >= m) for p, m in CUTS], 1)        # (T, J)
    component = cut_model.emission_component
    w = np.zeros((J * K, flat_weights(emission, starts[0]).shape[1]))
    delta = 0.0
    for k in range(K):
        rows = np.flatnonzero(path == k)
        if not cut[rows].any():
            continue                                       # no cell of this state has a trial
        with enable_x64():
            blocks = component.prepare_m_step(y[rows], inputs[rows])
            stats = tuple(pow2_rows(b.X, b.targets, aggregate(cut[rows] * 1.0, b)) for b in blocks)
            fitted = step(starts[k], stats)
            again = step(fitted, stats)
        fitted_w = flat_weights(emission, fitted)
        for j in range(J):
            if cut[rows, j].any():
                w[j * K + k] = fitted_w[j]
                delta = max(delta, float(np.abs(flat_weights(emission, again)[j] - fitted_w[j]).max()))
    kept = np.concatenate([cut & (path == k)[:, None] for k in range(K)], 1)  # (T, K*J), state-major
    kept = kept.reshape(len(y), K, J).transpose(0, 2, 1).reshape(len(y), J * K)  # cut-major
    n_trials = kept.sum(0)
    n_engaged = kept[y != 2].sum(0)
    n_sessions = np.array([len(np.unique(session[kept[:, c]])) for c in range(J * K)])
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


def freeze_run(run, round2_run, cohorts, subject_index):
    """Refits, labellings and every {setting, subject, state} fit of one K-grid run, both cohorts."""
    import jax
    from aind_dynamic_foraging_models.glm_hmm import GLMHMM, fit_em
    from aind_dynamic_foraging_models.glm_hmm.align import match_by_weights

    files = run["dir"] / "files" / "outputs" / "glmhmm"
    fit = json.loads((files / "fit.json").read_text())
    cfg, g = fit["config"], fit["global"]
    K, L, emission = int(cfg["num_states"]), int(cfg["n_lags"]), cfg["emission"]
    files2 = round2_run["dir"] / "files" / "outputs" / "glmhmm"
    fit2 = json.loads((files2 / "fit.json").read_text())
    cfg2 = fit2["config"]
    if flat_weights(emission, fit2["global"]["emissions"]).tolist() != flat_weights(emission, g["emissions"]).tolist():
        raise RuntimeError(f"{round2_run['wandb_run_id']} did not load {run['wandb_run_id']}'s global fit")

    model = GLMHMM(K, L, emission, NEWTON_STEPS)              # labelling (no M-step is run on it)
    em_model = GLMHMM(K, L, emission, cfg2["num_newton_steps"])  # the refits, as round 2 ran them
    cut_model = GLMHMM(len(CUTS), L, emission, NEWTON_STEPS)     # one state's nine cuts at once
    global_params, _ = load_params(model, g)
    em_params, em_props = load_params(em_model, g)
    prior = refit_prior(cfg2, em_params)
    starts = []                                                  # global state k, once per cut
    for k in range(K):
        tiled, cut_props = load_params(cut_model, {
            "initial_probs": np.full(len(CUTS), 1.0 / len(CUTS)), "transition_matrix": np.eye(len(CUTS)),
            "emissions": {n: np.repeat(np.asarray(v, np.float64)[k:k + 1], len(CUTS), 0)
                          for n, v in g["emissions"].items()}})
        starts.append(jax.tree_util.tree_map(lambda x: np.asarray(x, np.float64), tiled.emissions))
    G = flat_weights(emission, g["emissions"])
    step = compiled_m_step(cut_model.emission_component, cut_props.emissions)

    settings = {setting_key(*s): {"labelling": s[0], "posterior_min": s[1], "run_min": s[2],
                                  "cells": {"subject": [], "state": [], "n_trials": [], "n_engaged": [],
                                            "n_sessions": []},
                                  "n_cells_not_frozen": {}, "max_second_step_change": 0.0}
                for s in SETTINGS}
    weights = {key: [] for key in settings}
    refit = {"subject": [], "converged": [], "em_iters": [], "matched_identity": [],
             "n_train_sessions": [], "heldout_max_abs_diff": []}
    refit_w, refit_A = [], []

    def one_refit(item):
        s, records = item
        train = [r for r in records if r["split"] == "train"]
        if not train:
            return s, None
        return s, fit_em(em_model, em_params, em_props, buckets_of(train, L, cfg), prior,
                         cfg2["max_em_iters"], cfg2["em_tol"])

    clock = {"refits": 0.0, "labels": 0.0, "cells": 0.0}
    for cohort, subjects in cohorts.items():
        for key in settings:
            settings[key]["n_cells_not_frozen"][cohort] = 0
        names = list(subjects)
        for start in range(0, len(names), SUBJECTS_PER_CHUNK):
            chunk = {s: subjects[s] for s in names[start:start + SUBJECTS_PER_CHUNK]}
            t0 = time.time()
            with ThreadPoolExecutor(REFIT_WORKERS) as pool:
                refits = dict(pool.map(one_refit, chunk.items()))
            t1 = time.time()
            records_all = [r for rs in chunk.values() for r in rs]
            global_labels = labels_of(model, global_params, records_all, cfg)
            clock["refits"] += t1 - t0
            clock["labels"] += time.time() - t1
            i = 0
            for s, records in chunk.items():
                by_labelling = {"global": global_labels[i:i + len(records)]}
                i += len(records)
                result = refits[s]
                if result is None:                  # no train sessions: no refit labelling
                    print(f"    {s}: no train sessions, no refit", flush=True)
                else:
                    w_refit = flat_weights(emission, result.params.emissions)
                    perm, _ = match_by_weights(w_refit, G)
                    diff = None
                    if s in fit2["subjects"]:
                        saved = fit2["subjects"][s]
                        diff = float(max(np.abs(w_refit - flat_weights(emission, saved["emissions"])).max(),
                                         np.abs(np.asarray(result.params.transitions.transition_matrix)
                                                - np.asarray(saved["transition_matrix"])).max()))
                        if diff > REFIT_TOL:
                            raise RuntimeError(f"{s}: refit differs from round 2's by {diff:.3g}")
                    elif cohort == "heldout":
                        raise RuntimeError(f"{s}: held-out mouse missing from round 2's refits")
                    refit["subject"].append(subject_index[s])
                    refit["converged"].append(bool(result.converged))
                    refit["em_iters"].append(int(len(result.log_joint) - 1))
                    refit["matched_identity"].append(bool((perm == np.arange(K)).all()))
                    refit["n_train_sessions"].append(sum(r["split"] == "train" for r in records))
                    refit["heldout_max_abs_diff"].append(diff)
                    refit_w.append(w_refit.astype(np.float32))
                    refit_A.append(np.asarray(result.params.transitions.transition_matrix, np.float32))
                    t1 = time.time()
                    by_labelling["refit"] = labels_of(model, result.params, records, cfg)
                    clock["labels"] += time.time() - t1
                for labelling, labels in by_labelling.items():
                    t1 = time.time()
                    w, n, n_eng, n_ses, delta = fit_cells(cut_model, starts, step, records, labels,
                                                          emission, L, cfg["design"])
                    clock["cells"] += time.time() - t1
                    for j, (p, m) in enumerate(CUTS):
                        entry = settings[setting_key(labelling, p, m)]
                        entry["max_second_step_change"] = max(entry["max_second_step_change"], delta)
                        for k in range(K):
                            c = j * K + k
                            if n[c] < MIN_FROZEN_TRIALS:
                                entry["n_cells_not_frozen"][cohort] += 1
                                continue
                            entry["cells"]["subject"].append(subject_index[s])
                            entry["cells"]["state"].append(k)
                            entry["cells"]["n_trials"].append(int(n[c]))
                            entry["cells"]["n_engaged"].append(int(n_eng[c]))
                            entry["cells"]["n_sessions"].append(int(n_ses[c]))
                            weights[setting_key(labelling, p, m)].append(w[c].astype(np.float32))
            release_compiled(f"{cohort} {start + len(chunk)}/{len(names)}; seconds so far "
                             + ", ".join(f"{k} {v:.0f}" for k, v in clock.items()))
    diffs = [d for d in refit["heldout_max_abs_diff"] if d is not None]
    record = {
        "group": run["group"], "emission": emission, "num_states": K, "n_lags": L,
        "wandb_run_id": run["wandb_run_id"], "fit_json_sha256": sha256(files / "fit.json"),
        "global_weights": G.round(6).tolist(),
        "global_transition_matrix": np.asarray(g["transition_matrix"]).round(6).tolist(),
        "round2": {"group": ROUND2_GROUP, "wandb_run_id": round2_run["wandb_run_id"],
                   "fit_json_sha256": sha256(files2 / "fit.json"),
                   **{k: cfg2[k] for k in ("prior_sigma_subject", "transition_strength_subject",
                                           "transition_alpha_subject", "subject_prior_center",
                                           "max_em_iters", "em_tol", "num_newton_steps")},
                   "heldout_refits_checked": len(diffs),
                   "heldout_refit_max_abs_diff": max(diffs) if diffs else None},
        "refit": refit,
        "settings": settings,
    }
    arrays = {f"W_{key}": np.asarray(w, np.float32).reshape(-1, G.shape[1]) for key, w in weights.items()}
    arrays["refit_W"] = np.asarray(refit_w, np.float32).reshape(-1, K, G.shape[1])
    arrays["refit_A"] = np.asarray(refit_A, np.float32).reshape(-1, K, K)
    return record, arrays


def committed(w):
    """Weights as the repository keeps them (float16), back in float64 for scoring."""
    return np.asarray(w).astype(np.float16).astype(np.float64)


def score_part(stem):
    """Cluster scores and plotted coordinates of every setting of one part.

    Uses the producer's own score functions on the weights rounded as committed. States are
    ranked by the anchor setting's trials (state 1 = most used), so a state keeps its rank and
    colour in every setting; the plotted axes are PC1-3 of the anchor's training cells.
    """
    import r3_state_spread as P

    part = json.loads(stem.with_suffix(".json").read_text())
    arrays = np.load(stem.with_suffix(".npz"))
    subjects, record = part["subjects"], part["record"]
    cohort_of = np.array([s["cohort"] for s in subjects])
    emission, K, names = record["emission"], record["num_states"], record["feature_names"]
    anchor = {k: np.asarray(v) for k, v in record["settings"][ANCHOR]["cells"].items()}
    used = np.bincount(anchor["state"], weights=anchor["n_trials"], minlength=K)
    rank = np.empty(K, int)
    rank[np.argsort(-used, kind="stable")] = np.arange(K)
    fit_axes = (anchor["n_trials"] >= P.MIN_TRIALS) & (cohort_of[anchor["subject"]] == "training")
    mean, axes, share = P.pca_axes(committed(arrays[f"W_{ANCHOR}"])[fit_axes], n=N_PCS)
    side = [i for i, n in enumerate(names) if n.startswith("side:")]
    scores, coords = {}, {}
    for j, (key, entry) in enumerate(record["settings"].items()):
        c = {k: np.asarray(v, int) for k, v in entry["cells"].items()}
        W = committed(arrays[f"W_{key}"])
        keep = c["n_trials"] >= P.MIN_TRIALS if len(W) else np.zeros(0, bool)
        state = rank[c["state"]] if len(W) else c["state"]
        cohort = cohort_of[c["subject"]] if len(W) else c["subject"]
        scores[key] = {}
        for ci, name in enumerate(P.COHORTS):
            m = keep & (cohort == name)
            row = {"n_cells": int(m.sum()), "n_trials_kept": int(c["n_trials"][m].sum()) if len(W) else 0,
                   "n_dropped": int((~keep & (cohort == name)).sum() + entry["n_cells_not_frozen"][name])}
            if m.sum() > K and len(np.unique(state[m])) > 1 or (K == 1 and m.sum() > 1):
                rng = np.random.default_rng([P.SEED, EMISSIONS.index(emission), K, j, ci])
                plane = (W[m] - mean) @ axes[:2].T
                row.update(P.scores(W[m], state[m], c["subject"][m], K, rng, plane, c["n_trials"][m]))
                if emission == "hurdle":
                    row["side_only"] = P.side_only(W[m][:, side], state[m], K)
            scores[key][name] = row
        named = [f for _, f in P.named_features(emission, names, W)] if len(W) else []
        coords[key] = (np.concatenate([(W - mean) @ axes.T, np.stack(named, 1)], 1) if len(W)
                       else np.zeros((0, N_PCS + 6)))
    return {"state_rank": rank.tolist(), "pca_variance_share": share, "pca_mean": mean.tolist(),
            "pca_axes": axes.tolist(), "scores": scores}, coords


def write_outputs(out_dir, parts, runs, round2_runs):
    """frozen_r3.json and the npz files, from every part and its scores.

    Files (float16 weights and coordinates keep each under the repository's 10 MiB cap; the
    float32 parts stay on /allen):
    - ``frozen_r3_cells_<emission>_<labelling>.npz``: every cut's cells per K (subject, state,
      trial counts) and their plotted coordinates (PC1-3, then the six named weights);
    - ``frozen_r3_weights_<emission>_<labelling>.npz``: full weights of the labelling's
      uncut setting (p = 0, m = 1), in the cells file's order;
    - ``frozen_r3_refits_<emission>.npz``: every mouse's round-2 refit (weights, transitions).
    """
    import r3_state_spread as P

    files, frozen, subjects = {}, [], None
    for stem in sorted(parts):
        part = json.loads(stem.with_suffix(".json").read_text())
        if subjects is not None and part["subjects"] != subjects:
            raise RuntimeError(f"{stem.name}: subjects differ from the other parts")
        subjects, record = part["subjects"], part["record"]
        arrays = np.load(stem.with_suffix(".npz"))
        scored = json.loads(stem.with_name(stem.name + "_scores.json").read_text())
        coords = np.load(stem.with_name(stem.name + "_coords.npz"))
        emission, K = record["emission"], record["num_states"]
        for key, entry in record["settings"].items():
            c = entry.pop("cells")
            kind = files.setdefault((f"cells_{emission}_{entry['labelling']}"), {})
            kind[f"K{K}_{key}_subject"] = np.asarray(c["subject"], np.int16)
            kind[f"K{K}_{key}_state"] = np.asarray(c["state"], np.int8)
            for n in ("n_trials", "n_engaged", "n_sessions"):
                kind[f"K{K}_{key}_{n}"] = np.asarray(c[n], np.int32)
            kind[f"K{K}_{key}_coords"] = coords[key].astype(np.float16)
            entry["n_cells_frozen"] = len(c["state"])
            entry["scores"] = scored["scores"][key]
            if entry["posterior_min"] == 0.0 and entry["run_min"] == 1:
                files.setdefault(f"weights_{emission}_{entry['labelling']}", {})[f"K{K}"] = \
                    arrays[f"W_{key}"].astype(np.float16)
        refits = files.setdefault(f"refits_{emission}", {})
        refits[f"K{K}_W"] = arrays["refit_W"].astype(np.float16)
        refits[f"K{K}_A"] = arrays["refit_A"].astype(np.float16)
        record.update({k: scored[k] for k in ("state_rank", "pca_variance_share", "pca_mean", "pca_axes")})
        frozen.append(record)
    paths = []
    for name, content in sorted(files.items()):
        path = out_dir / f"frozen_r3_{name}.npz"
        np.savez_compressed(path, **content)
        paths.append(path)
    snapshot, build_id = snapshot_source()     # refuses a snapshot other than the runs read
    sources = [snapshot]
    for run in [*runs.values(), *round2_runs]:
        fit_path = run["dir"] / "files" / "outputs" / "glmhmm" / "fit.json"
        sources += [{"type": "wandb_run", "id": f"{PROJECT}/{run['wandb_run_id']}"},
                    {"type": "file", "id": str(fit_path), "sha256": sha256(fit_path)}]
    meta = build_meta("analysis/freeze_r3.py", WANDB_GROUPS + [ROUND2_GROUP], study_root=HERE.parent,
                      kind="freeze", code=[__file__, HERE / "freeze_round1.py", HERE / "freeze_round2.py",
                                           Path(P.__file__)],
                      inputs=[ROUND2], sources=sources)
    write_artifact(out_dir / OUT.name, {
        "_meta": meta,
        "data": {"config": "mice_snapshot_scaling", "snapshot": SNAPSHOT,
                 "manifest_sha256": snapshot["sha256"], "build_id": build_id, "ignore_policy": "include",
                 "sessions": "every session (train and eval) of each mouse; refits on train sessions",
                 "labellings": {"global": "Viterbi and smoothed posteriors under each run's global fit",
                                "refit": "the same under each mouse's round-2 refit (r2's pick at that K)"},
                 "cuts": {"posterior_min": list(POSTERIOR_MIN), "run_min": list(RUN_MIN),
                          "rule": "a trial counts for its Viterbi state if that state's smoothed posterior "
                                  ">= posterior_min and its same-state Viterbi run within the session is "
                                  ">= run_min trials long"},
                 "anchor": ANCHOR, "prior_sigma": PRIOR_SIGMA, "newton_steps": NEWTON_STEPS,
                 "min_frozen_trials": MIN_FROZEN_TRIALS, "min_scored_trials": P.MIN_TRIALS,
                 "coords": f"PC1-{N_PCS} of the anchor's training cells (>= {P.MIN_TRIALS} trials), "
                           "then the six named weights of the producer's named_features"},
        "subjects": subjects,
        "runs": sorted(frozen, key=lambda r: (r["emission"], r["num_states"])),
    }, outputs=paths)
    print(f"wrote {out_dir / OUT.name} and {[p.name for p in paths]}")


def main():
    """Freeze K-grid runs into part files (``R3_PARTS``), or merge the parts (``R3_MERGE=1``).

    ``R3_RUNS=hurdle:3,reference:5`` limits the runs.
    """
    smoke = os.environ.get("R3_SMOKE") == "1"
    parts = Path(os.environ["R3_PARTS"]) if "R3_PARTS" in os.environ else None
    runs = {key: run for key, run in find_runs().items() if key[0] in WANDB_GROUPS}
    for (group, *_), run in runs.items():
        run["group"] = group
    picked = json.loads(ROUND2.read_text())["best_by_k"]
    if os.environ.get("R3_SCORE") == "1":
        for stem in sorted({p.with_suffix("") for p in parts.glob("*_K??.json")}):
            emission, K = stem.name.split("_K")
            if "R3_RUNS" in os.environ and f"{emission}:{int(K)}" not in os.environ["R3_RUNS"].split(","):
                continue
            t0 = time.time()
            scored, coords = score_part(stem)
            np.savez(stem.with_name(stem.name + "_coords.npz"), **coords)
            stem.with_name(stem.name + "_scores.json").write_text(json.dumps(scored))
            print(f"  scored {stem.name} in {time.time() - t0:.0f} s", flush=True)
        return
    if os.environ.get("R3_MERGE") == "1":
        stems = sorted({p.with_suffix("") for p in parts.glob("*_K??.json")})
        if len(stems) != len(runs) and not smoke:
            raise RuntimeError(f"{len(stems)} parts for {len(runs)} runs")
        done = {(s.name.split("_K")[0], int(s.name.split("_K")[1])) for s in stems}
        runs = {k: r for k, r in runs.items() if (k[1], k[2]) in done}
        round2_runs = find_round2_runs()
        used = [round2_runs[cell_key({"emission": e, "num_states": K, "prior_sigma_subject": picked[e][str(K)]["sigma"],
                                      "transition_strength_subject": picked[e][str(K)]["c"]})] for e, K in sorted(done)]
        # A smoke merge writes to an ignored directory, never over the committed files.
        out_dir = HERE / "_cache_smoke" if smoke else HERE
        out_dir.mkdir(exist_ok=True)
        write_outputs(out_dir, stems, runs, used)
        return
    if smoke and "R3_RUNS" not in os.environ:
        runs = {key: run for key, run in runs.items() if key[2] == 3}
    if "R3_RUNS" in os.environ:
        wanted = {(e, int(k)) for e, k in (x.split(":") for x in os.environ["R3_RUNS"].split(","))}
        runs = {key: run for key, run in runs.items() if (key[1], key[2]) in wanted}
    round2_runs = find_round2_runs()
    print(f"{len(runs)} runs", flush=True)
    cohorts = load_cohorts()
    if smoke:
        n = int(os.environ.get("R3_SMOKE_MICE", 20))
        cohorts = {c: dict(list(s.items())[:n]) for c, s in cohorts.items()}
    subjects = [{"subject_id": s, "cohort": c, "n_sessions": len(rs),
                 "n_train_sessions": sum(r["split"] == "train" for r in rs),
                 "n_trials": int(sum(len(r["choices"]) for r in rs))}
                for c, ss in cohorts.items() for s, rs in ss.items()]
    subject_index = {s["subject_id"]: i for i, s in enumerate(subjects)}
    print({c: len(s) for c, s in cohorts.items()}, flush=True)

    for (group, emission, K, L), run in sorted(runs.items(), key=lambda kv: kv[0]):
        t0 = time.time()
        pick = picked[emission][str(K)]
        round2_run = round2_runs[cell_key({"emission": emission, "num_states": K,
                                           "prior_sigma_subject": pick["sigma"],
                                           "transition_strength_subject": pick["c"]})]
        if round2_run["wandb_run_id"] != pick["wandb_run_id"]:
            raise RuntimeError(f"{emission} K={K}: round-2 run {round2_run['wandb_run_id']} is not "
                               f"the picked {pick['wandb_run_id']}")
        record, arrays = freeze_run(run, round2_run, cohorts, subject_index)
        record["feature_names"] = feature_names(emission, L)
        anchor = record["settings"][setting_key("global", 0.0, 1)]
        print(f"  {group} K={K}: {len(anchor['cells']['state'])} anchor cells, "
              f"refit diff {record['round2']['heldout_refit_max_abs_diff']}, "
              f"{sum(not x for x in record['refit']['matched_identity'])} refits not aligned, "
              f"{time.time() - t0:.0f} s", flush=True)
        if parts is not None:
            parts.mkdir(parents=True, exist_ok=True)
            np.savez(parts / f"{emission}_K{K:02d}.npz", **arrays)
            (parts / f"{emission}_K{K:02d}.json").write_text(json.dumps({"subjects": subjects, "record": record}))


if __name__ == "__main__":
    main()
