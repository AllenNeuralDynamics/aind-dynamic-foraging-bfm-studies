"""Freeze round 1 of study 14 (lag scan, L = 15 check, hurdle and reference K grids).

Reads each pinned run's output files on /allen (written by the wrapper's GLMHMMTrainer) and the
held-out cohort's trials (through the wrapper's own loader, so the rows are the ones the runs
scored), and writes ``frozen_round1.json``: scores, global parameters, per-subject scores, and,
for the K-grid runs, state summaries and example sessions. ``r1_round_1.py`` reads only that
file.

Needs the /allen mount and the snapshot, so it runs on an HPC compute node:

    WRAPPER_ROOT=<wrapper checkout> DISPATCHER_ROOT=<dispatcher checkout> \\
        make -C studies/14-glmhmm-vs-gru-heldout freeze-round1
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
from _meta import build_meta, write_artifact  # noqa: E402

WANDB_GROUPS = [
    "lag-scan@20261027-200129",
    "lag-scan@20261027-223410",
    "hurdle@20261027-212905",
    "reference@20261027-213006",
]
K_GRID_GROUPS = {"hurdle@20261027-212905", "reference@20261027-213006"}
RUN_ROOT = Path("/allen/aind/scratch/han.hou/outputs/disrnn/wandb")
PROJECT = "AIND-disRNN/mice_glmhmm"
OUT = HERE / "frozen_round1.json"
# The data the runs read: their sweep's data overrides, plus the block reward probabilities
# for the example sessions (extra columns change no row).
DATA_OVERRIDES = [
    "data=mice_snapshot_scaling",
    "data.ignore_policy=include",
    "+data.cols_to_retain=[trial,subject_id,ses_idx,animal_response,earned_reward,"
    "current_stage_actual,reward_probabilityL,reward_probabilityR]",
]
MATURE = {"STAGE_FINAL", "GRADUATED"}
N_EXAMPLES, EXAMPLE_LENGTH = 4, (400, 700)
SUMMARY_PREFIXES = ("glmhmm/", "heldout/")


def sha256(path):
    """sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_identity(files):
    """(group, emission, K, L) of a run directory, or None.

    A run that ended normally has W&B's ``config.yaml``. A run whose process was killed after
    it wrote ``fit.json`` (XLA's code allocator exhausting the kernel's memory-map limit while
    writing posteriors) has no ``config.yaml``; its launch identity is then read from the
    ``+meta.*`` overrides in ``wandb-metadata.json`` and its model from ``fit.json``.
    """
    try:
        config = yaml.safe_load((files / "config.yaml").read_text())
        meta = (config.get("meta") or {}).get("value") or {}
        variant, launch_id = meta.get("variant"), meta.get("launch_id")
    except (OSError, yaml.YAMLError):
        try:
            args = json.loads((files / "wandb-metadata.json").read_text()).get("args", [])
        except (OSError, json.JSONDecodeError):
            return None
        kv = dict(a.lstrip("+").split("=", 1) for a in args if "=" in a)
        variant, launch_id = kv.get("meta.variant"), kv.get("meta.launch_id")
    group = f"{variant}@{launch_id}"
    fit_path = files / "outputs" / "glmhmm" / "fit.json"
    if group not in WANDB_GROUPS or not fit_path.exists():
        return None
    cfg = json.loads(fit_path.read_text())["config"]
    return group, cfg["emission"], int(cfg["num_states"]), int(cfg["n_lags"])


def find_runs():
    """Every run of the pinned groups that wrote fit.json, one per (group, emission, K, L)."""
    runs = {}
    for run_dir in sorted(RUN_ROOT.glob("run-2026102*")):
        files = run_dir / "files"
        key = run_identity(files)
        if key is None:
            continue
        fit = json.loads((files / "outputs" / "glmhmm" / "fit.json").read_text())
        try:
            summary = json.loads((files / "wandb-summary.json").read_text())
            ended_normally = True
        except (OSError, json.JSONDecodeError):
            summary, ended_normally = fit["summary"], False
        if "glmhmm/select/eval_likelihood" not in summary:
            continue
        if key in runs:
            raise RuntimeError(f"two finished runs for {key}: {runs[key]['dir']} and {run_dir}")
        runs[key] = {"dir": run_dir, "summary": summary, "ended_normally": ended_normally,
                     "model": {"emission": key[1], "num_states": key[2], "n_lags": key[3]},
                     "wandb_run_id": run_dir.name.split("-")[-1]}
    return runs


def recompute_posteriors(fit, trials):
    """Smoothed posteriors and Viterbi paths of every held-out trial, from fit.json.

    The trainer's own computation (``GLMHMMTrainer._write_posteriors``): each subject's refit
    parameters, ``state_posteriors`` over its sessions. Done here because runs killed while
    writing ``heldout_trials.parquet`` kept ``fit.json`` but not the parquet. Compiled code is
    released after each run, so this process stays under the memory-map limit.
    """
    import jax
    from aind_dynamic_foraging_models.glm_hmm import GLMHMM, make_buckets, state_posteriors

    cfg = fit["config"]
    K, L = int(cfg["num_states"]), int(cfg["n_lags"])
    model = GLMHMM(K, L, cfg["emission"], cfg["num_newton_steps"])
    frames = []
    for subject, sub in trials.groupby("subject_id", sort=False):
        p = fit["subjects"][subject]
        params, _ = model.initialize(
            initial_probs=np.asarray(p["initial_probs"]),
            transition_matrix=np.asarray(p["transition_matrix"]),
            **{k: np.asarray(v) for k, v in p["emissions"].items()})
        sessions = [g for _, g in sub.groupby("ses_idx", sort=False)]
        buckets = make_buckets([(g["choice"].to_numpy(), g["reward"].to_numpy()) for g in sessions],
                               L, cfg["design"], cfg["bucket_size"])
        posts = state_posteriors(model, params, buckets)       # session index -> posteriors
        for i, g in enumerate(sessions):
            post = posts[i]
            frame = {"subject_id": subject, "ses_idx": g["ses_idx"].to_numpy(),
                     "trial": g["trial"].to_numpy(), "viterbi": np.asarray(post["viterbi"]).astype(np.int8)}
            for k in range(K):
                frame[f"smoothed_{k}"] = np.asarray(post["smoothed"])[:, k]
            frames.append(pd.DataFrame(frame))
    jax.clear_caches()
    return pd.concat(frames, ignore_index=True)


def load_heldout_trials():
    """The held-out cohort's trials, from the loader the runs used."""
    wrapper, dispatcher = Path(os.environ["WRAPPER_ROOT"]), Path(os.environ["DISPATCHER_ROOT"])
    sys.path.insert(0, str(wrapper / "code"))
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    with initialize_config_dir(config_dir=str(dispatcher / "code" / "config"), version_base=None):
        cfg = compose(config_name="config", overrides=DATA_OVERRIDES + ["model=glmhmm"])
    loader = instantiate(cfg.data, split="heldout", multisubject=True, subject_ids=None)
    raw = loader.load().raw
    raw = raw.assign(subject_id=raw["subject_id"].astype(str), ses_idx=raw["ses_idx"].astype(str))
    keep = ["subject_id", "ses_idx", "trial", "animal_response", "earned_reward",
            "reward_probabilityL", "reward_probabilityR"]
    return raw[keep].sort_values(["subject_id", "ses_idx", "trial"]).reset_index(drop=True)


def class_probs(emission, n_lags, emissions, inputs):
    """p(class | state, x) under the global parameters, shape (N, K, 3)."""
    from aind_dynamic_foraging_models.glm_hmm import GLMHMM
    from aind_dynamic_foraging_models.glm_hmm.emissions import (
        ParamsHurdleEmissions,
        ParamsReferenceEmissions,
    )
    from jax.experimental import enable_x64

    K = np.asarray(next(iter(emissions.values()))).shape[0]
    component = GLMHMM(K, n_lags, emission).emission_component
    with enable_x64():
        if emission == "hurdle":
            params = ParamsHurdleEmissions(np.asarray(emissions["side_weights"]),
                                           np.asarray(emissions["engage_weights"]))
        else:
            params = ParamsReferenceEmissions(np.asarray(emissions["weights"]))
        out = []
        for start in range(0, len(inputs), 500_000):
            out.append(np.exp(np.asarray(component.class_log_probs(params, inputs[start:start + 500_000]))))
    return np.concatenate(out)


def build_all_inputs(trials, n_lags):
    """Model inputs of every held-out trial, in ``trials`` order."""
    from aind_dynamic_foraging_models.glm_hmm.design import build_inputs

    blocks = []
    for _, session in trials.groupby(["subject_id", "ses_idx"], sort=False):
        blocks.append(build_inputs(session["choice"].to_numpy(), session["reward"].to_numpy(),
                                   n_lags))
    return np.concatenate(blocks)


def run_lengths(states, sessions):
    """(state, length) of every Viterbi run, never across a session boundary."""
    change = np.r_[True, (states[1:] != states[:-1]) | (sessions[1:] != sessions[:-1])]
    starts = np.flatnonzero(change)
    lengths = np.diff(np.r_[starts, len(states)])
    return states[starts], lengths


def behaviour(frame):
    """Model-free behaviour of a set of trials (choice 0 L, 1 R, 2 ignore)."""
    responded = frame["choice"] != 2
    prev_resp = (frame["prev_choice"] != 2) & frame["prev_choice"].notna()
    stay = responded & prev_resp & (frame["choice"] == frame["prev_choice"])
    both = responded & prev_resp
    better = responded & (frame["pR"] != frame["pL"])
    chose_better = (frame["choice"] == (frame["pR"] > frame["pL"]).astype(int)) & better

    def rate(num, den):
        return float(num.sum() / den.sum()) if den.sum() else None

    return {
        "n_trials": int(len(frame)),
        "p_ignore": float(1 - responded.mean()) if len(frame) else None,
        "p_right_given_respond": rate(responded & (frame["choice"] == 1), responded),
        "p_stay_after_reward": rate(stay & (frame["prev_reward"] == 1), both & (frame["prev_reward"] == 1)),
        "p_stay_after_no_reward": rate(stay & (frame["prev_reward"] == 0), both & (frame["prev_reward"] == 0)),
        "p_choose_better": rate(chose_better, better),
    }


def state_summaries(run, trials, inputs):
    """Occupancy, dwell, predicted and observed behaviour per state, and subject alignment."""
    from aind_dynamic_foraging_models.glm_hmm import match_by_weights

    files = run["dir"] / "files" / "outputs" / "glmhmm"
    K, L, emission = int(run["model"]["num_states"]), int(run["model"]["n_lags"]), run["model"]["emission"]
    fit = json.loads((files / "fit.json").read_text())
    post = recompute_posteriors(fit, trials)
    post = post.sort_values(["subject_id", "ses_idx", "trial"]).reset_index(drop=True)
    if not (post[["subject_id", "ses_idx", "trial"]].values == trials[["subject_id", "ses_idx", "trial"]].values).all():
        raise RuntimeError(f"{run['wandb_run_id']}: posterior rows do not match the loaded trials")
    smoothed = [f"smoothed_{k}" for k in range(K)]
    check = None
    parquet = files / "heldout_trials.parquet"
    if parquet.exists():   # the trainer's own posteriors, where it got to write them
        saved = pd.read_parquet(parquet, columns=["subject_id", "ses_idx", "trial", "viterbi"] + smoothed)
        saved = saved.assign(subject_id=saved["subject_id"].astype(str), ses_idx=saved["ses_idx"].astype(str))
        saved = saved.sort_values(["subject_id", "ses_idx", "trial"]).reset_index(drop=True)
        check = {"max_abs_smoothed_diff": float(np.abs(saved[smoothed].to_numpy() - post[smoothed].to_numpy()).max()),
                 "viterbi_agreement": float((saved["viterbi"].to_numpy() == post["viterbi"].to_numpy()).mean())}
        if check["max_abs_smoothed_diff"] > 1e-3:
            raise RuntimeError(f"{run['wandb_run_id']}: recomputed posteriors differ from the trainer's: {check}")
    gamma = post[smoothed].to_numpy(np.float64)
    viterbi = post["viterbi"].to_numpy()
    emissions = fit["global"]["emissions"]
    probs = class_probs(emission, L, emissions, inputs)           # (N, K, 3)
    weights = gamma / gamma.sum(0, keepdims=True)

    groups = {
        "all": np.ones(len(trials), bool),
        "eval": (trials["split"] == "eval").to_numpy(),
        "mature": trials["stage"].isin(MATURE).to_numpy(),
        "non_mature": (trials["stage"].notna() & ~trials["stage"].isin(MATURE)).to_numpy(),
        "stage_unknown": trials["stage"].isna().to_numpy(),
    }
    occupancy = {name: gamma[mask].mean(0).tolist() for name, mask in groups.items()}
    deciles = np.minimum((trials["position"].to_numpy() * 10).astype(int), 9)
    by_position = [gamma[deciles == d].mean(0).tolist() for d in range(10)]

    sessions = trials["ses_idx"].to_numpy()
    run_state, run_len = run_lengths(viterbi, sessions)
    dwell = []
    for k in range(K):
        lengths = run_len[run_state == k]
        dwell.append({"n_runs": int(len(lengths)),
                      "mean": float(lengths.mean()) if len(lengths) else None,
                      "median": float(np.median(lengths)) if len(lengths) else None})
    states_per_session = (pd.crosstab(sessions, viterbi, normalize="index")
                          .reindex(columns=range(K), fill_value=0.0).to_numpy())
    states = []
    for k in range(K):
        predicted = (weights[:, k, None] * probs[:, k, :]).sum(0)
        states.append({
            "predicted_class_probs": predicted.tolist(),          # left, right, ignore
            "observed_viterbi": behaviour(trials[viterbi == k]),
            "sessions_using": float((states_per_session[:, k] >= 0.1).mean()),
        })

    # How well each subject's refit states line up with the global states.
    if emission == "hurdle":
        g = np.concatenate([np.asarray(emissions["side_weights"]), np.asarray(emissions["engage_weights"])], 1)
    else:
        g = np.asarray(emissions["weights"]).reshape(K, -1)
    identity, distance = 0, []
    for subject in fit["subjects"].values():
        e = subject["emissions"]
        w = (np.concatenate([np.asarray(e["side_weights"]), np.asarray(e["engage_weights"])], 1)
             if emission == "hurdle" else np.asarray(e["weights"]).reshape(K, -1))
        permutation, cost = match_by_weights(w, g)
        identity += int((permutation == np.arange(K)).all())
        distance.append(float(np.sqrt(np.diag(cost).sum() / g.size)))
    return {
        "occupancy": occupancy,
        "occupancy_by_session_decile": by_position,
        "dwell_viterbi": dwell,
        "states": states,
        "posterior_check_against_trainer": check,
        "alignment": {"n_subjects": len(fit["subjects"]),
                      "identity_matched": identity,
                      "rms_weight_distance_median": float(np.median(distance))},
        "_posteriors": post,
    }


def pick_examples(trials):
    """Held-out eval sessions in a mature stage, spread over how often the mouse ignored."""
    sessions = (trials[(trials["split"] == "eval") & trials["stage"].isin(MATURE)]
                .groupby(["subject_id", "ses_idx"])
                .agg(n=("trial", "size"), ignore=("choice", lambda c: float((c == 2).mean()))))
    sessions = sessions[sessions["n"].between(*EXAMPLE_LENGTH)].reset_index()
    sessions = sessions.sort_values(["ignore", "ses_idx"]).reset_index(drop=True)
    picked, subjects = [], set()
    for q in np.linspace(0.1, 0.95, N_EXAMPLES):
        order = np.argsort(np.abs(np.arange(len(sessions)) - q * (len(sessions) - 1)), kind="stable")
        for i in order:
            row = sessions.iloc[i]
            if row["subject_id"] not in subjects:
                picked.append(row["ses_idx"])
                subjects.add(row["subject_id"])
                break
    return picked


def main():
    runs = find_runs()
    print(f"{len(runs)} finished runs in {len(WANDB_GROUPS)} groups")
    raw = load_heldout_trials()
    sample = next(r for k, r in runs.items()
                  if k[0] in K_GRID_GROUPS and (r["dir"] / "files" / "outputs" / "glmhmm" / "heldout_trials.parquet").exists())
    parquet = sample["dir"] / "files" / "outputs" / "glmhmm" / "heldout_trials.parquet"
    base = pd.read_parquet(parquet, columns=["subject_id", "ses_idx", "trial", "split", "choice", "stage"])
    base = base.assign(subject_id=base["subject_id"].astype(str), ses_idx=base["ses_idx"].astype(str))
    trials = base.merge(raw, on=["subject_id", "ses_idx", "trial"], how="left", validate="one_to_one")
    if trials["earned_reward"].isna().any():
        raise RuntimeError("held-out trials missing from the loader's rows")
    if not (trials["choice"] == trials["animal_response"]).all():
        raise RuntimeError("loader choices differ from the scored ones")
    trials = trials.sort_values(["subject_id", "ses_idx", "trial"]).reset_index(drop=True)
    trials["reward"] = trials["earned_reward"].astype(int)
    trials = trials.rename(columns={"reward_probabilityL": "pL", "reward_probabilityR": "pR"})
    trials["stage"] = trials["stage"].where(trials["stage"].notna() & (trials["stage"] != "None"))
    grouped = trials.groupby("ses_idx", sort=False)
    trials["position"] = grouped.cumcount() / grouped["trial"].transform("size")
    trials["prev_choice"] = grouped["choice"].shift(1)
    trials["prev_reward"] = grouped["reward"].shift(1)

    examples = pick_examples(trials)
    example_rows = {s: trials.index[trials["ses_idx"] == s].to_numpy() for s in examples}
    frozen_examples = {
        s: {"subject_id": str(trials.loc[rows[0], "subject_id"]),
            "stage": str(trials.loc[rows[0], "stage"]),
            "choice": trials.loc[rows, "choice"].astype(int).tolist(),
            "reward": trials.loc[rows, "reward"].astype(int).tolist(),
            "pL": trials.loc[rows, "pL"].round(3).tolist(),
            "pR": trials.loc[rows, "pR"].round(3).tolist(),
            "runs": {}}
        for s, rows in example_rows.items()
    }

    inputs_by_lag, frozen_runs, sources = {}, [], []
    for (group, emission, K, L), run in sorted(runs.items(), key=lambda kv: kv[0]):
        files = run["dir"] / "files"
        out_files = files / "outputs" / "glmhmm"
        paths = [p for p in (files / "wandb-summary.json", files / "wandb-metadata.json",
                             out_files / "fit.json", out_files / "heldout_per_subject.csv",
                             out_files / "heldout_trials.parquet") if p.exists()]
        digests = {p.name: sha256(p) for p in paths}
        sources.append({"type": "wandb_run", "id": f"{PROJECT}/{run['wandb_run_id']}"})
        sources += [{"type": "file", "id": str(p), "sha256": digests[p.name]} for p in paths]
        fit = json.loads((out_files / "fit.json").read_text())
        per_subject = pd.read_csv(out_files / "heldout_per_subject.csv")
        summary = {k: v for k, v in run["summary"].items()
                   if k.startswith(SUMMARY_PREFIXES)
                   and (isinstance(v, (int, float, bool, str))
                        or (isinstance(v, list) and all(isinstance(x, (int, float, bool)) for x in v)))}
        record = {
            "group": group, "emission": emission, "num_states": K, "n_lags": L,
            "wandb_run_id": run["wandb_run_id"], "sha256": digests,
            "ended_normally": run["ended_normally"],
            "summary": summary, "config": fit["config"], "global": fit["global"],
            "global_log_joint_trace_len": len(fit["global_log_joint_trace"]),
            "per_subject": per_subject[["subject_id", "n_eval_trials", "eval_log_lik_sum",
                                        "eval_likelihood", "eval_likelihood_LR_engaged",
                                        "eval_likelihood_engage", "em_iters", "converged"]]
            .assign(subject_id=lambda d: d["subject_id"].astype(str)).to_dict(orient="list"),
        }
        if group in K_GRID_GROUPS:
            if L not in inputs_by_lag:
                inputs_by_lag[L] = build_all_inputs(trials, L)
            states = state_summaries(run, trials, inputs_by_lag[L])
            post = states.pop("_posteriors")
            record["state_summary"] = states
            for s, rows in example_rows.items():
                g = post.loc[rows, [f"smoothed_{k}" for k in range(K)]].to_numpy()
                frozen_examples[s]["runs"][run["wandb_run_id"]] = {
                    "smoothed": np.round(g, 3).tolist(),
                    "viterbi": post.loc[rows, "viterbi"].astype(int).tolist(),
                }
            print(f"  {group} K={K}: states frozen")
        frozen_runs.append(record)

    meta = build_meta("analysis/freeze_round1.py", WANDB_GROUPS, study_root=HERE.parent,
                      kind="freeze", code=[__file__], sources=sources)
    write_artifact(OUT, {
        "_meta": meta,
        "data": {"config": "mice_snapshot_scaling", "snapshot": "20260603", "ignore_policy": "include",
                 "split": "heldout", "n_heldout_trials": int(len(trials)),
                 "n_heldout_subjects": int(trials["subject_id"].nunique()),
                 "n_heldout_sessions": int(trials["ses_idx"].nunique())},
        "runs": frozen_runs,
        "examples": frozen_examples,
    })
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
