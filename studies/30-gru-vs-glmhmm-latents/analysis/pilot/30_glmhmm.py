"""Stage 3: GLM-HMM posteriors (hurdle, K = 3, L = 10; W&B AIND-disRNN/mice_glmhmm u4j7ukti).

Step 1 (filtered convention): for one held-out mouse, recompute with its refit parameters
(a) p(z_t | y_1..t) and (b) the one-step predictive p(z_t | y_1..t-1) = filtered_{t-1} A
(pi at t = 0), and compare each to the trainer's saved filtered_k in heldout_trials.parquet.

Pilot mice: the GLOBAL parameters for every mouse (one shared set of state labels), on their
eval sessions: logp_left/right/ignore (one-step predictive), pred_k (b), filtered_k (a),
smoothed_k, viterbi. Checks that sum_k pred_k p(y | z=k, x_t) reproduces the predictive.
"""
import json
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

PILOT = Path("/allen/aind/scratch/han.hou/pilots/study30-probe")
OUT = PILOT / "out"
GLM_DIR = Path("/allen/aind/scratch/han.hou/outputs/disrnn/wandb/run-20261027_213114-u4j7ukti/files/outputs/glmhmm")
sys.path.insert(0, str(Path(os.environ["WRAPPER_ROOT"]) / "code"))

import jax  # noqa: E402
from aind_dynamic_foraging_models.glm_hmm import GLMHMM, make_buckets, predictive_log_probs, state_posteriors  # noqa: E402
from aind_dynamic_foraging_models.glm_hmm.design import build_inputs  # noqa: E402
from model_trainers.glmhmm_trainer import _sessions  # noqa: E402


def params_of(model, p):
    params, _ = model.initialize(
        initial_probs=np.asarray(p["initial_probs"]),
        transition_matrix=np.asarray(p["transition_matrix"]),
        **{k: np.asarray(v) for k, v in p["emissions"].items()})
    return params


def class_probs(model, emissions, inputs):
    """p(class | state, x_t), shape (T, K, 3), from the emission component."""
    from aind_dynamic_foraging_models.glm_hmm.emissions import ParamsHurdleEmissions
    from jax.experimental import enable_x64
    with enable_x64():
        p = ParamsHurdleEmissions(np.asarray(emissions["side_weights"]), np.asarray(emissions["engage_weights"]))
        return np.exp(np.asarray(model.emission_component.class_log_probs(p, inputs)))


@partial(jax.jit, static_argnums=0)
def _dynamax_predicted(model, params, labels, inputs):
    """dynamax hmm_filter's predicted_probs p(z_t | y_<t), one session (cross-check)."""
    from dynamax.hidden_markov_model.inference import hmm_filter
    pi, A, ll = model._inference_args(params, labels, inputs)
    return hmm_filter(pi, A, ll).predicted_probs


def posteriors(model, p, records, cfg):
    """Per-session dicts: logp (T,3), filtered, pred, smoothed (T,K), viterbi, pred_dynamax."""
    from jax.experimental import enable_x64
    params = params_of(model, p)
    buckets = make_buckets([(r["choices"], r["rewards"]) for r in records], model.n_lags,
                           cfg["design"], cfg["bucket_size"])
    lp = predictive_log_probs(model, params, buckets)
    post = state_posteriors(model, params, buckets)
    A = np.asarray(p["transition_matrix"])
    pi = np.asarray(p["initial_probs"])
    out = []
    for i, r in enumerate(records):
        f = post[i]["filtered"]
        pred = np.vstack([pi[None, :], f[:-1] @ A])
        x = build_inputs(r["choices"], r["rewards"], model.n_lags)
        with enable_x64():
            pd_ = np.asarray(_dynamax_predicted(model, jax.tree_util.tree_map(
                lambda a: jax.numpy.asarray(a, dtype=jax.numpy.float64), params),
                np.asarray(r["choices"], np.int32), x))
        cp = class_probs(model, p["emissions"], x)
        mix = np.einsum("tk,tkc->tc", pred, cp)
        out.append({"logp": lp[i], "filtered": f, "smoothed": post[i]["smoothed"],
                    "viterbi": post[i]["viterbi"], "pred": pred, "pred_dynamax": pd_,
                    "mix_check": float(np.max(np.abs(mix - np.exp(lp[i]))))})
    return out


def step1(model, fit, cfg):
    hm = json.loads((OUT / "step1_heldout_mouse_meta.json").read_text())
    sid = hm["subject_id"]
    raw = pd.read_parquet(OUT / "step1_heldout_mouse_raw.parquet")
    pq = pd.read_parquet(GLM_DIR / "heldout_trials.parquet")
    pq = pq[pq["subject_id"].astype(str) == sid].copy()
    pq["ses_idx"] = pq["ses_idx"].astype(str)
    res = {"subject_id": sid}
    for split, ids in [("eval", hm["eval_session_ids"]), ("train", hm["train_session_ids"])]:
        recs = _sessions(raw, ids)
        posts = posteriors(model, fit["subjects"][sid], recs, cfg)
        rows = []
        for r, po in zip(recs, posts):
            fr = {"ses_idx": str(r["ses_idx"]), "trial": r["trial"], "choice_mine": r["choices"]}
            for k in range(model.num_states):
                fr[f"a_{k}"] = po["filtered"][:, k]
                fr[f"b_{k}"] = po["pred"][:, k]
                fr[f"bd_{k}"] = po["pred_dynamax"][:, k]
                fr[f"s_{k}"] = po["smoothed"][:, k]
            fr["logp_mine"] = po["logp"][np.arange(len(r["choices"])), r["choices"]]
            rows.append(pd.DataFrame(fr))
        mine = pd.concat(rows, ignore_index=True)
        saved = pq[pq["split"] == split]
        j = saved.merge(mine, on=["ses_idx", "trial"], how="inner", validate="one_to_one")
        assert len(j) == len(saved) == len(mine), (split, len(j), len(saved), len(mine))
        assert (j["choice"].to_numpy() == j["choice_mine"].to_numpy()).all()
        K = model.num_states
        F = j[[f"filtered_{k}" for k in range(K)]].to_numpy()
        lp_saved = j[["logp_left", "logp_right", "logp_ignore"]].to_numpy()[np.arange(len(j)), j["choice"].to_numpy()]
        res[split] = {
            "n_trials": int(len(j)), "n_sessions": int(j["ses_idx"].nunique()),
            "maxabs_saved_vs_a_filtered_incl_t": float(np.max(np.abs(F - j[[f"a_{k}" for k in range(K)]].to_numpy()))),
            "maxabs_saved_vs_b_predictive": float(np.max(np.abs(F - j[[f"b_{k}" for k in range(K)]].to_numpy()))),
            "maxabs_b_numpy_vs_dynamax_predicted": float(np.max(np.abs(j[[f"b_{k}" for k in range(K)]].to_numpy() - j[[f"bd_{k}" for k in range(K)]].to_numpy()))),
            "maxabs_saved_vs_smoothed": float(np.max(np.abs(j[[f"smoothed_{k}" for k in range(K)]].to_numpy() - j[[f"s_{k}" for k in range(K)]].to_numpy()))),
            "maxabs_logp_observed": float(np.max(np.abs(lp_saved - j["logp_mine"].to_numpy()))),
            "mean_abs_a_minus_b": float(np.mean(np.abs(j[[f"a_{k}" for k in range(K)]].to_numpy() - j[[f"b_{k}" for k in range(K)]].to_numpy()))),
        }
    print("step1", json.dumps(res, indent=1), flush=True)
    return res


def pilot(model, fit, cfg):
    ev = pd.read_parquet(OUT / "pilot_eval_raw.parquet")
    order = json.loads((OUT / "pilot_meta.json").read_text())["eval_session_order"]
    frames, mix, dyn = [], [], []
    for n, (sid, sub) in enumerate(ev.groupby("subject_id", sort=False)):
        ids = [x for x in order if x in set(sub["ses_idx"])]
        recs = _sessions(sub, ids)
        posts = posteriors(model, fit["global"], recs, cfg)
        for r, po in zip(recs, posts):
            fr = {"subject_id": sid, "ses_idx": str(r["ses_idx"]), "trial": r["trial"],
                  "glm_choice": r["choices"], "glm_reward": r["rewards"],
                  "logp_left": po["logp"][:, 0], "logp_right": po["logp"][:, 1], "logp_ignore": po["logp"][:, 2],
                  "viterbi": po["viterbi"].astype(np.int8)}
            for k in range(model.num_states):
                fr[f"pred_{k}"] = po["pred"][:, k]
                fr[f"filtered_{k}"] = po["filtered"][:, k]
                fr[f"smoothed_{k}"] = po["smoothed"][:, k]
            frames.append(pd.DataFrame(fr))
            mix.append(po["mix_check"])
            dyn.append(float(np.max(np.abs(po["pred"] - po["pred_dynamax"]))))
        if (n + 1) % 10 == 0:
            jax.clear_caches()
            print("mice done", n + 1, flush=True)
    out = pd.concat(frames, ignore_index=True)
    obs = out[["logp_left", "logp_right", "logp_ignore"]].to_numpy()[np.arange(len(out)), out["glm_choice"].to_numpy()]
    summary = {"n_trials": int(len(out)), "normalized_likelihood": float(np.exp(obs.mean())),
               "max_mix_vs_predictive": float(max(mix)), "max_pred_numpy_vs_dynamax": float(max(dyn))}
    print("pilot glmhmm", summary, flush=True)
    out.to_parquet(OUT / "glmhmm_pilot_posteriors.parquet")
    return summary


def main():
    fit = json.loads((GLM_DIR / "fit.json").read_text())
    cfg = fit["config"]
    model = GLMHMM(int(cfg["num_states"]), int(cfg["n_lags"]), cfg["emission"], cfg["num_newton_steps"])
    res = {"step1": step1(model, fit, cfg)}
    jax.clear_caches()
    res["pilot"] = pilot(model, fit, cfg)
    res["config"] = cfg
    res["global"] = {"initial_probs": fit["global"]["initial_probs"],
                     "transition_matrix": fit["global"]["transition_matrix"]}
    (OUT / "glmhmm_summary.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
